# aes_kit/core/trainer.py
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm # 進捗表示
import numpy as np
import os
from typing import Dict, Optional

from ..models.base_model import BaseModel
from .evaluator import Evaluator
from ..utils.logging import logger

class Trainer:
    """モデルの学習を実行するクラス"""
    def __init__(
        self,
        model: BaseModel,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        optimizer: torch.optim.Optimizer = None,
        loss_fn: torch.nn.Module = None,
        evaluator: Optional[Evaluator] = None,
        device: str = 'cpu',
        config: Optional[dict] = None
    ) -> None:
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.device = device
        self.config = config if config else {}

        self.output_dir = self.config.get('output_dir', 'aes_output')
        os.makedirs(self.output_dir, exist_ok=True)

        # デフォルト設定
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.get('learning_rate', 1e-4))
        if self.loss_fn is None:
            self.loss_fn = torch.nn.MSELoss()
        if self.evaluator is None:
            self.evaluator = Evaluator() # デフォルト設定で初期化

        self.required_inputs = self.model.get_required_inputs()
        logger.info(f"Trainer initialized. Device: {self.device}")
        logger.info(f"Model requires inputs: {self.required_inputs}")
        logger.info(f"Output directory: {self.output_dir}")

    def _prepare_batch(self, batch: dict) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """バッチデータをモデル入力形式に整形し、デバイスに転送する"""
        inputs = {}
        for key in self.required_inputs:
            if key in batch:
                # GPUに送る前に dtype を確認・調整する必要がある場合も
                inputs[key] = batch[key].to(self.device)
            else:
                raise ValueError(f"Model requires input '{key}', but it was not found in the batch.")

        # ターゲット (スコア) をデバイスに転送
        targets = batch['score'].float().to(self.device) # スコアはfloat想定
        if targets.ndim == 1:
             targets = targets.unsqueeze(1) # (batch_size, 1) の形状にする

        return inputs, targets

    def train_epoch(self, dataloader: DataLoader) -> float:
        """1エポックの学習を実行"""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc="Training Epoch", leave=False)

        for batch in progress_bar:
            self.optimizer.zero_grad()
            inputs, targets = self._prepare_batch(batch)

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Training Epoch finished. Average Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self, dataloader: DataLoader) -> dict:
        """評価データセットでモデルを評価"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_prompt_ids = []
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch in progress_bar:
                inputs, targets = self._prepare_batch(batch)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                if 'prompt_id' in batch: # prompt_id があれば収集
                    all_prompt_ids.extend(batch['prompt_id'].cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Evaluation finished. Average Loss: {avg_loss:.4f}")

        eval_metrics = {'eval_loss': avg_loss}
        if self.evaluator:
            predictions_np = np.array(all_predictions)
            targets_np = np.array(all_targets)
            prompt_ids_np = np.array(all_prompt_ids) if all_prompt_ids else None

            # 全体メトリクス
            overall_metrics = self.evaluator.compute_metrics(
                predictions=predictions_np,
                targets=targets_np,
                prompt_ids=prompt_ids_np
            )
            eval_metrics.update(overall_metrics)
            logger.info(f"Overall Evaluation Metrics: {overall_metrics}")

            # プロンプト別メトリクス (prompt_idsがある場合)
            if prompt_ids_np is not None and len(np.unique(prompt_ids_np)) > 1 :
                per_prompt_metrics = self.evaluator.compute_metrics_per_prompt(
                    predictions=predictions_np,
                    targets=targets_np,
                    prompt_ids=prompt_ids_np
                )
                eval_metrics['per_prompt'] = per_prompt_metrics
                # ログ表示用に整形
                for pid, metrics in per_prompt_metrics.items():
                    logger.info(f"  Prompt {pid} Metrics: {metrics}")

        return eval_metrics

    def train(self, num_epochs: int, batch_size: int, eval_batch_size: Optional[int] = None):
        """学習ループ全体を実行"""
        logger.info("Starting training...")
        if eval_batch_size is None:
            eval_batch_size = batch_size

        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=eval_batch_size) if self.eval_dataset else None

        best_eval_metric = -float('inf') # 例: QWKを最大化する場合
        metric_to_monitor = self.config.get('metric_for_best_model', 'qwk')

        for epoch in range(num_epochs):
            logger.info(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            train_loss = self.train_epoch(train_dataloader)

            if eval_dataloader and self.evaluator:
                eval_results = self.evaluate(eval_dataloader)

                # ベストモデルの保存 (例: QWKに基づいて)
                current_metric = eval_results.get(metric_to_monitor)
                if current_metric is not None and current_metric > best_eval_metric:
                    best_eval_metric = current_metric
                    save_path = os.path.join(self.output_dir, "best_model.pth")
                    self.model.save(save_path)
                    logger.info(f"New best model saved to {save_path} (Metric: {metric_to_monitor}={best_eval_metric:.4f})")

        logger.info("Training finished.")
        # 最終モデルの保存
        final_save_path = os.path.join(self.output_dir, "final_model.pth")
        self.model.save(final_save_path)
        logger.info(f"Final model saved to {final_save_path}")