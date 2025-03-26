# aes_kit/core/predictor.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Optional, Union
from tqdm.auto import tqdm

from ..models.base_model import BaseModel
from ..features.base_extractor import BaseFeatureExtractor
from ..data.utils import round_and_clip_score
from ..utils.logging import logger

class PredictDataset(Dataset):
    """予測用の一時的なデータセット"""
    def __init__(self,
                 essays: List[str],
                 prompt_ids: Optional[List[int]] = None,
                 tokenizer=None,
                 max_length=512,
                 feature_extractor: Optional[BaseFeatureExtractor] = None,
                 required_inputs: List[str] = ['input_ids', 'attention_mask']):
        self.essays = essays
        self.prompt_ids = prompt_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature_extractor = feature_extractor
        self.required_inputs = required_inputs

        if prompt_ids is not None and len(essays) != len(prompt_ids):
             raise ValueError("Length mismatch between essays and prompt_ids")
        if 'prompt_id' in required_inputs and prompt_ids is None:
             raise ValueError("Model requires 'prompt_id', but it was not provided.")

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, idx):
        essay = self.essays[idx]
        output = {'essay': essay}

        if self.prompt_ids:
            output['prompt_id'] = self.prompt_ids[idx] # prompt_id を int のまま保持

        # トークナイズ (必要なら)
        if self.tokenizer and ('input_ids' in self.required_inputs or 'attention_mask' in self.required_inputs):
            encoding = self.tokenizer(
                essay,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            output['input_ids'] = encoding['input_ids'].squeeze(0)
            output['attention_mask'] = encoding['attention_mask'].squeeze(0)
            if 'token_type_ids' in encoding:
                 output['token_type_ids'] = encoding['token_type_ids'].squeeze(0)

        # 特徴量抽出 (必要なら)
        if self.feature_extractor and 'features' in self.required_inputs:
             features = self.feature_extractor.extract(essay)
             output['features'] = torch.tensor(features, dtype=torch.float32)

        # 不要なキーを削除（DataLoaderでエラーにならないように）
        final_output = {}
        for key in self.required_inputs + ['prompt_id']: # prompt_idは後処理で使う可能性があるので残す
             if key in output:
                 final_output[key] = output[key]

        return final_output


class Predictor:
    """学習済みモデルを使って予測を行うクラス"""
    def __init__(self,
                 model: BaseModel,
                 tokenizer = None, # モデルがテキスト入力を必要とする場合
                 device: str = 'cpu',
                 config: Optional[dict] = None,
                 feature_extractor: Optional[BaseFeatureExtractor] = None): # モデルが特徴量を必要とする場合
        self.model = model.to(device).eval() # 評価モードにしてデバイスへ
        self.tokenizer = tokenizer
        self.device = device
        self.config = config if config else {}
        self.feature_extractor = feature_extractor
        self.required_inputs = self.model.get_required_inputs()
        self.max_length = self.config.get('max_length', 512)

        logger.info(f"Predictor initialized. Device: {self.device}")
        logger.info(f"Model requires inputs: {self.required_inputs}")


    def predict(self,
                essays: Union[str, List[str]],
                prompt_ids: Optional[Union[int, List[int]]] = None,
                batch_size: int = 32,
                output_raw_scores: bool = False,
                score_ranges: Optional[Dict[int, tuple]] = None) -> Union[np.ndarray, float, int]:
        """
        エッセイのスコアを予測する。

        Args:
            essays: 予測するエッセイ（単一の文字列または文字列のリスト）。
            prompt_ids: 各エッセイに対応するプロンプトID（単一のintまたはintのリスト）。
                        モデルがプロンプトIDを必要とする場合や、スコアのクリップに必要。
            batch_size: 予測時のバッチサイズ。
            output_raw_scores: Trueの場合、モデルの生の出力を返す。Falseの場合、丸めてクリップした整数スコアを返す。
            score_ranges: スコアのクリップに使用するプロンプトIDごとの範囲。Noneの場合、デフォルト(ASAP)を使用。

        Returns:
            予測スコア（単一の数値またはNumpy配列）。
        """
        if isinstance(essays, str):
            essays = [essays]
            if prompt_ids is not None and isinstance(prompt_ids, int):
                prompt_ids = [prompt_ids]

        dataset = PredictDataset(
            essays=essays,
            prompt_ids=prompt_ids,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            feature_extractor=self.feature_extractor,
            required_inputs=self.required_inputs
        )

        # prompt_idを別途保持 (DataLoaderがTensorに変換してしまうため)
        original_prompt_ids = dataset.prompt_ids

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_predictions = []
        progress_bar = tqdm(dataloader, desc="Predicting", leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                inputs = {}
                for key in self.required_inputs:
                     if key in batch:
                         inputs[key] = batch[key].to(self.device)
                     else:
                         # PredictDataset でチェック済みのはずだが念のため
                         raise ValueError(f"Required input '{key}' missing during prediction.")

                outputs = self.model(inputs)
                all_predictions.extend(outputs.cpu().numpy().flatten())

        predictions_np = np.array(all_predictions)

        if output_raw_scores:
            logger.info("Outputting raw prediction scores.")
            return predictions_np[0] if len(predictions_np) == 1 and isinstance(essays, str) else predictions_np
        else:
            # スコアを丸めてクリップ
            if original_prompt_ids is None:
                logger.warning("prompt_ids not provided for score rounding/clipping. Rounding only.")
                processed_scores = np.round(predictions_np).astype(int)
            else:
                processed_scores = np.array([
                    round_and_clip_score(pred, pid)
                    for pred, pid in zip(predictions_np, original_prompt_ids)
                ])
            logger.info("Outputting rounded and clipped scores.")
            return processed_scores[0] if len(processed_scores) == 1 and isinstance(essays, str) else processed_scores