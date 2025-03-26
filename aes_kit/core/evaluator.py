# aes_kit/core/evaluator.py
import numpy as np
from typing import List, Dict, Optional
from ..utils.metrics import quadratic_weighted_kappa, mean_squared_error, mean_absolute_error, ASAP_PROMPT_RANGES
from ..data.utils import round_and_clip_score # 丸めとクリップ関数
from ..utils.logging import logger

class Evaluator:
    """評価指標を計算するクラス"""
    def __init__(self,
                 prompt_score_ranges: Optional[Dict[int, tuple]] = None,
                 default_metrics: List[str] = ['qwk', 'mse', 'mae']):
        """
        Args:
            prompt_score_ranges: プロンプトIDごとのスコア範囲 (min, max) の辞書。
                                QWK計算やスコアのクリップに使用。
            default_metrics: デフォルトで計算する指標のリスト。
        """
        self.prompt_score_ranges = prompt_score_ranges if prompt_score_ranges else ASAP_PROMPT_RANGES
        self.default_metrics = default_metrics
        logger.info(f"Evaluator initialized. Score ranges: {self.prompt_score_ranges}")

    def compute_metrics(self,
                        predictions: np.ndarray,
                        targets: np.ndarray,
                        prompt_ids: Optional[np.ndarray] = None,
                        metric_list: Optional[List[str]] = None) -> Dict[str, float]:
        """
        予測値と目標値から評価指標を計算する。

        Args:
            predictions: モデルの予測スコア (floatの配列)。
            targets: 正解スコア (float or int の配列)。
            prompt_ids: 各サンプルに対応するプロンプトIDの配列 (QWKやクリップで使用)。
            metric_list: 計算する指標のリスト。Noneならデフォルトを使用。

        Returns:
            指標名をキー、計算結果を値とする辞書。
        """
        if metric_list is None:
            metric_list = self.default_metrics

        results = {}
        targets = np.array(targets)
        predictions = np.array(predictions)

        if prompt_ids is not None:
            prompt_ids = np.array(prompt_ids)
            if len(predictions) != len(prompt_ids) or len(targets) != len(prompt_ids):
                 raise ValueError("Length mismatch between predictions, targets, and prompt_ids")
            # プロンプトごとにスコアを丸めてクリップ
            processed_preds_for_qwk = np.array([
                round_and_clip_score(pred, pid)
                for pred, pid in zip(predictions, prompt_ids)
            ])
            processed_targets_for_qwk = np.array([
                round_and_clip_score(tgt, pid) # ターゲットも念のため
                for tgt, pid in zip(targets, prompt_ids)
            ])
            # QWKの全体計算には、全プロンプトを通したmin/maxが必要
            overall_min = min(r[0] for r in self.prompt_score_ranges.values() if r[0] is not None)
            overall_max = max(r[1] for r in self.prompt_score_ranges.values() if r[1] is not None)

        else:
            # prompt_idがない場合、QWK計算のための範囲が不明瞭
            # 回帰指標のみ計算するか、全体で丸めるかの判断が必要
            # ここでは丸めるだけにする
            logger.warning("prompt_ids not provided. QWK might be inaccurate or score ranges assumed.")
            processed_preds_for_qwk = np.round(predictions).astype(int)
            processed_targets_for_qwk = np.round(targets).astype(int)
            overall_min, overall_max = None, None # 不明

        for metric in metric_list:
            try:
                if metric == 'qwk':
                    if prompt_ids is None and (overall_min is None or overall_max is None):
                         logger.warning("Cannot compute QWK accurately without prompt_ids or overall score range.")
                         results[metric] = np.nan
                    else:
                        # prompt_ids がある場合は、それに基づいて処理された値を使う
                        results[metric] = quadratic_weighted_kappa(
                            processed_targets_for_qwk,
                            processed_preds_for_qwk,
                            min_rating=overall_min,
                            max_rating=overall_max
                        )
                elif metric == 'mse':
                    results[metric] = mean_squared_error(targets, predictions)
                elif metric == 'mae':
                    results[metric] = mean_absolute_error(targets, predictions)
                # 他の指標も追加可能
            except Exception as e:
                logger.error(f"Error computing metric '{metric}': {e}")
                results[metric] = np.nan # エラー時は NaN

        return results

    def compute_metrics_per_prompt(self,
                                   predictions: np.ndarray,
                                   targets: np.ndarray,
                                   prompt_ids: np.ndarray,
                                   metric_list: Optional[List[str]] = None) -> Dict[int, Dict[str, float]]:
        """プロンプトごとに評価指標を計算する"""
        if metric_list is None:
            metric_list = self.default_metrics

        unique_prompts = sorted(np.unique(prompt_ids))
        per_prompt_results = {}

        for pid in unique_prompts:
            mask = (prompt_ids == pid)
            prompt_preds = predictions[mask]
            prompt_targets = targets[mask]

            if len(prompt_preds) == 0:
                continue

            min_r, max_r = self.prompt_score_ranges.get(pid, (None, None))

            # プロンプト固有のスコア範囲で処理
            processed_preds_for_qwk = round_and_clip_score(prompt_preds, pid)
            processed_targets_for_qwk = round_and_clip_score(prompt_targets, pid) # 念のため

            prompt_metrics = {}
            for metric in metric_list:
                 try:
                    if metric == 'qwk':
                         if min_r is None or max_r is None:
                             logger.warning(f"Score range for prompt {pid} unknown. Skipping QWK.")
                             prompt_metrics[metric] = np.nan
                         else:
                             prompt_metrics[metric] = quadratic_weighted_kappa(
                                 processed_targets_for_qwk,
                                 processed_preds_for_qwk,
                                 min_rating=min_r,
                                 max_rating=max_r
                             )
                    elif metric == 'mse':
                         prompt_metrics[metric] = mean_squared_error(prompt_targets, prompt_preds)
                    elif metric == 'mae':
                         prompt_metrics[metric] = mean_absolute_error(prompt_targets, prompt_preds)
                 except Exception as e:
                     logger.error(f"Error computing metric '{metric}' for prompt {pid}: {e}")
                     prompt_metrics[metric] = np.nan
            per_prompt_results[pid] = prompt_metrics

        return per_prompt_results