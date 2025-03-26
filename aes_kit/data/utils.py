# aes_kit/data/utils.py
import numpy as np
from sklearn.model_selection import train_test_split
from ..utils.metrics import ASAP_PROMPT_RANGES # スコア範囲をインポート

def normalize_score(score, prompt_id):
    """スコアを0-1の範囲に正規化する（オプション）"""
    min_score, max_score = ASAP_PROMPT_RANGES.get(prompt_id, (None, None))
    if min_score is not None and max_score is not None and max_score > min_score:
        return (score - min_score) / (max_score - min_score)
    return score # 範囲が不明、または除算できない場合はそのまま返す

def denormalize_score(normalized_score, prompt_id):
    """正規化されたスコアを元の範囲に戻す（オプション）"""
    min_score, max_score = ASAP_PROMPT_RANGES.get(prompt_id, (None, None))
    if min_score is not None and max_score is not None and max_score > min_score:
        return normalized_score * (max_score - min_score) + min_score
    return normalized_score

def round_and_clip_score(score, prompt_id):
    """スコアを丸めて範囲内にクリップする"""
    min_score, max_score = ASAP_PROMPT_RANGES.get(prompt_id, (None, None))
    if min_score is None or max_score is None:
        return round(score) # 範囲不明なら丸めるだけ

    # 予測値は浮動小数点数なので、まず丸める
    rounded_score = np.round(score).astype(int)
    # 範囲内にクリップする
    clipped_score = np.clip(rounded_score, min_score, max_score)
    return clipped_score


# データ分割関数などはここに追加できる (例: promptごとに層化抽出するなど)