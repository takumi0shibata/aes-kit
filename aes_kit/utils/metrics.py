# aes_kit/utils/metrics.py
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_squared_error, mean_absolute_error

def quadratic_weighted_kappa(y_true, y_pred, min_rating=None, max_rating=None):
    """
    Quadratic Weighted Kappa (QWK) を計算する。
    y_true, y_pred は整数スコアのリストまたはNumpy配列。
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    if min_rating is None:
        min_rating = min(np.min(y_true), np.min(y_pred))
    if max_rating is None:
        max_rating = max(np.max(y_true), np.max(y_pred))

    # スコアが範囲外の場合、クリップする（必須ではないが一般的）
    y_true = np.clip(y_true, min_rating, max_rating)
    y_pred = np.clip(y_pred, min_rating, max_rating)

    return cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=list(range(min_rating, max_rating + 1)))

# ASAP データセットの公式スコア範囲 (参考)
ASAP_PROMPT_RANGES = {
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60)
}

def get_score_range(prompt_id):
    """プロンプトIDに対応するスコア範囲を取得"""
    return ASAP_PROMPT_RANGES.get(prompt_id, (None, None))