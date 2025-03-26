# aes_kit/features/base_extractor.py
from abc import ABC, abstractmethod
import numpy as np

class BaseFeatureExtractor(ABC):
    """特徴量抽出器の基底クラス"""
    @abstractmethod
    def extract(self, text: str, prompt_info=None) -> np.ndarray:
        """
        テキストから特徴量を抽出し、Numpy配列として返す。
        prompt_info: プロンプト固有の情報（テキストなど）が必要な場合に使用。
        """
        raise NotImplementedError

    @abstractmethod
    def feature_names(self) -> list[str]:
        """抽出される特徴量の名前リストを返す"""
        raise NotImplementedError