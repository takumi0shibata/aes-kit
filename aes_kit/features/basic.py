# aes_kit/features/basic.py
import numpy as np
from .base_extractor import BaseFeatureExtractor
# nltkやspacyを使う場合はここでインポート
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# nltk.download('punkt') # 初回実行時

class BasicFeatureExtractor(BaseFeatureExtractor):
    """基本的なテキスト特徴量を抽出するクラス"""
    def extract(self, text: str, prompt_info=None) -> np.ndarray:
        # 簡単な例: 文字数と単語数
        char_count = len(text)
        word_count = len(text.split())
        # sent_count = len(sent_tokenize(text)) # nltk使用例
        # avg_word_len = char_count / word_count if word_count > 0 else 0 # nltk使用例

        # より多くの特徴量を追加できる
        # 例: 語彙の多様性 (unique words / total words), 平均文長など

        features = np.array([char_count, word_count], dtype=np.float32)
        return features

    def feature_names(self) -> list[str]:
        return ['char_count', 'word_count']

# 必要に応じて LinguisticFeatureExtractor など、より高度なクラスを追加