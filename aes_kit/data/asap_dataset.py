# aes_kit/data/asap_dataset.py
import pandas as pd
from .base_dataset import BaseDataset
from ..utils.logging import logger

class ASAPDataset(BaseDataset):
    """
    ASAP (Automated Student Assessment Prize) データセット用クラス。
    TSV形式のファイルを読み込む。
    """
    def load_data(self, data_source: str) -> pd.DataFrame:
        """TSVファイルを読み込む"""
        try:
            # encoding='ISO-8859-1' or 'latin1' が必要な場合が多い
            df = pd.read_csv(data_source, sep='\t', encoding='ISO-8859-1')
            logger.info(f"Successfully loaded data from {data_source}")
            # 必須カラムの存在チェック
            required_cols = [self.prompt_id_col, self.essay_col, self.score_col]
            if not all(col in df.columns for col in required_cols):
                 raise ValueError(f"Data source must contain columns: {required_cols}")
            # スコアカラムが複数ある場合への対応（例: domain2_scoreなど）はここでは省略
            return df
        except Exception as e:
            logger.error(f"Error loading data from {data_source}: {e}")
            raise