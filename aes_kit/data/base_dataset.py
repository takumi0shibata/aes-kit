# aes_kit/data/base_dataset.py
from abc import ABC, abstractmethod
import pandas as pd
import torch
from torch.utils.data import Dataset
from ..utils.logging import logger

class BaseDataset(Dataset, ABC):
    """
    AESデータセットの基底クラス。
    """
    def __init__(
        self,
        data_source,
        tokenizer=None,
        feature_extractor=None,
        prompt_id_col='prompt_id',
        essay_col='essay',
        score_col='domain1_score', # ASAPデータセットのデフォルト
        max_length=512,
    ) -> None:
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.prompt_id_col = prompt_id_col
        self.essay_col = essay_col
        self.score_col = score_col
        self.max_length = max_length

        self.data = self.load_data(data_source)
        self.prompt_ids = sorted(self.data[self.prompt_id_col].unique())
        logger.info(f"Loaded dataset with {len(self.data)} samples.")
        logger.info(f"Found prompt IDs: {self.prompt_ids}")

    @abstractmethod
    def load_data(self, data_source) -> pd.DataFrame:
        """データを読み込み、DataFrameとして返す抽象メソッド"""
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        essay = str(item[self.essay_col])
        prompt_id = int(item[self.prompt_id_col])
        score = float(item[self.score_col])

        output = {
            'essay': essay,
            'prompt_id': prompt_id,
            'score': score
        }

        # トークナイズ
        if self.tokenizer:
            # 注意: Tokenizer によっては prompt を一緒に入れるなどの処理が必要になる場合がある
            encoding = self.tokenizer(
                essay,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt' # PyTorch Tensorを返す
            )
            # .squeeze(0) でバッチ次元を削除
            output['input_ids'] = encoding['input_ids'].squeeze(0)
            output['attention_mask'] = encoding['attention_mask'].squeeze(0)
            if 'token_type_ids' in encoding: # BERTなどの場合
                 output['token_type_ids'] = encoding['token_type_ids'].squeeze(0)

        # 特徴量抽出
        if self.feature_extractor:
            # prompt_info など、追加情報が必要な場合がある
            features = self.feature_extractor.extract(essay)
            output['features'] = torch.tensor(features, dtype=torch.float32) # Tensorに変換

        return output

    def get_prompt_ids(self) -> list[int]:
        """データセット内のプロンプトIDリストを返す"""
        return self.prompt_ids

    def filter_by_prompt(self, prompt_id: int):
        """特定のプロンプトIDでデータをフィルタリングした新しいインスタンスを返す"""
        if prompt_id not in self.prompt_ids:
            raise ValueError(f"Prompt ID {prompt_id} not found in dataset.")

        filtered_data = self.data[self.data[self.prompt_id_col] == prompt_id].copy()

        # 新しいインスタンスを作成して返す (効率は良くないがシンプル)
        # より効率的にするには、indexを保持するなどの工夫が必要
        new_dataset = self.__class__.__new__(self.__class__) # インスタンスを生成
        # 親クラスの__init__は呼ばず、属性をコピー
        for k, v in self.__dict__.items():
            setattr(new_dataset, k, v)
        # データだけ差し替え
        new_dataset.data = filtered_data
        new_dataset.prompt_ids = [prompt_id]
        logger.info(f"Filtered dataset for prompt ID {prompt_id}. New size: {len(new_dataset.data)}")
        return new_dataset