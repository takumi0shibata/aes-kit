# aes_kit/models/base_model.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import List, Dict

class BaseModel(nn.Module, ABC):
    """AESモデルの基底クラス"""
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        モデルの順伝播。
        inputs: バッチデータを含む辞書。キーは 'input_ids', 'attention_mask', 'features' など。
        returns: 予測スコアのテンソル (バッチサイズ, 1)
        """
        raise NotImplementedError

    @abstractmethod
    def get_required_inputs(self) -> List[str]:
        """
        モデルが forward メソッドで必要とする入力辞書のキーのリストを返す。
        例: ['input_ids', 'attention_mask']
        """
        raise NotImplementedError

    def save(self, path: str):
        """モデルの状態辞書を保存"""
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = 'cpu'):
        """モデルの状態辞書を読み込み"""
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval() # 評価モードにする