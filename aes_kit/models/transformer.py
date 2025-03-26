# aes_kit/models/transformer.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import List, Dict
from .base_model import BaseModel
from ..utils.logging import logger

class TransformerModel(BaseModel):
    """TransformerベースのAESモデル"""
    def __init__(self, config: dict):
        super().__init__(config)
        model_name = config.get('model_name', 'bert-base-uncased')
        self.mode = config.get('mode', 'cross-prompt') # 'prompt-specific' or 'cross-prompt'
        self.use_prompt_embedding = config.get('use_prompt_embedding', False) and self.mode == 'cross-prompt'
        self.num_prompts = config.get('num_prompts', 8) # ASAPの場合など

        # 事前学習済みモデルのロード
        transformer_config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=transformer_config)
        hidden_size = transformer_config.hidden_size

        # Cross-prompt 用の Prompt Embedding (オプション)
        if self.use_prompt_embedding:
            self.prompt_embedding = nn.Embedding(self.num_prompts + 1, hidden_size) # +1は未知のプロンプト用
            logger.info(f"Using Prompt Embedding for {self.num_prompts} prompts.")
            # prompt_id は 1から始まることが多いので、0番目はパディング/未使用とするか、
            # 使う際に prompt_id - 1 のように調整する必要がある。
            # ここでは、IDが1からnum_promptsまで来ると仮定し、Embeddingは0からnum_prompts-1を使うように調整する。

        # 回帰用のヘッド
        self.regression_head = nn.Linear(hidden_size, 1)

        # 必要な入力キーを定義
        self._required_inputs = ['input_ids', 'attention_mask']
        if self.use_prompt_embedding:
            self._required_inputs.append('prompt_id')

        logger.info(f"TransformerModel initialized with {model_name}")
        logger.info(f"Mode: {self.mode}, Use Prompt Embedding: {self.use_prompt_embedding}")
        logger.info(f"Required inputs: {self._required_inputs}")


    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        # token_type_ids もあれば使う (BERTなど)
        token_type_ids = inputs.get('token_type_ids', None)

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        # [CLS] トークンの表現を取得 (多くのTransformerモデルで一般的)
        pooled_output = transformer_outputs.last_hidden_state[:, 0, :] # (batch_size, hidden_size)

        # Prompt Embedding を使う場合、[CLS]表現に加算する
        if self.use_prompt_embedding:
            prompt_ids = inputs['prompt_id']
            # prompt_id が 1-based の場合、0-based に調整 (例)
            # ここでは prompt_id が 0 から num_prompts-1 の範囲に来ると仮定
            # 実際のデータに合わせて調整が必要
            # prompt_ids_adjusted = prompt_ids - 1 # もし1-basedなら
            prompt_embeds = self.prompt_embedding(prompt_ids) # (batch_size, hidden_size)
            pooled_output = pooled_output + prompt_embeds

        # 回帰ヘッドを通してスコアを予測
        logits = self.regression_head(pooled_output) # (batch_size, 1)

        return logits

    def get_required_inputs(self) -> List[str]:
        return self._required_inputs