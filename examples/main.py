# main.py (ライブラリ利用例)
import torch
import pandas as pd
from transformers import AutoTokenizer

from aes_kit.data import ASAPDataset
from aes_kit.models import TransformerModel
from aes_kit.core import Trainer, Predictor, Evaluator
from aes_kit.core.losses import MSELoss
from aes_kit.utils.metrics import ASAP_PROMPT_RANGES

def run_aes_experiment():
    # --- 設定 ---
    config = {
        'model_type': 'transformer',
        'model_name': 'bert-base-uncased', # 小さなモデルでテストするなら 'prajjwal1/bert-tiny' など
        'mode': 'cross-prompt', # 'prompt-specific' or 'cross-prompt'
        'use_prompt_embedding': True, # Cross-promptの場合
        'num_prompts': 8, # ASAPデータセットの場合
        'max_length': 512,
        'batch_size': 8, # メモリに応じて調整
        'eval_batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 1, # 簡単なテストのため1エポック
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        # ↓↓ !!!注意!!! データパスを実際のパスに修正してください ↓↓
        'data_dir': './asap_data', # ASAPデータセットのTSVファイルがあるディレクトリ
        'train_file': 'train.tsv',
        'dev_file': 'dev.tsv',
        'output_dir': './aes_output', # モデルの出力先
        'metric_for_best_model': 'qwk', # ベストモデル選択基準
    }

    print(f"Using device: {config['device']}")

    # --- データ準備 ---
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    train_path = f"{config['data_dir']}/{config['train_file']}"
    dev_path = f"{config['data_dir']}/{config['dev_file']}"

    # データセットが存在するか確認 (簡単なチェック)
    import os
    if not os.path.exists(train_path) or not os.path.exists(dev_path):
        print(f"Error: Data files not found in {config['data_dir']}")
        print("Please download the ASAP dataset and place train.tsv and dev.tsv there.")
        return

    print("Loading datasets...")
    train_dataset = ASAPDataset(
        train_path,
        tokenizer=tokenizer,
        max_length=config['max_length'],
        score_col='domain1_score' # 使用するスコアカラム
    )
    dev_dataset = ASAPDataset(
        dev_path,
        tokenizer=tokenizer,
        max_length=config['max_length'],
        score_col='domain1_score'
    )

    # Prompt-specific の場合 (例: prompt_id=1 のみ)
    # config['mode'] = 'prompt-specific'
    # config['prompt_id'] = 1
    # train_dataset = train_dataset.filter_by_prompt(config['prompt_id'])
    # dev_dataset = dev_dataset.filter_by_prompt(config['prompt_id'])
    # # モデル設定も変更が必要な場合がある (例: prompt embedding 不要)
    # config['use_prompt_embedding'] = False

    # --- モデル初期化 ---
    print("Initializing model...")
    model = TransformerModel(config)

    # --- 学習準備 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    loss_fn = MSELoss()
    evaluator = Evaluator(prompt_score_ranges=ASAP_PROMPT_RANGES)

    # --- Trainerで学習 ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        optimizer=optimizer,
        loss_fn=loss_fn,
        evaluator=evaluator,
        device=config['device'],
        config=config
    )

    print("Starting training...")
    trainer.train(
        num_epochs=config['epochs'],
        batch_size=config['batch_size'],
        eval_batch_size=config['eval_batch_size']
    )

    # --- Predictorで予測 (学習済みモデルを使用) ---
    print("\nInitializing Predictor with the trained model...")
    # final_model.pth または best_model.pth をロードする場合:
    # model_load_path = os.path.join(config['output_dir'], 'final_model.pth')
    # model.load(model_load_path, device=config['device'])

    predictor = Predictor(
        model=model, # 学習後のモデルをそのまま使用
        tokenizer=tokenizer,
        device=config['device'],
        config=config
    )

    print("Predicting sample essays...")
    sample_essays = [
        "This is the first essay I wrote for this prompt.",
        "This second essay might be a bit better structured."
    ]
    # モデルが prompt_id を必要とする場合 (cross-prompt + prompt_embedding)
    sample_prompt_ids = [1, 1] # 例: 両方ともプロンプト1用

    # 生スコアを取得する場合
    raw_scores = predictor.predict(
        sample_essays,
        prompt_ids=sample_prompt_ids if config['use_prompt_embedding'] else None,
        output_raw_scores=True
    )
    print(f"Raw predicted scores: {raw_scores}")

    # 丸めてクリップしたスコアを取得する場合
    final_scores = predictor.predict(
        sample_essays,
        prompt_ids=sample_prompt_ids, # クリップのために prompt_id が必要
        output_raw_scores=False
    )
    print(f"Final (rounded & clipped) predicted scores: {final_scores}")

if __name__ == "__main__":
    # ASAPデータセットをダウンロードし、'asap_data'ディレクトリに配置してください
    # https://www.kaggle.com/c/asap-aes/data
    # train.tsv と dev.tsv が必要です。
    run_aes_experiment()