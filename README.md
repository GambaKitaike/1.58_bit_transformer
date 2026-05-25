# BitNet b1.58 日本語継続事前学習

Microsoft BitNet b1.58 (2B-4T) を日本語テキストで継続事前学習するためのスクリプト群。  
RTX 5060 (8GB VRAM) 環境向けに、QLoRA + gradient checkpointing で最適化。

## アーキテクチャ

- **ベースモデル**: `microsoft/bitnet-b1.58-2B-4T-bf16` (2.4B params)
- **手法**: QLoRA (4-bit NF4量子化 + LoRAアダプタ)
- **学習バックエンド**: Unsloth (優先) → HuggingFace + PEFT (フォールバック)
- **データ**: llm-jp-corpus-v4 / Wikipedia日本語 / mC4日本語

### BitNet b1.58 について

BitNet b1.58 は重みを3値 {-1, 0, +1} に量子化し、活性化を8bitに量子化するアーキテクチャ。
bf16版はフルプレシジョンのマスターウェイトを格納しており、学習/ファインチューニングに使用する。
QLoRAではベースウェイトを4-bit NF4形式で保持し、LoRAアダプタのみbf16で学習する。

## セットアップ

```bash
# 仮想環境（推奨）
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 依存関係
pip install -r requirements.txt

# flash-attn (オプション、性能向上)
pip install flash-attn --no-build-isolation
```

> **Note**: Unsloth が BitNet をサポートしていない場合、自動的に HuggingFace + PEFT バックエンドにフォールバックします。

## 使い方

### 1. データ準備

```bash
# Wikipedia 日本語 (推奨、最も簡単)
python scripts/prepare_data.py --source wikipedia --max-documents 30000

# mC4 日本語 (ウェブテキスト)
python scripts/prepare_data.py --source mc4 --max-documents 30000

# llm-jp-corpus-v4 (要事前クローン)
git clone https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4.git data/llm-jp-corpus-v4
python scripts/prepare_data.py --source llm-jp-corpus-v4 --local-path data/llm-jp-corpus-v4

# カスタムJSONL
python scripts/prepare_data.py --source local_jsonl --local-path my_data.jsonl
```

### 2. 学習

```bash
# デフォルト設定で学習
python scripts/train.py

# パラメータ上書き
python scripts/train.py --batch-size 1 --grad-accum 16 --lr 5e-6

# WandB無効
python scripts/train.py --no-wandb

# チェックポイントからの再開
python scripts/train.py --resume-from outputs/checkpoint-400
```

### 3. 評価

```bash
# 基本評価（perplexity + テキスト生成サンプル）
python scripts/evaluate.py --model-path outputs/final_model

# ベースモデルとの比較
python scripts/evaluate.py --model-path outputs/final_model --compare-base

# 対話モード
python scripts/evaluate.py --model-path outputs/final_model --interactive
```

## 設定ファイル

`configs/train_config.yaml` で全パラメータを管理:

| カテゴリ | パラメータ | デフォルト | 説明 |
|---------|----------|----------|------|
| model | max_seq_length | 2048 | シーケンス長 |
| model | load_in_4bit | true | QLoRA 4-bit量子化 |
| lora | r | 16 | LoRA rank |
| lora | lora_alpha | 32 | LoRA alpha |
| training | per_device_train_batch_size | 4 | バッチサイズ |
| training | gradient_accumulation_steps | 4 | 勾配累積 |
| training | learning_rate | 1e-5 | 学習率 |
| training | warmup_ratio | 0.05 | ウォームアップ比率 |

## VRAM使用量目安 (RTX 5060, 8GB)

| 設定 | VRAM |
|------|------|
| seq_len=2048, batch=4, QLoRA | ~5-6 GB |
| seq_len=2048, batch=2, QLoRA | ~4-5 GB |
| seq_len=4096, batch=2, QLoRA | ~6-7 GB |

VRAM不足の場合は `batch-size` を減らし `grad-accum` を増やしてください。

## ディレクトリ構成

```
1.58_bit_transformer/
├── configs/
│   └── train_config.yaml    # 学習設定
├── data/
│   └── processed/           # 前処理済みデータ (自動生成)
├── outputs/
│   ├── logs/                # TensorBoard/WandBログ
│   ├── checkpoint-*/        # チェックポイント
│   └── final_model/         # 最終モデル (LoRAアダプタ)
├── scripts/
│   ├── prepare_data.py      # データ準備
│   ├── train.py             # 学習スクリプト
│   └── evaluate.py          # 評価スクリプト
├── requirements.txt         # 依存関係
└── README.md
```

## ライセンス

- ベースモデル (`bitnet-b1.58-2B-4T-bf16`): MIT License
- llm-jp-corpus-v4: 各サブセットごとに異なるライセンス（使用時に確認してください）
