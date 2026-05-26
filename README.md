# BitNet b1.58 日本語継続事前学習

Microsoft BitNet b1.58 (2B-4T) を日本語テキストで継続事前学習するためのスクリプト群。  
RTX 5060 (8GB VRAM) 環境向けに、LoRA + gradient checkpointing で最適化。

## アーキテクチャ

- **ベースモデル**: `microsoft/bitnet-b1.58-2B-4T-bf16` (2.4B params)
- **手法**: LoRA (bf16ベース + LoRAアダプタ)
- **学習バックエンド**: HuggingFace transformers + PEFT
- **データ**: llm-jp-corpus-v4 / Wikipedia日本語 / mC4日本語

### BitNet b1.58 について

BitNet b1.58 は重みを3値 {-1, 0, +1} に量子化し、活性化を8bitに量子化するアーキテクチャ。
bf16版はフルプレシジョンのマスターウェイトを格納しており、学習/ファインチューニングに使用する。
BitNetは独自の量子化設定（BitNetQuantConfig）を持つため、bitsandbytesによる外部4-bit量子化（QLoRA）は不可。
ベースウェイトをbf16のまま保持し、LoRAアダプタのみ学習する構成を採用。

## セットアップ

```bash
# 仮想環境（推奨）
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 依存関係
pip install -r requirements.txt
```

> **Note**: BitNetアーキテクチャはUnsloth非対応のため、HuggingFace transformers + PEFT で学習します。

## 使い方

### 1. データ準備

```bash
# Wikipedia 日本語 (推奨、最も簡単)
python scripts/prepare_data.py --source wikipedia --max-documents 12000

# mC4 日本語 (ウェブテキスト)
python scripts/prepare_data.py --source mc4 --max-documents 12000

# llm-jp-corpus-v4 (要事前クローン)
git clone https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4.git data/llm-jp-corpus-v4
python scripts/prepare_data.py --source llm-jp-corpus-v4 --local-path data/llm-jp-corpus-v4

# カスタムJSONL
python scripts/prepare_data.py --source local_jsonl --local-path my_data.jsonl
```

### 2. 学習

```bash
# デフォルト設定で学習
python scripts/train.py --no-wandb

# パラメータ上書き
python scripts/train.py --batch-size 1 --grad-accum 16 --lr 5e-6

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

| カテゴリ | パラメータ | 値 | 説明 |
|---------|----------|----------|------|
| model | max_seq_length | 1024 | シーケンス長 |
| lora | r | 8 | LoRA rank |
| lora | lora_alpha | 16 | LoRA alpha |
| training | per_device_train_batch_size | 1 | バッチサイズ |
| training | gradient_accumulation_steps | 16 | 勾配累積（実効batch=16） |
| training | learning_rate | 1e-5 | 学習率 |
| data | max_documents | 12000 | 学習文書数 |

## VRAM使用量目安 (RTX 5060, 8GB)

| 設定 | 専用VRAM | 備考 |
|------|----------|------|
| seq_len=1024, batch=1, LoRA r=8 | ~7.6 GB | 本番設定（推奨） |
| seq_len=2048, batch=1, LoRA r=8 | OOM | 共有メモリに溢れる |
| seq_len=1024, batch=2, LoRA r=8 | OOM | 同上 |

> **重要**: `device_map="auto"` は使用しないこと。VRAMオーバー時にシステムRAMへの  
> 自動オフロードが起き、PCIeボトルネックで学習速度が100倍以上低下する。  
> `device_map={"": 0}` で全レイヤーをGPU強制配置すること。

## 実験結果

### 学習条件

| 項目 | 値 |
|------|-----|
| データ | Wikipedia日本語 12,000文書 |
| エポック数 | 1 |
| 学習時間 | 7時間55分 |
| 総ステップ数 | 941 |

### 評価結果

| 指標 | 値 |
|------|-----|
| Train Loss | 4.4887 |
| Eval Loss | 4.4346 |
| Eval Perplexity | **84.32** |

### 生成品質の所感

日本語プロンプトに対して最初の1〜2文は日本語で生成されるが、その後英語に切り替わる傾向がある。  
ベースモデルが英語中心のデータで事前学習されており、1.2万文書・1エポックの学習量では英語バイアスを上書きするには不十分。  
実用的な日本語生成には、より大規模な学習データ（数十億トークン以上）が必要と考えられる。

## 実装上のハマりどころと解決策

実際に開発・実行する中で遭遇した問題と解決策をまとめる。

### 1. BitNetにQLoRAは使えない

**問題**: `BitsAndBytesConfig` を渡すと `ValueError: The model is quantized with BitNetQuantConfig but you are passing a BitsAndBytesConfig config.` が発生。

**原因**: BitNet bf16モデルは内部に独自の `BitNetQuantConfig`（3値量子化の設定）を持つため、bitsandbytesの外部4-bit量子化と競合する。

**解決策**: `BitsAndBytesConfig` を使わず、bf16のままロードしてLoRAアダプタのみ学習する。

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": 0},
    dtype=torch.bfloat16,
)
```

### 2. Triton非対応でWeightQuantがクラッシュ

**問題**: 学習開始直後に `torch._inductor.exc.TritonMissing: Cannot find a working triton installation.` が発生。

**原因**: BitNetの `WeightQuant` カスタム関数を `torch.compile`（dynamo）がコンパイルしようとするが、WindowsではTritonが動かない。

**解決策**: スクリプト冒頭でdynamoを無効化する。

```python
import torch._dynamo
torch._dynamo.config.disable = True
```

### 3. device_map="auto" による激遅問題

**問題**: 学習速度が 720秒/ステップ（推定232時間）という異常な遅さになった。

**原因**: bf16の2.4Bモデル（~4.8GB）＋アクティベーション＋オプティマイザがVRAM 8GBを超え、`device_map="auto"` がモデルの一部をシステムRAMに配置。GPU⇔RAM間のPCIe転送（~32GB/s）がボトルネックになった。

**解決策**: `device_map={"": 0}` に変更してGPU強制配置。seq_len=2048→1024、batch=2→1に削減してVRAMに収める。

```
device_map="auto"  → 720秒/ステップ（共有GPUメモリ15.9GB使用）
device_map={"": 0} →  30秒/ステップ（専用VRAM 7.6GB以内）
```

### 4. Unsloth非対応

**問題**: BitNetはUnslothのサポート対象外。Llama/Mistral/Qwen/Gemmaなどのメジャーアーキテクチャのみ対応。

**解決策**: 素のHuggingFace transformers + PEFTで代替。Unslothの最適化（Tritonカーネル）は得られないが、正しく学習は動作する。

---

## ディレクトリ構成

```
1.58_bit_transformer/
├── configs/
│   └── train_config.yaml    # 学習設定
├── data/
│   └── processed/           # 前処理済みデータ (自動生成)
├── outputs/
│   ├── checkpoint-*/        # チェックポイント
│   └── final_model/         # 最終モデル (LoRAアダプタ)
├── scripts/
│   ├── prepare_data.py      # データ準備
│   ├── train.py             # 学習スクリプト
│   └── evaluate.py          # 評価スクリプト
├── requirements.txt         # 依存関係
└── README.md
```

## 今後の改善方向

- **学習データの大規模化**: 実用的な日本語生成には数十億トークン以上が必要
- **クラウドGPUの活用**: Google Colab A100等でより大規模な学習を実施
- **ベースモデルの変更**: 日本語特化モデル（llm-jp等）をベースにした方がLoRAでの改善効果が高い
- **ドメイン特化**: 汎用日本語ではなく特定ドメイン（技術文書等）に特化することで少ないデータでも効果が出る可能性

## ライセンス

- ベースモデル (`bitnet-b1.58-2B-4T-bf16`): MIT License
- llm-jp-corpus-v4: 各サブセットごとに異なるライセンス（使用時に確認してください）
