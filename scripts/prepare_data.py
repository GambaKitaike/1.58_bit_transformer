"""
BitNet b1.58 日本語継続事前学習 - データ準備スクリプト

llm-jp-corpus-v4 / Wikipedia / mc4 から日本語テキストを抽出し、
トークナイズ・チャンキングして学習用データセットを作成する。

Usage (from project root):
    python scripts/prepare_data.py --config configs/train_config.yaml
    python scripts/prepare_data.py --source wikipedia --max-documents 30000
"""

import argparse
import json
import os
import re
from pathlib import Path

import yaml
import torch
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# データソース別ロード
# ─────────────────────────────────────────────

def load_wikipedia_ja(max_documents: int) -> Dataset:
    """HuggingFace の日本語Wikipediaデータセットをロード"""
    print("[INFO] Loading Japanese Wikipedia from HuggingFace...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
    if max_documents and len(ds) > max_documents:
        ds = ds.shuffle(seed=42).select(range(max_documents))
    return ds


def load_mc4_ja(max_documents: int) -> Dataset:
    """HuggingFace の mC4 日本語サブセットをロード"""
    print("[INFO] Loading Japanese mC4 from HuggingFace...")
    ds = load_dataset("mc4", "ja", split="train", streaming=True)
    texts = []
    for i, example in enumerate(tqdm(ds, desc="Loading mC4", total=max_documents)):
        if i >= max_documents:
            break
        texts.append({"text": example["text"]})
    return Dataset.from_list(texts)


def load_llm_jp_corpus_v4(local_path: str, max_documents: int, text_field: str = "text") -> Dataset:
    """
    llm-jp-corpus-v4 のローカルファイルからロード。
    JSONL形式を想定（1行1JSON、textフィールドにテキスト）。

    GitLabからのクローン:
        git clone https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4.git
    """
    print(f"[INFO] Loading llm-jp-corpus-v4 from {local_path}...")
    path = Path(local_path)

    if not path.exists():
        raise FileNotFoundError(
            f"パスが見つかりません: {local_path}\n"
            "llm-jp-corpus-v4は以下からクローンしてください:\n"
            "  git clone https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4.git"
        )

    jsonl_files = sorted(path.rglob("*.jsonl")) + sorted(path.rglob("*.jsonl.gz"))
    if not jsonl_files:
        jsonl_files = sorted(path.rglob("*.json")) + sorted(path.rglob("*.json.gz"))

    if not jsonl_files:
        raise FileNotFoundError(f"JSONL/JSONファイルが見つかりません: {local_path}")

    print(f"[INFO] Found {len(jsonl_files)} data files")

    texts = []
    for fpath in tqdm(jsonl_files, desc="Loading files"):
        try:
            ds = load_dataset("json", data_files=str(fpath), split="train")
            for row in ds:
                if text_field in row and row[text_field]:
                    texts.append({"text": row[text_field]})
                    if max_documents and len(texts) >= max_documents:
                        break
        except Exception as e:
            print(f"[WARN] Skipping {fpath}: {e}")
            continue
        if max_documents and len(texts) >= max_documents:
            break

    print(f"[INFO] Loaded {len(texts)} documents")
    return Dataset.from_list(texts)


def load_local_jsonl(local_path: str, max_documents: int, text_field: str = "text") -> Dataset:
    """カスタムJSONLファイルからロード"""
    print(f"[INFO] Loading local JSONL from {local_path}...")
    ds = load_dataset("json", data_files=local_path, split="train")
    if text_field != "text" and text_field in ds.column_names:
        ds = ds.rename_column(text_field, "text")
    if max_documents and len(ds) > max_documents:
        ds = ds.shuffle(seed=42).select(range(max_documents))
    return ds


# ─────────────────────────────────────────────
# テキスト前処理
# ─────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    """基本的なテキストクリーニング"""
    text = _URL_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def is_quality_japanese(text: str, min_length: int = 100) -> bool:
    """日本語テキストの品質フィルタ"""
    if len(text) < min_length:
        return False
    ja_chars = sum(1 for c in text if '\u3040' <= c <= '\u9fff' or '\uf900' <= c <= '\ufaff')
    ja_ratio = ja_chars / len(text) if text else 0
    if ja_ratio < 0.1:
        return False
    if len(set(text)) / len(text) < 0.01:
        return False
    return True


def filter_and_clean(dataset: Dataset, min_length: int, max_length: int) -> Dataset:
    """テキストのクリーニングとフィルタリング"""
    print("[INFO] Cleaning and filtering texts...")

    def process(example):
        text = clean_text(example["text"])
        return {"text": text, "keep": is_quality_japanese(text, min_length)}

    dataset = dataset.map(process, num_proc=4, desc="Cleaning")
    before_len = len(dataset)
    dataset = dataset.filter(lambda x: x["keep"])
    dataset = dataset.remove_columns(["keep"])

    dataset = dataset.filter(lambda x: len(x["text"]) <= max_length)
    print(f"[INFO] Filtered: {before_len} -> {len(dataset)} documents")
    return dataset


# ─────────────────────────────────────────────
# トークナイズ & チャンキング
# ─────────────────────────────────────────────

def tokenize_and_chunk(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> Dataset:
    """
    テキストをトークナイズし、固定長チャンクに分割。
    連続事前学習向けにパッキング（複数文書を連結してチャンクに）。
    """
    print(f"[INFO] Tokenizing and chunking (seq_len={max_seq_length})...")
    eos_token_id = tokenizer.eos_token_id

    all_input_ids = []
    buffer = []

    for example in tqdm(dataset, desc="Tokenizing"):
        tokens = tokenizer.encode(example["text"], add_special_tokens=False)
        tokens.append(eos_token_id)
        buffer.extend(tokens)

        while len(buffer) >= max_seq_length:
            chunk = buffer[:max_seq_length]
            all_input_ids.append(chunk)
            buffer = buffer[max_seq_length:]

    if len(buffer) > max_seq_length // 2:
        chunk = buffer[:max_seq_length]
        chunk += [tokenizer.pad_token_id or eos_token_id] * (max_seq_length - len(chunk))
        all_input_ids.append(chunk)

    print(f"[INFO] Created {len(all_input_ids)} chunks of length {max_seq_length}")

    result_dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": [[1] * len(ids) for ids in all_input_ids],
        "labels": [ids.copy() for ids in all_input_ids],
    })

    return result_dataset


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BitNet日本語継続事前学習 - データ準備")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--source", type=str, default=None,
                        help="データソース: wikipedia, mc4, llm-jp-corpus-v4, local_jsonl")
    parser.add_argument("--local-path", type=str, default=None)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]

    source = args.source or data_cfg["source"]
    local_path = args.local_path or data_cfg.get("local_path")
    max_documents = args.max_documents or data_cfg["max_documents"]
    max_seq_length = args.max_seq_length or model_cfg["max_seq_length"]
    output_dir = args.output_dir or data_cfg["output_dir"]
    text_field = data_cfg.get("text_field", "text")
    min_text_length = data_cfg.get("min_text_length", 100)
    max_text_length = data_cfg.get("max_text_length", 50000)
    val_split = data_cfg.get("validation_split", 0.02)

    print("=" * 60)
    print("BitNet b1.58 日本語継続事前学習 - データ準備")
    print("=" * 60)
    print(f"  データソース    : {source}")
    print(f"  最大文書数      : {max_documents}")
    print(f"  シーケンス長    : {max_seq_length}")
    print(f"  出力先          : {output_dir}")
    print("=" * 60)

    # データロード
    if source == "wikipedia":
        raw_dataset = load_wikipedia_ja(max_documents)
    elif source == "mc4":
        raw_dataset = load_mc4_ja(max_documents)
    elif source == "llm-jp-corpus-v4":
        if not local_path:
            raise ValueError(
                "llm-jp-corpus-v4を使用するには --local-path を指定してください。\n"
                "クローン: git clone https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4.git"
            )
        raw_dataset = load_llm_jp_corpus_v4(local_path, max_documents, text_field)
    elif source == "local_jsonl":
        if not local_path:
            raise ValueError("local_jsonlを使用するには --local-path を指定してください")
        raw_dataset = load_local_jsonl(local_path, max_documents, text_field)
    else:
        raise ValueError(f"不明なデータソース: {source}")

    # フィルタリング & クリーニング
    cleaned_dataset = filter_and_clean(raw_dataset, min_text_length, max_text_length)

    # トークナイザーロード
    print(f"[INFO] Loading tokenizer: {model_cfg['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # トークナイズ & チャンキング
    chunked_dataset = tokenize_and_chunk(cleaned_dataset, tokenizer, max_seq_length)

    # Train/Validation 分割
    split = chunked_dataset.train_test_split(test_size=val_split, seed=42)
    dataset_dict = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)

    print()
    print("=" * 60)
    print("データ準備完了")
    print("=" * 60)
    print(f"  学習データ   : {len(dataset_dict['train'])} chunks")
    print(f"  検証データ   : {len(dataset_dict['validation'])} chunks")
    print(f"  トークン総数 : {len(dataset_dict['train']) * max_seq_length:,}")
    print(f"  保存先       : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
