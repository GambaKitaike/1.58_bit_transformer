"""
BitNet b1.58 日本語継続事前学習 - 評価スクリプト

学習済みモデルの perplexity 評価、テキスト生成サンプル、
ベースモデルとの比較を行う。

Usage (from project root):
    python scripts/evaluate.py --model-path outputs/final_model
    python scripts/evaluate.py --model-path outputs/final_model --interactive
"""

import argparse
import math
import sys
import time

import torch
import yaml
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
    load_in_4bit: bool = True,
    device: str = "cuda",
):
    """ファインチューニング済みモデル（ベース + LoRAアダプタ）をロード"""
    print(f"[INFO] Loading base model: {base_model_name}")
    print(f"[INFO] Loading adapter from: {adapter_path}")

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def load_base_model(
    model_name: str,
    load_in_4bit: bool = True,
):
    """ベースモデルのみをロード（比較用）"""
    print(f"[INFO] Loading base model (no adapter): {model_name}")

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────
# Perplexity 評価
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_perplexity(model, dataset, batch_size: int = 4, max_samples: int = None):
    """データセット上でperplexityを計算"""
    total_loss = 0.0
    total_tokens = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating PPL"):
        batch = dataset[i:i + batch_size]
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long).cuda()
        attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long).cuda()
        labels = input_ids.clone()
        labels[labels == model.config.pad_token_id if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id else -1] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        num_tokens = attention_mask.sum().item()
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return {"loss": avg_loss, "perplexity": perplexity, "num_tokens": total_tokens}


# ─────────────────────────────────────────────
# テキスト生成
# ─────────────────────────────────────────────

@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
):
    """テキスト生成"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    elapsed = time.time() - start_time

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    tokens_per_sec = len(generated_tokens) / elapsed

    return {
        "text": generated_text,
        "num_tokens": len(generated_tokens),
        "elapsed_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
    }


# ─────────────────────────────────────────────
# サンプルプロンプト
# ─────────────────────────────────────────────

SAMPLE_PROMPTS = [
    "日本の歴史について簡潔に説明すると、",
    "人工知能の未来について考えると、",
    "東京は日本の首都であり、",
    "量子コンピュータとは、",
    "日本語の特徴として、",
    "地球温暖化の問題について、",
]


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BitNet b1.58 日本語モデル評価")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--model-path", type=str, default="outputs/final_model",
                        help="学習済みLoRAアダプタのパス")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="評価用データセットのパス")
    parser.add_argument("--compare-base", action="store_true",
                        help="ベースモデルとの比較を行う")
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--interactive", action="store_true",
                        help="対話モードで生成テスト")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    data_dir = args.data_dir or config["data"]["output_dir"]

    print("=" * 60)
    print("BitNet b1.58 日本語モデル - 評価")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available.")
        sys.exit(1)

    # ファインチューニング済みモデルのロード
    ft_model, tokenizer = load_finetuned_model(
        base_model_name=model_cfg["name"],
        adapter_path=args.model_path,
        load_in_4bit=model_cfg.get("load_in_4bit", True),
    )

    # ───── Perplexity 評価 ─────
    try:
        dataset = load_from_disk(data_dir)
        val_dataset = dataset["validation"]

        print()
        print("-" * 40)
        print("Perplexity 評価 (Fine-tuned)")
        print("-" * 40)
        ft_metrics = evaluate_perplexity(
            ft_model, val_dataset, args.batch_size, args.max_eval_samples
        )
        print(f"  Loss       : {ft_metrics['loss']:.4f}")
        print(f"  Perplexity : {ft_metrics['perplexity']:.2f}")
        print(f"  Tokens     : {ft_metrics['num_tokens']:,}")

        if args.compare_base:
            print()
            print("-" * 40)
            print("Perplexity 評価 (Base)")
            print("-" * 40)
            del ft_model
            torch.cuda.empty_cache()

            base_model, _ = load_base_model(model_cfg["name"])
            base_metrics = evaluate_perplexity(
                base_model, val_dataset, args.batch_size, args.max_eval_samples
            )
            print(f"  Loss       : {base_metrics['loss']:.4f}")
            print(f"  Perplexity : {base_metrics['perplexity']:.2f}")

            print()
            print("-" * 40)
            print("比較")
            print("-" * 40)
            ppl_diff = ft_metrics["perplexity"] - base_metrics["perplexity"]
            ppl_pct = (ppl_diff / base_metrics["perplexity"]) * 100
            print(f"  PPL差分    : {ppl_diff:+.2f} ({ppl_pct:+.1f}%)")

            del base_model
            torch.cuda.empty_cache()

            ft_model, tokenizer = load_finetuned_model(
                base_model_name=model_cfg["name"],
                adapter_path=args.model_path,
            )
    except FileNotFoundError:
        print(f"[WARN] Dataset not found at {data_dir}, skipping PPL evaluation")

    # ───── テキスト生成サンプル ─────
    print()
    print("=" * 60)
    print("テキスト生成サンプル")
    print("=" * 60)

    for prompt in SAMPLE_PROMPTS:
        result = generate_text(
            ft_model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"\n--- Prompt: {prompt}")
        print(f"    Generated ({result['num_tokens']} tokens, "
              f"{result['tokens_per_sec']:.1f} tok/s):")
        print(f"    {result['text'][:300]}")

    # ───── 対話モード ─────
    if args.interactive:
        print()
        print("=" * 60)
        print("対話モード (quit/exit で終了)")
        print("=" * 60)

        while True:
            try:
                prompt = input("\nPrompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue

            result = generate_text(
                ft_model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            print(f"\n{result['text']}")
            print(f"\n({result['num_tokens']} tokens, {result['tokens_per_sec']:.1f} tok/s)")

    print()
    print("評価完了")


if __name__ == "__main__":
    main()
