"""
BitNet b1.58 日本語継続事前学習 - 学習スクリプト

Unsloth + QLoRA を最初に試み、非対応の場合は HuggingFace transformers + PEFT に
自動フォールバックする。RTX 5060 (8GB VRAM) 向けに最適化。

Usage (from project root):
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --backend hf_peft --batch-size 2
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
import torch._dynamo
torch._dynamo.config.disable = True
import yaml
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# バックエンド: Unsloth
# ─────────────────────────────────────────────

def load_model_unsloth(model_cfg: dict, lora_cfg: dict):
    """Unsloth の FastLanguageModel でモデルをロード"""
    from unsloth import FastLanguageModel

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(model_cfg.get("dtype"), None)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=dtype,
        load_in_4bit=model_cfg.get("load_in_4bit", True),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0),
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=model_cfg["max_seq_length"],
    )

    return model, tokenizer


# ─────────────────────────────────────────────
# バックエンド: HuggingFace + PEFT (フォールバック)
# ─────────────────────────────────────────────

def load_model_hf_peft(model_cfg: dict, lora_cfg: dict):
    """
    HuggingFace transformers + PEFT でモデルをロード。
    BitNetモデルは独自のBitNetQuantConfigを持つため、
    BitsAndBytesConfigによる外部量子化は不可。bf16 + LoRAで学習する。
    """
    from peft import LoraConfig, get_peft_model

    print("[INFO] Loading model with HuggingFace + PEFT backend...")
    print("[INFO] BitNet uses native quantization - loading in bf16 with LoRA (not QLoRA)")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    compute_dtype = dtype_map.get(model_cfg.get("dtype"), torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        device_map={"": 0},
        dtype=compute_dtype,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # ベース重みを凍結
    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )

    model = get_peft_model(model, lora_config)
    return model, tokenizer


# ─────────────────────────────────────────────
# モデルロード (自動選択)
# ─────────────────────────────────────────────

def load_model(model_cfg: dict, lora_cfg: dict, backend: str = "auto"):
    """
    backendの設定に応じてモデルをロード。
    "auto": Unslothを試み、失敗したらHF+PEFTにフォールバック。
    """
    if backend == "unsloth":
        return load_model_unsloth(model_cfg, lora_cfg)
    elif backend == "hf_peft":
        return load_model_hf_peft(model_cfg, lora_cfg)
    elif backend == "auto":
        try:
            print("[INFO] Trying Unsloth backend...")
            model, tokenizer = load_model_unsloth(model_cfg, lora_cfg)
            print("[INFO] Unsloth backend loaded successfully!")
            return model, tokenizer
        except Exception as e:
            print(f"[WARN] Unsloth failed ({e}), falling back to HF+PEFT...")
            return load_model_hf_peft(model_cfg, lora_cfg)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ─────────────────────────────────────────────
# データコレーター
# ─────────────────────────────────────────────

class PretrainDataCollator:
    """事前学習用データコレーター: input_ids をそのまま返し labels を設定"""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ─────────────────────────────────────────────
# 学習パラメータ情報の表示
# ─────────────────────────────────────────────

def print_trainable_parameters(model):
    total_params = 0
    trainable_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    pct = 100 * trainable_params / total_params if total_params > 0 else 0
    print(f"[INFO] Parameters: {total_params:,} total, {trainable_params:,} trainable ({pct:.2f}%)")


def estimate_vram_usage(model, train_cfg: dict, seq_len: int):
    """大まかなVRAM使用量を推定"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    base_params = total_params - trainable_params

    # BitNet bf16: base weights in bf16 (2 bytes) + LoRA adapters in bf16 (2 bytes)
    model_mem = base_params * 2 + trainable_params * 2
    # AdamW 8bit: 2 states per trainable param, ~1 byte each
    optimizer_mem = trainable_params * 2 * 1

    total_mb = (model_mem + optimizer_mem) / (1024 ** 2)
    print(f"[INFO] Estimated VRAM (model+optimizer): ~{total_mb:.0f} MB")
    print(f"[INFO] Remaining for activations/cache: ~{8192 - total_mb:.0f} MB")
    if total_mb > 7000:
        print("[WARN] VRAM tight! Consider --batch-size 1 --grad-accum 16")


# ─────────────────────────────────────────────
# WandB初期化
# ─────────────────────────────────────────────

def init_wandb(wandb_cfg: dict):
    try:
        import wandb
        wandb.init(
            project=wandb_cfg.get("project", "bitnet-b158-japanese"),
            entity=wandb_cfg.get("entity"),
            tags=wandb_cfg.get("tags", []),
            config=wandb_cfg,
        )
        print("[INFO] WandB initialized")
        return True
    except Exception as e:
        print(f"[WARN] WandB init failed: {e}")
        print("[INFO] Continuing with TensorBoard only")
        return False


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BitNet b1.58 日本語継続事前学習")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--backend", type=str, default=None, choices=["unsloth", "hf_peft", "auto"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    wandb_cfg = config.get("wandb", {})
    backend = args.backend or config.get("backend", "auto")

    # CLI overrides
    if args.batch_size:
        train_cfg["per_device_train_batch_size"] = args.batch_size
    if args.grad_accum:
        train_cfg["gradient_accumulation_steps"] = args.grad_accum
    if args.lr:
        train_cfg["learning_rate"] = args.lr
    if args.epochs:
        train_cfg["num_train_epochs"] = args.epochs
    data_dir = args.data_dir or data_cfg["output_dir"]
    output_dir = args.output_dir or train_cfg["output_dir"]

    report_to = train_cfg.get("report_to", ["tensorboard"])
    if args.no_wandb and "wandb" in report_to:
        report_to.remove("wandb")

    effective_batch = (
        train_cfg["per_device_train_batch_size"]
        * train_cfg["gradient_accumulation_steps"]
    )

    print("=" * 60)
    print("BitNet b1.58 日本語継続事前学習")
    print("=" * 60)
    print(f"  バックエンド       : {backend}")
    print(f"  モデル             : {model_cfg['name']}")
    print(f"  シーケンス長       : {model_cfg['max_seq_length']}")
    print(f"  LoRA rank          : {lora_cfg['r']}")
    print(f"  バッチサイズ       : {train_cfg['per_device_train_batch_size']}")
    print(f"  勾配累積           : {train_cfg['gradient_accumulation_steps']}")
    print(f"  実効バッチサイズ   : {effective_batch}")
    print(f"  学習率             : {train_cfg['learning_rate']}")
    print(f"  エポック数         : {train_cfg['num_train_epochs']}")
    print(f"  量子化             : BitNet native (bf16 + LoRA)")
    print(f"  Gradient Checkpoint: {train_cfg.get('gradient_checkpointing', True)}")
    print("=" * 60)

    # CUDA確認
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. GPU is required.")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[INFO] GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # WandB初期化
    if "wandb" in report_to and not args.no_wandb:
        init_wandb(wandb_cfg)

    # データロード
    print(f"[INFO] Loading prepared dataset from {data_dir}...")
    if not Path(data_dir).exists():
        print(f"[ERROR] Dataset not found at {data_dir}")
        print("       Run prepare_data.py first:")
        print("       python prepare_data.py --config configs/train_config.yaml")
        sys.exit(1)

    dataset = load_from_disk(data_dir)
    print(f"[INFO] Train: {len(dataset['train'])} samples, Val: {len(dataset['validation'])} samples")

    # モデルロード
    model, tokenizer = load_model(model_cfg, lora_cfg, backend)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print_trainable_parameters(model)
    estimate_vram_usage(model, train_cfg, model_cfg["max_seq_length"])

    # データコレーター
    data_collator = PretrainDataCollator(
        pad_token_id=tokenizer.pad_token_id,
    )

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_steps=max(1, int(len(dataset["train"]) / effective_batch * train_cfg.get("warmup_ratio", 0.05))),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_dir=None,
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy=train_cfg.get("evaluation_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 200),
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 2),
        dataloader_pin_memory=train_cfg.get("dataloader_pin_memory", True),
        remove_unused_columns=False,
        report_to=report_to,
        run_name=train_cfg.get("run_name", "bitnet-b158-ja-cpt"),
        seed=train_cfg.get("seed", 42),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        resume_from_checkpoint=args.resume_from,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )

    # 学習開始
    print()
    print("=" * 60)
    print("学習開始")
    print("=" * 60)

    train_result = trainer.train(resume_from_checkpoint=args.resume_from)

    # メトリクス保存
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # 最終評価
    print()
    print("[INFO] Running final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # モデル保存
    final_dir = os.path.join(output_dir, "final_model")
    print(f"[INFO] Saving model to {final_dir}...")

    # LoRAアダプタの保存
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # TrainerState保存
    trainer.save_state()

    print()
    print("=" * 60)
    print("学習完了")
    print("=" * 60)
    print(f"  Train Loss    : {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Eval Loss     : {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"  Eval Perplexity: {torch.exp(torch.tensor(eval_metrics.get('eval_loss', 0))):.2f}")
    print(f"  保存先        : {final_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
