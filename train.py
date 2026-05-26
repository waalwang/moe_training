"""
train.py

Weighted SFT training for MoE models (Gemma 4 26B).

Usage:
    # Cloud QLoRA (attention-only, recommended):
    python train.py --profile cloud

    # Cloud QLoRA with shared MLP adapters:
    python train.py --profile cloud --qlora-section qlora_with_mlp

    # Cloud full FT (multi-GPU required):
    python train.py --profile cloud_full

    # Dry run:
    python train.py --profile cloud --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
import yaml
from peft import get_peft_model
from transformers import TrainerCallback
from trl import SFTConfig

from data_loader import load_from_config
from model_loader import (
    build_lora_config,
    load_model_and_tokenizer,
    load_moe_model_and_tokenizer,
    log_trainable_parameters,
)
from weighted_sft_trainer import WeightedSFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SyncStateStepsCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        state.compute_steps(args, state.max_steps)


class DeleteOptimizerCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps:
            return
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        opt = os.path.join(ckpt, "optimizer.pt")
        if os.path.exists(opt):
            os.remove(opt)
            logger.info("Deleted %s", opt)


def load_config(config_path: str, profile: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    hw = cfg["profiles"][profile]
    cfg["training"].update(hw)
    cfg["_profile"] = profile
    cfg["_full_finetune"] = cfg["training"].get("full_finetune", False)
    return cfg


def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    mask = labels != -100
    correct = ((preds[:, :-1] == labels[:, 1:]) & mask[:, 1:]).sum()
    total = mask[:, 1:].sum()
    return {"accuracy": (correct / total).item() if total > 0 else 0.0}


def build_training_args(cfg: dict) -> SFTConfig:
    t = cfg["training"]
    return SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t.get("warmup_ratio", 0.05),
        weight_decay=t["weight_decay"],
        lr_scheduler_type=t["lr_scheduler_type"],
        lr_scheduler_kwargs=t.get("lr_scheduler_kwargs"),
        bf16=t["bf16"],
        fp16=t["fp16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        eval_strategy=t.get("eval_strategy", "steps"),
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to=t.get("report_to", "none"),
        run_name=t.get("run_name"),
        max_length=t["max_length"],
        dataset_num_proc=t.get("dataset_num_proc"),
        seed=cfg["data"].get("seed", 42),
        remove_unused_columns=False,
    )


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser(description="MoE Weighted SFT training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--profile", default="cloud",
                        choices=["local", "cloud", "cloud_full"])
    parser.add_argument("--qlora-section", default="qlora_with_mlp",
                        help="Config section for LoRA (qlora, qlora_with_mlp)")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.profile)
    full_ft = cfg["_full_finetune"]

    logger.info("Loading SFT chain dataset...")
    dataset = load_from_config(cfg)

    if args.dry_run:
        for split_name, split in dataset.items():
            logger.info("--- %s: %d examples ---", split_name, len(split))
            if len(split) > 0:
                sample = split[0]
                logger.info("  turns: %d", len(sample["messages"]))
                tw = sample["turn_weights"]
                asst_w = [w for w in tw if w > 0]
                if asst_w:
                    logger.info("  turn_weights: %d turns, %d assistant, "
                                "asst mean=%.3f", len(tw), len(asst_w),
                                sum(asst_w) / len(asst_w))
                else:
                    logger.info("  turn_weights: %d turns, no assistant", len(tw))
                content = sample["messages"][0]["content"][:120]
                logger.info("  first turn: %s...", content)

    if cfg["model"].get("is_moe") and not full_ft:
        model, tokenizer = load_moe_model_and_tokenizer(cfg)
    else:
        model, tokenizer = load_model_and_tokenizer(cfg)

    if full_ft:
        total = sum(p.numel() for p in model.parameters())
        logger.info("Full fine-tune: %d params, all trainable", total)
    else:
        lora_config = build_lora_config(cfg, section=args.qlora_section)
        model = get_peft_model(model, lora_config)
        log_trainable_parameters(model)

    if args.dry_run:
        logger.info("Dry run complete.")
        return

    training_args = build_training_args(cfg)

    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[SyncStateStepsCallback(), DeleteOptimizerCallback()],
    )

    logger.info("Starting weighted SFT training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.info("Training complete. Best checkpoint kept in %s", cfg["training"]["output_dir"])


if __name__ == "__main__":
    main()
