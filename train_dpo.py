"""
train_dpo.py

DPO training for MoE models (Gemma 4 26B).

Usage:
    # QLoRA from SFT checkpoint (recommended):
    python train_dpo.py --profile cloud --checkpoint outputs/sft/checkpoint-XXXX

    # With shared MLP adapters:
    python train_dpo.py --profile cloud --checkpoint outputs/sft/checkpoint-XXXX \
        --qlora-section qlora_with_mlp

    # Full FT (multi-GPU):
    python train_dpo.py --profile cloud_full --checkpoint outputs/sft/checkpoint-XXXX

    # Dry run:
    python train_dpo.py --profile cloud --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os

import torch
import yaml
from peft import get_peft_model
from trl import DPOConfig

from dpo_data_loader import load_from_config
from model_loader import (
    build_lora_config,
    load_model_and_tokenizer,
    load_moe_model_and_tokenizer,
    log_trainable_parameters,
)
from weighted_dpo_trainer import WeightedDPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str, profile: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    hw = cfg["profiles"][profile]
    cfg["training"].update(hw)
    cfg["_profile"] = profile

    dpo = cfg.get("dpo", {})
    for key in ("output_dir", "learning_rate", "num_train_epochs", "run_name",
                "save_steps", "eval_steps"):
        if key in dpo:
            cfg["training"][key] = dpo[key]

    cfg["_full_finetune"] = cfg["training"].get("full_finetune", False)
    return cfg


def build_dpo_args(cfg: dict) -> DPOConfig:
    t = cfg["training"]
    dpo = cfg.get("dpo", {})
    full_ft = cfg["_full_finetune"]

    precompute = dpo.get("precompute_ref_log_probs", False)
    if full_ft and not precompute:
        logger.info("Full FT detected -- enabling precompute_ref_log_probs")
        precompute = True

    return DPOConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t.get("warmup_ratio", 0.05),
        weight_decay=t["weight_decay"],
        lr_scheduler_type=t["lr_scheduler_type"],
        bf16=t["bf16"],
        fp16=t["fp16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        eval_strategy=t.get("eval_strategy", "steps"),
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=t.get("report_to", "none"),
        run_name=t.get("run_name"),
        max_length=t["max_length"],
        dataset_num_proc=t.get("dataset_num_proc"),
        seed=cfg["data"].get("seed", 42),
        remove_unused_columns=False,
        beta=dpo.get("beta", 0.1),
        loss_type=dpo.get("loss_type", ["sigmoid"]),
        loss_weights=dpo.get("loss_weights"),
        precompute_ref_log_probs=precompute,
    )


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser(description="MoE DPO training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--profile", default="cloud",
                        choices=["local", "cloud", "cloud_full"])
    parser.add_argument("--checkpoint", default=None,
                        help="SFT checkpoint path (policy init + reference)")
    parser.add_argument("--qlora-section", default="dpo_qlora",
                        help="Config section for LoRA (dpo_qlora, qlora_with_mlp)")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.profile)
    full_ft = cfg["_full_finetune"]
    dpo = cfg.get("dpo", {})

    logger.info("Loading DPO dataset...")
    dataset = load_from_config(cfg)

    if args.dry_run:
        for split_name, split in dataset.items():
            logger.info("--- %s: %d examples ---", split_name, len(split))
            if len(split) > 0:
                sample = split[0]
                logger.info("  chosen turns: %d", len(sample["chosen"]))
                logger.info("  rejected turns: %d", len(sample["rejected"]))
                if "chosen_weight" in sample:
                    logger.info("  chosen_weight: %.3f", sample["chosen_weight"])

    if cfg["model"].get("is_moe") and not full_ft:
        model, tokenizer = load_moe_model_and_tokenizer(cfg, args.checkpoint)
    else:
        model, tokenizer = load_model_and_tokenizer(cfg, args.checkpoint)

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

    training_args = build_dpo_args(cfg)

    trainer = WeightedDPOTrainer(
        chosen_weighting=dpo.get("chosen_weighting", False),
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    logger.info(
        "Starting DPO training (beta=%.2f, loss=%s, weights=%s, chosen_weighting=%s)...",
        dpo.get("beta", 0.1),
        dpo.get("loss_type", ["sigmoid"]),
        dpo.get("loss_weights"),
        dpo.get("chosen_weighting", False),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.info("DPO training complete. Best checkpoint in %s", cfg["training"]["output_dir"])


if __name__ == "__main__":
    main()
