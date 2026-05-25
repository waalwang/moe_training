"""
model_loader.py

Shared model/tokenizer loading for Gemma 4 MoE.

Gemma 4 registers as Gemma4ForConditionalGeneration (multimodal arch),
which AutoModelForCausalLM cannot load. We try CausalLM first (works for
most models), then fall back to AutoModelForConditionalGeneration.

When checkpoint is a LoRA adapter directory (has adapter_config.json),
loads the base model, merges the adapter in memory, and returns the
merged model -- no disk writes.
"""

from __future__ import annotations

import json
import logging
import os

import torch
from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def build_quantization_config(cfg: dict) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg["training"]["bf16"] else torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(cfg: dict, section: str = "qlora") -> LoraConfig:
    qlora = cfg[section]
    return LoraConfig(
        r=qlora["r"],
        lora_alpha=qlora["lora_alpha"],
        lora_dropout=qlora["lora_dropout"],
        target_modules=qlora["target_modules"],
        exclude_modules=qlora.get("exclude_modules"),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def _load_pretrained(model_name: str, model_kwargs: dict):
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except ValueError:
        pass

    from transformers import AutoModelForConditionalGeneration
    logger.info("AutoModelForCausalLM failed, using AutoModelForConditionalGeneration")
    return AutoModelForConditionalGeneration.from_pretrained(model_name, **model_kwargs)


def _is_adapter_checkpoint(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def _load_and_merge_adapter(checkpoint: str, model_kwargs: dict):
    with open(os.path.join(checkpoint, "adapter_config.json")) as f:
        adapter_cfg = json.load(f)
    base_model_path = adapter_cfg["base_model_name_or_path"]
    logger.info("Loading base model: %s", base_model_path)
    base_model = _load_pretrained(base_model_path, model_kwargs)
    logger.info("Merging adapter from: %s", checkpoint)
    model = PeftModel.from_pretrained(base_model, checkpoint)
    model = model.merge_and_unload()
    return model


def _patch_moe_experts(model, cfg: dict) -> None:
    """Quantize Gemma4TextExperts to Linear4bit if present. No-op on non-MoE models.

    If model.moe_cache_dir is set in config, loads pre-built cache from
    convert_gemma4_moe.py instead of re-quantizing from scratch.
    """
    cache_dir = cfg.get("model", {}).get("moe_cache_dir")
    t = cfg["training"]
    compute_dtype = torch.bfloat16 if t["bf16"] else torch.float16

    if cache_dir:
        from moe_quant_patch import load_expert_cache
        logger.info("Loading MoE expert cache from %s", cache_dir)
        n = load_expert_cache(model, cache_dir)
    else:
        from moe_quant_patch import patch_moe_experts_to_4bit
        n = patch_moe_experts_to_4bit(model, compute_dtype=compute_dtype)

    if n > 0:
        _log_quantization_stats(model)


def _offload_unused_towers(model) -> None:
    """Move vision/audio towers to CPU for text-only training."""
    # Gemma4ForConditionalGeneration wraps Gemma4Model under self.model;
    # vision/audio attributes live there, not on the top-level wrapper.
    target = getattr(model, "model", model)
    offloaded = []
    for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        mod = getattr(target, attr, None)
        if mod is not None:
            mod.to("cpu")
            offloaded.append(attr)
    if offloaded:
        logger.info("Offloaded to CPU (text-only SFT): %s", ", ".join(offloaded))
        torch.cuda.empty_cache()


def _log_quantization_stats(model) -> None:
    try:
        import bitsandbytes as bnb
        n_quant = sum(1 for _, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit))
        n_linear = sum(1 for _, m in model.named_modules() if isinstance(m, torch.nn.Linear))
        logger.info("Quantization check: %d Linear4bit, %d nn.Linear", n_quant, n_linear)
        if n_quant == 0:
            logger.warning("No Linear4bit layers found -- model may not be 4-bit quantized!")
    except ImportError:
        pass


def load_model_and_tokenizer(cfg: dict, checkpoint: str | None = None):
    t = cfg["training"]
    full_ft = cfg["_full_finetune"]

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if full_ft:
        if t["bf16"]:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif t["fp16"]:
            model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["quantization_config"] = build_quantization_config(cfg)

    attn_impl = t.get("attn_implementation")
    if attn_impl and attn_impl != "eager":
        model_kwargs["attn_implementation"] = attn_impl

    if checkpoint and _is_adapter_checkpoint(checkpoint):
        logger.info("Adapter checkpoint detected, will merge in memory")
        merge_kwargs = {k: v for k, v in model_kwargs.items()
                        if k != "quantization_config"}
        merge_kwargs["torch_dtype"] = (
            torch.bfloat16 if t["bf16"] else torch.float16
        )
        model = _load_and_merge_adapter(checkpoint, merge_kwargs)
        if not full_ft:
            model = prepare_model_for_kbit_training(model)
        tokenizer_src = checkpoint
    else:
        model_name = checkpoint or cfg["model"]["name"]
        logger.info("Loading model: %s", model_name)
        model = _load_pretrained(model_name, model_kwargs)
        if not full_ft:
            _log_quantization_stats(model)
            _patch_moe_experts(model, cfg)
            _offload_unused_towers(model)
            model = prepare_model_for_kbit_training(model)
        tokenizer_src = model_name

    logger.info("Profile: %s | full_finetune=%s", cfg["_profile"], full_ft)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer
