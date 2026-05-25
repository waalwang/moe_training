"""
export_merged.py

Merges a LoRA SFT adapter into the Gemma 4 MoE base model and saves
the result as a standard HF bf16 model, ready for GGUF conversion.

Loads in bf16 without quantization so the saved weights are clean.
Expert FFN stays in the original 3D-Parameter format (no _QuantizedExperts),
which is what llama.cpp's convert_hf_to_gguf.py expects.

Requires ~55 GB VRAM+RAM combined (device_map="auto" handles the split).

Usage:
    python export_merged.py \
        --adapter outputs/sft/checkpoint-2000 \
        --output  outputs/merged_bf16
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_base_model(model_name: str):
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    try:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except ValueError:
        from transformers import AutoModelForConditionalGeneration
        logger.info("Falling back to AutoModelForConditionalGeneration")
        return AutoModelForConditionalGeneration.from_pretrained(model_name, **model_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True,
                        help="Path to LoRA adapter checkpoint directory")
    parser.add_argument("--output", required=True,
                        help="Output directory for merged bf16 model")
    args = parser.parse_args()

    adapter_cfg_path = os.path.join(args.adapter, "adapter_config.json")
    if not os.path.exists(adapter_cfg_path):
        raise FileNotFoundError(f"No adapter_config.json in {args.adapter}")

    with open(adapter_cfg_path) as f:
        adapter_cfg = json.load(f)
    model_name = adapter_cfg["base_model_name_or_path"]
    logger.info("Base model: %s", model_name)

    logger.info("Loading base model in bf16 (device_map=auto) ...")
    model = load_base_model(model_name)

    logger.info("Loading and merging LoRA adapter from %s ...", args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    logger.info("Merge complete.")

    logger.info("Saving merged model to %s ...", args.output)
    model.save_pretrained(args.output, safe_serialization=True)

    logger.info("Saving tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)

    size_gb = sum(
        os.path.getsize(os.path.join(args.output, f)) / 1e9
        for f in os.listdir(args.output)
    )
    logger.info("Saved %.1f GB to %s", size_gb, args.output)
    logger.info("Ready for: python llama.cpp/convert_hf_to_gguf.py %s --outtype f16",
                args.output)


if __name__ == "__main__":
    main()
