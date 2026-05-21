"""
convert_gemma4_moe.py

Offline pre-quantization of Gemma 4 26B MoE expert FFN weights to NF4.

bitsandbytes quantizes attention/MLP nn.Linear layers on-the-fly at load time,
but the MoE expert weights are stored as packed 3D nn.Parameters which
bitsandbytes cannot quantize automatically. This script:

  1. Loads the model with bitsandbytes 4-bit (attention layers quantized).
  2. Applies patch_moe_experts_to_4bit() to quantize the expert Parameters.
  3. Saves a compact expert quant cache (~11 GB) to disk.

At training time the cache is loaded instead of re-quantizing, keeping startup
fast and memory usage low.

Usage:
    python convert_gemma4_moe.py \
        --model-name google/gemma-4-26b-A4B-it \
        --output-dir ./gemma4_moe_quant_cache

    # To verify an existing cache:
    python convert_gemma4_moe.py --verify --output-dir ./gemma4_moe_quant_cache
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache save / load
# ---------------------------------------------------------------------------

def _quant_state_to_cpu(qs):
    """Return QuantState.as_dict() with all tensors moved to CPU."""
    d = qs.as_dict()
    return {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


def save_expert_cache(model, output_dir: str) -> None:
    """
    Walk all _QuantizedExperts modules and serialise their Linear4bit weights
    to {output_dir}/expert_cache.pt.

    Cache structure:
        {
            "<module_path>": {
                "num_experts": int,
                "hidden_dim":  int,
                "intermediate_dim": int,
                "gate_up": [{"weight": Tensor(uint8,cpu), "quant_state": dict}, ...],
                "down":    [{"weight": Tensor(uint8,cpu), "quant_state": dict}, ...],
            },
            ...
        }
    """
    from moe_quant_patch import _QuantizedExperts

    os.makedirs(output_dir, exist_ok=True)
    cache: dict = {}

    for full_name, module in model.named_modules():
        if not isinstance(module, _QuantizedExperts):
            continue

        entry: dict = {
            "num_experts": module.num_experts,
            "hidden_dim": module.hidden_dim,
            "intermediate_dim": module.intermediate_dim,
            "gate_up": [],
            "down": [],
        }

        for layer in module.gate_up_experts:
            entry["gate_up"].append({
                "weight": layer.weight.data.cpu(),
                "quant_state": _quant_state_to_cpu(layer.weight.quant_state),
            })

        for layer in module.down_experts:
            entry["down"].append({
                "weight": layer.weight.data.cpu(),
                "quant_state": _quant_state_to_cpu(layer.weight.quant_state),
            })

        cache[full_name] = entry
        logger.info("Cached %s (%d experts)", full_name, module.num_experts)

    cache_path = os.path.join(output_dir, "expert_cache.pt")
    torch.save(cache, cache_path)
    size_gb = os.path.getsize(cache_path) / 1e9
    logger.info("Saved expert cache: %s (%.1f GB, %d blocks)", cache_path, size_gb, len(cache))


def load_expert_cache(model, cache_dir: str) -> int:
    """
    Load pre-quantized expert weights from cache and replace the model's
    Gemma4TextExperts modules with _QuantizedExperts backed by cached weights.

    Must be called after model load but before prepare_model_for_kbit_training.
    Returns the number of blocks restored from cache.
    """
    from bitsandbytes.functional import QuantState
    from bitsandbytes.nn import Linear4bit
    from bitsandbytes.nn.modules import Params4bit
    from moe_quant_patch import _QuantizedExperts

    cache_path = os.path.join(cache_dir, "expert_cache.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Expert cache not found: {cache_path}")

    logger.info("Loading expert quant cache from %s", cache_path)
    cache: dict = torch.load(cache_path, map_location="cpu", weights_only=False)

    patched = 0
    for full_name, entry in cache.items():
        # Navigate to parent module.
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        original = getattr(parent, parts[-1])

        device = next(model.parameters()).device

        def _restore_expert_list(entries: list[dict]) -> nn.ModuleList:
            from torch import nn
            layers = nn.ModuleList()
            for e in entries:
                qs_dict = e["quant_state"]
                # Reconstruct QuantState from dict.
                qs = QuantState.from_dict(qs_dict, device=device)
                p = Params4bit(
                    e["weight"].to(device),
                    requires_grad=False,
                    quant_state=qs,
                    quant_type=qs.quant_type,
                    bnb_quantized=True,
                )
                in_f = qs.shape[1]
                out_f = qs.shape[0]
                layer = Linear4bit(in_f, out_f, bias=False)
                layer.weight = p
                layer.quant_state = qs
                layers.append(layer.to(device))
            return layers

        quantized = _QuantizedExperts.__new__(_QuantizedExperts)
        torch.nn.Module.__init__(quantized)
        quantized.num_experts = entry["num_experts"]
        quantized.hidden_dim = entry["hidden_dim"]
        quantized.intermediate_dim = entry["intermediate_dim"]
        # Reuse act_fn from the original module.
        quantized.act_fn = original.act_fn

        quantized.gate_up_experts = _restore_expert_list(entry["gate_up"])
        quantized.down_experts = _restore_expert_list(entry["down"])

        # Free original bf16 Parameters before swapping.
        if hasattr(original, "gate_up_proj"):
            del original.gate_up_proj
        if hasattr(original, "down_proj"):
            del original.down_proj
        torch.cuda.empty_cache()

        setattr(parent, parts[-1], quantized)
        patched += 1
        logger.debug("Restored %s from cache", full_name)

    torch.cuda.empty_cache()
    logger.info("Restored %d expert blocks from cache", patched)
    return patched


# ---------------------------------------------------------------------------
# Conversion entry point
# ---------------------------------------------------------------------------

def convert(model_name: str, output_dir: str, bf16: bool) -> None:
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from moe_quant_patch import patch_moe_experts_to_4bit

    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading %s ...", model_name)
    t0 = time.time()

    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
    except ValueError:
        from transformers import AutoModelForConditionalGeneration
        logger.info("Falling back to AutoModelForConditionalGeneration")
        model = AutoModelForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            device_map="auto",
            trust_remote_code=True,
        )

    logger.info("Model loaded in %.1fs", time.time() - t0)

    logger.info("Quantizing MoE expert blocks ...")
    t1 = time.time()
    n = patch_moe_experts_to_4bit(model, compute_dtype=compute_dtype)
    if n == 0:
        logger.error("No Gemma4TextExperts found -- is this a Gemma 4 MoE model?")
        sys.exit(1)
    logger.info("Quantized %d blocks in %.1fs", n, time.time() - t1)

    logger.info("Saving expert cache to %s ...", output_dir)
    save_expert_cache(model, output_dir)

    # Save tokenizer alongside cache for convenience.
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.save_pretrained(output_dir)

    # Save a small manifest.
    manifest = {
        "model_name": model_name,
        "compute_dtype": "bfloat16" if bf16 else "float16",
        "quant_type": "nf4",
        "expert_blocks": n,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Done. Cache ready at: %s", output_dir)


def verify(output_dir: str) -> None:
    cache_path = os.path.join(output_dir, "expert_cache.pt")
    manifest_path = os.path.join(output_dir, "manifest.json")

    if not os.path.exists(cache_path):
        logger.error("Cache file missing: %s", cache_path)
        sys.exit(1)

    size_gb = os.path.getsize(cache_path) / 1e9
    logger.info("Cache file: %s (%.1f GB)", cache_path, size_gb)

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    logger.info("Expert blocks in cache: %d", len(cache))

    first_key = next(iter(cache))
    entry = cache[first_key]
    logger.info("Sample block: %s", first_key)
    logger.info("  num_experts=%d  hidden=%d  intermediate=%d",
                entry["num_experts"], entry["hidden_dim"], entry["intermediate_dim"])
    logger.info("  gate_up[0] weight shape: %s  dtype: %s",
                entry["gate_up"][0]["weight"].shape, entry["gate_up"][0]["weight"].dtype)

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        logger.info("Manifest: %s", manifest)

    logger.info("Verification passed.")


def main():
    parser = argparse.ArgumentParser(description="Convert Gemma 4 MoE experts to 4-bit cache")
    parser.add_argument("--model-name", default="google/gemma-4-26b-A4B-it")
    parser.add_argument("--output-dir", default="./gemma4_moe_quant_cache")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 compute dtype (default: bf16)")
    parser.add_argument("--verify", action="store_true", help="Verify an existing cache, do not convert")
    args = parser.parse_args()

    if args.verify:
        verify(args.output_dir)
    else:
        convert(args.model_name, args.output_dir, bf16=not args.fp16)


if __name__ == "__main__":
    main()
