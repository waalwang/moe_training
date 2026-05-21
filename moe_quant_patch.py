"""
moe_quant_patch.py

Monkey-patches Gemma4TextExperts to replace packed 3D nn.Parameter expert
weights with per-expert Linear4bit modules so bitsandbytes can quantize them.

Background:
  Gemma4TextExperts stores all expert FFN weights as two 3D Parameters:
    gate_up_proj: [num_experts, 2*intermediate_dim, hidden_dim]  (bf16)
    down_proj:    [num_experts, hidden_dim, intermediate_dim]    (bf16)

  bitsandbytes only quantizes nn.Linear submodules. These Parameters are
  invisible to it, so they stay in bf16 and consume ~40-50 GB on a 26B MoE.

Usage:
  model, tokenizer = load_model_and_tokenizer(cfg)
  n = patch_moe_experts_to_4bit(model)
  logger.info("Patched %d MoE expert blocks", n)
  model = prepare_model_for_kbit_training(model)
  model = get_peft_model(model, lora_config)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _QuantizedExperts(nn.Module):
    """Drop-in replacement for Gemma4TextExperts using per-expert Linear4bit."""

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn,
        gate_up_data: torch.Tensor,  # [E, 2*I, H] bf16 on CUDA
        down_data: torch.Tensor,     # [E, H, I]   bf16 on CUDA
        compute_dtype: torch.dtype,
    ):
        super().__init__()
        from bitsandbytes.nn import Linear4bit
        from bitsandbytes.nn.modules import Params4bit

        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn

        self.gate_up_experts: nn.ModuleList = nn.ModuleList()
        self.down_experts: nn.ModuleList = nn.ModuleList()

        device = gate_up_data.device

        for i in range(num_experts):
            gu = Linear4bit(
                hidden_dim, 2 * intermediate_dim,
                bias=False, compute_dtype=compute_dtype, quant_type="nf4",
            )
            gu.weight = Params4bit(
                gate_up_data[i].contiguous(),
                requires_grad=False, quant_type="nf4",
                compress_statistics=True, quant_storage=torch.uint8,
            )
            gu = gu.to(device)  # triggers Params4bit quantization

            d = Linear4bit(
                intermediate_dim, hidden_dim,
                bias=False, compute_dtype=compute_dtype, quant_type="nf4",
            )
            d.weight = Params4bit(
                down_data[i].contiguous(),
                requires_grad=False, quant_type="nf4",
                compress_statistics=True, quant_storage=torch.uint8,
            )
            d = d.to(device)

            self.gate_up_experts.append(gu)
            self.down_experts.append(d)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            gate, up = self.gate_up_experts[expert_idx](current_state).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self.down_experts[expert_idx](current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


def load_expert_cache(model, cache_dir: str) -> int:
    """Load pre-quantized expert cache produced by convert_gemma4_moe.py."""
    from convert_gemma4_moe import load_expert_cache as _load
    return _load(model, cache_dir)


def patch_moe_experts_to_4bit(model, compute_dtype: torch.dtype = torch.bfloat16) -> int:
    """
    Replace all Gemma4TextExperts modules with _QuantizedExperts.

    Must be called AFTER model loading but BEFORE prepare_model_for_kbit_training
    and get_peft_model.

    Memory note: during patching each expert block temporarily holds both the
    original bf16 3D Parameters and the new Linear4bit modules. Peak overhead
    per block is roughly 3x the bf16 block size. Blocks are processed one at a
    time and freed immediately.

    Returns the number of expert blocks patched.
    """
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
    except ImportError as e:
        raise ImportError("Gemma4TextExperts not found -- is this a Gemma 4 model?") from e

    # Collect (parent, attr_name, module) first to avoid mutating while iterating.
    targets: list[tuple[nn.Module, str, nn.Module]] = []
    for full_name, module in model.named_modules():
        if isinstance(module, Gemma4TextExperts):
            parts = full_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            targets.append((parent, parts[-1], module))

    if not targets:
        logger.warning("No Gemma4TextExperts found -- model may not be Gemma 4 MoE")
        return 0

    for parent, attr, module in targets:
        gate_up_data = module.gate_up_proj.data  # [E, 2*I, H]
        down_data = module.down_proj.data         # [E, H, I]

        quantized = _QuantizedExperts(
            num_experts=module.num_experts,
            hidden_dim=module.hidden_dim,
            intermediate_dim=module.intermediate_dim,
            act_fn=module.act_fn,
            gate_up_data=gate_up_data,
            down_data=down_data,
            compute_dtype=compute_dtype,
        )

        # Free original Parameters before replacing to avoid double-peak.
        del module.gate_up_proj
        del module.down_proj
        torch.cuda.empty_cache()

        setattr(parent, attr, quantized)
        logger.debug("Patched %s.%s (%d experts)", type(parent).__name__, attr, quantized.num_experts)

    torch.cuda.empty_cache()
    logger.info("Patched %d Gemma4TextExperts blocks to Linear4bit", len(targets))
    return len(targets)
