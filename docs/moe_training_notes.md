# MoE Training Notes -- Gemma 4 26B

## Architecture facts

- Model class: `Gemma4ForConditionalGeneration` (multimodal arch)
- `AutoModelForCausalLM` cannot load it -- `model_loader.py` falls back to `AutoModelForConditionalGeneration`
- 128 experts, top-8 active per token
- 30 transformer layers, all with MoE blocks enabled
- Attention: 16 heads, 8 KV heads, head_dim=256 (sliding layers), 512 (full attention layers)
- Layer pattern: 5 sliding attention + 1 full attention, repeated 5 times
- Expert FFN weights are packed as `nn.Parameter` tensors (not `nn.Linear`), so LoRA cannot target them directly
- Each layer has both a shared MLP (`gate_proj`, `up_proj`, `down_proj` as `nn.Linear`) and the MoE expert block
- Router: `nn.Linear(hidden_size, num_experts)` + per-expert scale -- small, ~360K params per layer

## LoRA target strategy

### Default: attention-only (q_proj, k_proj, v_proj, o_proj)

- Memory scales like fine-tuning a dense ~4B model
- Router and all 128 experts stay frozen -- pretrained specialization preserved
- Sufficient for most SFT/DPO tasks where the base model already has the right knowledge

### Escalation: attention + shared MLP (add gate_proj, up_proj, down_proj)

- The shared MLP runs alongside the MoE block in every layer
- Tuning it gives more representational leverage without touching experts
- Use `--qlora-section qlora_with_mlp` to enable
- Moderate memory increase (3 more linear layers per layer)

### Not recommended: tuning expert weights

- Expert weights are `nn.Parameter` (3D tensors: num_experts x dim x dim), not individual `nn.Linear` modules
- Standard LoRA/PEFT cannot target them
- Custom implementations exist but are fragile and memory-heavy
- If attention + shared MLP is not enough, consider increasing LoRA rank before going down this path

## Router: freeze it

- Training the router risks expert collapse (a few experts hog all tokens)
- The pretrained router is one of the most valuable things from pretraining
- If you must train it, keep the auxiliary load-balancing loss active
- Standard PEFT/LoRA does not target the router by default -- it stays frozen automatically

## Memory budget estimates (QLoRA 4-bit)

| Config | Weights | LoRA adapters | Optimizer | Activations | Total (approx) |
|---|---|---|---|---|---|
| Attn-only, r=16 | ~14GB | ~50MB | ~100MB | ~5-15GB | ~20-30GB |
| Attn+MLP, r=16 | ~14GB | ~120MB | ~240MB | ~5-15GB | ~20-30GB |
| Attn-only, r=64 (DPO) | ~14GB | ~200MB | ~400MB | ~5-15GB | ~20-30GB |

Activation memory depends heavily on sequence length and batch size.
Gradient checkpointing is enabled by default in all profiles.

## Differences from attunement_training (Qwen/dense)

- Model loading handles `AutoModelForConditionalGeneration` fallback
- LoRA targets exclude FFN by default (experts are not targetable, shared MLP is optional)
- Batch sizes reduced (26B weights even in 4-bit are ~14GB vs ~4GB for 7B)
- `transformers>=4.52` required for Gemma 4 model classes
- Deprecated v1 data loading code removed
- Model loading extracted to shared `model_loader.py`

## Text-only vs. multimodal training

- SFT/DPO on text-only data will not update vision encoder or multimodal projector
- Aggressive text-only fine-tuning can degrade vision capabilities (language model layers that process projected image tokens shift)
- If multimodal capability matters, mix text + multimodal data in training or run a separate multimodal stage
