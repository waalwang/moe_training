# MoE Training (Gemma 4 26B)

Weighted SFT and DPO training pipeline for Mixture-of-Experts models.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires `transformers>=4.52` for Gemma 4 support.

## Usage

### Full pipeline (QLoRA)

```bash
# 1. SFT -- train LoRA adapters on base model
python train.py --profile cloud

# 2. Eval -- pick the best SFT checkpoint
#    (eval merges adapters in memory, no disk writes)

# 3. DPO -- point at the best SFT adapter checkpoint
#    (auto-detects adapter, merges in memory, trains fresh DPO adapters)
python train_dpo.py --profile cloud --checkpoint outputs/sft/checkpoint-XXXX
```

When `--checkpoint` points to a LoRA adapter directory, `model_loader.py`
automatically loads the base model, merges the adapter in memory, and
returns the merged model. No manual merge step or extra disk space needed.
Each stage gets its own independent set of LoRA adapters.

### Attention + shared MLP adapters

Use if attention-only plateaus:

```bash
python train.py --profile cloud --qlora-section qlora_with_mlp
python train_dpo.py --profile cloud --checkpoint outputs/sft/checkpoint-XXXX \
    --qlora-section qlora_with_mlp
```

### Full fine-tune (multi-GPU)

```bash
python train.py --profile cloud_full
python train_dpo.py --profile cloud_full --checkpoint outputs/sft/checkpoint-XXXX
```

### Dry run (verify data loading + model init)

```bash
python train.py --profile cloud --dry-run
python train_dpo.py --profile cloud --dry-run
```

## Hardware profiles

| Profile | GPU | Batch | Precision | Mode |
|---|---|---|---|---|
| `local` | TITAN RTX 24GB | 1 | fp16 | QLoRA only |
| `cloud` | RTX PRO 6000 96GB | 2 | bf16 | QLoRA |
| `cloud_full` | Multi-GPU | 1 | bf16 | Full FT |

## Project structure

```
configs/default.yaml      -- model, LoRA, profile, and training configs
model_loader.py           -- shared model/tokenizer loading (handles Gemma 4 arch)
data_loader.py            -- SFT chain parquet -> HF Dataset
dpo_data_loader.py        -- DPO pair parquet -> HF Dataset
train.py                  -- weighted SFT entry point
train_dpo.py              -- weighted DPO entry point
weighted_sft_trainer.py   -- per-turn loss weighting for SFT
weighted_dpo_trainer.py   -- per-example chosen weighting for DPO
```
