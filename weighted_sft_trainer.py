"""
weighted_sft_trainer.py

Extends TRL's SFTTrainer to support per-turn loss weights.
Each training example carries a `turn_weights` field -- a list of floats,
one per conversation turn. Assistant turn weights scale token-level loss;
user/system turns have weight 0 (already masked by labels=-100).

Falls back to plain unweighted SFT when turn_weights is absent.

NOTE: TRL's DataCollatorForLanguageModeling only forwards standard columns
(input_ids, labels, attention_mask, etc.) and silently drops turn_weights.
We wrap the collator to preserve turn_weights through to compute_loss.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any

import torch
from transformers.trainer_utils import sort_checkpoints
from trl import SFTTrainer

logger = logging.getLogger(__name__)


class _WeightedCollatorWrapper:
    """Wraps TRL's data collator to preserve turn_weights in the batch."""

    def __init__(self, inner_collator):
        self.inner = inner_collator

    def __call__(self, features: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        turn_weights = [f.pop("turn_weights", None) for f in features]
        batch = self.inner(features, **kwargs)
        if any(tw is not None for tw in turn_weights):
            batch["turn_weights"] = turn_weights
        return batch


def _build_token_weights(labels: torch.Tensor, turn_weights: list[list[float]]) -> torch.Tensor:
    """Map per-turn weights onto token positions using label mask transitions.

    labels has -100 for masked tokens (user/system turns, padding) and real
    token ids for assistant turns. Each contiguous run of non-(-100) tokens
    is one assistant turn region, assigned the next assistant weight from
    turn_weights.
    """
    batch_size, seq_len = labels.shape
    token_w = torch.ones_like(labels, dtype=torch.float32)

    for b in range(batch_size):
        row_labels = labels[b]
        asst_weights = [w for w in turn_weights[b] if w > 0]
        region_idx = 0
        in_region = False
        cur_weight = 1.0

        for t in range(seq_len):
            if row_labels[t] != -100:
                if not in_region:
                    in_region = True
                    cur_weight = asst_weights[region_idx] if region_idx < len(asst_weights) else 1.0
                    region_idx += 1
                token_w[b, t] = cur_weight
            else:
                in_region = False
                token_w[b, t] = 0.0

    return token_w


class WeightedSFTTrainer(SFTTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_collator = _WeightedCollatorWrapper(self.data_collator)
        self._acc_correct = 0
        self._acc_total = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        turn_weights = inputs.pop("turn_weights", None)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )

        logits = outputs.logits
        labels = inputs["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != -100

        if turn_weights is None:
            loss = outputs.loss
        else:
            token_w = _build_token_weights(shift_labels, turn_weights)
            token_w = token_w.to(shift_logits.device, dtype=shift_logits.dtype)

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            per_token_loss = per_token_loss.view(shift_labels.size())

            weighted_loss = per_token_loss * token_w
            loss = weighted_loss.sum() / token_w.sum().clamp(min=1)

        with torch.no_grad():
            preds = shift_logits.argmax(dim=-1)
            correct = ((preds == shift_labels) & mask).sum().item()
            total = mask.sum().item()

        self._acc_correct += correct
        self._acc_total += total

        return (loss, outputs) if return_outputs else loss

    def _flush_accuracy(self):
        if self._acc_total > 0:
            acc = self._acc_correct / self._acc_total
            self._acc_correct = 0
            self._acc_total = 0
            return acc
        return None

    def log(self, logs, *args, **kwargs):
        acc = self._flush_accuracy()
        if acc is not None:
            logs["accuracy"] = acc
        super().log(logs, *args, **kwargs)

    def evaluate(self, *args, **kwargs):
        self._acc_correct = 0
        self._acc_total = 0
        return super().evaluate(*args, **kwargs)

    def _save_checkpoint(self, model, trial=None):
        limit = self.args.save_total_limit
        if limit and limit > 0:
            run_dir = self._get_output_dir(trial=trial)
            keep = limit - 1
            best = self.state.best_model_checkpoint
            checkpoints = sort_checkpoints(run_dir, use_mtime=True)
            protect_best = best if keep > 0 else None
            if len(checkpoints) > keep:
                to_delete = []
                for cp in checkpoints:
                    if len(checkpoints) - len(to_delete) <= keep:
                        break
                    if cp != protect_best:
                        to_delete.append(cp)
                for cp in to_delete:
                    logger.info("Pre-save rotation: removing %s", cp)
                    shutil.rmtree(cp, ignore_errors=True)

        saved_limit = self.args.save_total_limit
        self.args.save_total_limit = None
        try:
            super()._save_checkpoint(model, trial)
        finally:
            self.args.save_total_limit = saved_limit
