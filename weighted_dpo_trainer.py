"""
weighted_dpo_trainer.py

Extends TRL's DPOTrainer with optional per-example chosen-side weighting.

When chosen_weighting=True, each example's DPO loss is scaled by its
chosen_weight before reduction. Higher-quality chosen trajectories
contribute more to the gradient.

When chosen_weighting=False (default), behaves identically to DPOTrainer.
"""

from __future__ import annotations

import logging

import torch
from trl import DPOTrainer

logger = logging.getLogger(__name__)


class WeightedDPOTrainer(DPOTrainer):

    def __init__(self, chosen_weighting: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.chosen_weighting = chosen_weighting

    def _compute_loss(self, model, inputs, return_outputs):
        if self.chosen_weighting:
            w = inputs.pop("chosen_weight", None)
        else:
            inputs.pop("chosen_weight", None)
            w = None

        result = super()._compute_loss(model, inputs, return_outputs)

        if w is None:
            return result

        loss = result[0] if return_outputs else result
        w = w.to(loss.device, dtype=loss.dtype)
        loss = loss * w.mean()

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["chosen_weight_mean"].append(w.mean().item())

        return (loss, result[1]) if return_outputs else loss
