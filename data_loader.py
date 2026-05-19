"""
data_loader.py

Loads SFT chain parquet files and converts them into the format expected
by TRL's SFTTrainer + per-example weights for weighted SFT.

Each row has:
  - messages: list of {"role": ..., "content": ...}
  - turn_weights: list of floats (per-turn loss scaling)
"""

from __future__ import annotations

import glob
import json
import logging
import os

import numpy as np
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


def _compute_turn_weights(
    turns: list[dict],
    beta: float,
    weight_mode: str = "gate_amplify",
    weight_floor: float = 0.1,
) -> list[float]:
    weights = []
    for t in turns:
        if t["role"] != "assistant":
            weights.append(0.0)
            continue
        att = t.get("attunement_score")
        att_comp = max(weight_floor, att) if att is not None else 0.5

        if weight_mode == "attunement_only":
            weights.append(att_comp)
        elif weight_mode == "additive":
            score_comp = np.log1p(max(0, t.get("score", 0)))
            weights.append(att_comp + beta * score_comp)
        else:  # gate_amplify
            score_comp = np.log1p(max(0, t.get("score", 0)))
            weights.append(att_comp * (1.0 + beta * score_comp))
    return weights


def _load_shard(
    path: str,
    weight_beta: float,
    weight_mode: str,
    weight_floor: float,
    min_chain_depth: int,
    min_total_score: float,
) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    data = table.to_pydict()
    n = table.num_rows

    rows = []
    for i in range(n):
        chain_depth = data["chain_depth"][i]
        total_score = data["total_score"][i]
        if chain_depth < min_chain_depth:
            continue
        if total_score < min_total_score:
            continue

        turns = json.loads(data["turns"][i])
        messages = [{"role": t["role"], "content": t["content"]} for t in turns]
        turn_weights = _compute_turn_weights(
            turns, weight_beta, weight_mode, weight_floor,
        )

        rows.append({
            "messages": messages,
            "turn_weights": turn_weights,
        })

    logger.info("  %s: %d total, %d kept", os.path.basename(path), n, len(rows))
    return rows


def load_sft_dataset(
    data_dir: str,
    test_split: float = 0.05,
    seed: int = 42,
    min_chain_depth: int = 3,
    min_total_score: float = 0.0,
    weight_beta: float = 0.3,
    weight_mode: str = "gate_amplify",
    weight_floor: float = 0.1,
    weight_max: float = 3.0,
) -> DatasetDict:
    if os.path.isfile(data_dir):
        files = [data_dir]
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files at {data_dir}")

    logger.info("Found %d parquet shard(s) in %s", len(files), data_dir)

    rows = []
    for fpath in files:
        rows.extend(_load_shard(fpath, weight_beta, weight_mode,
                                weight_floor, min_chain_depth,
                                min_total_score))

    if not rows:
        raise ValueError(f"No rows loaded from {data_dir}")

    all_nonzero = [w for r in rows for w in r["turn_weights"] if w > 0]
    if all_nonzero:
        arr = np.array(all_nonzero)
        if arr.std() > 1e-8:
            scale = arr.mean()
            for r in rows:
                r["turn_weights"] = [
                    min(w / scale, weight_max) if w > 0 else 0.0
                    for w in r["turn_weights"]
                ]
            arr = np.clip(arr / scale, 0, weight_max)
        else:
            for r in rows:
                r["turn_weights"] = [1.0 if w > 0 else 0.0
                                     for w in r["turn_weights"]]
            arr = np.ones_like(arr)
        logger.info(
            "Loaded %d chains | turn weight stats (assistant only): "
            "mean=%.3f std=%.3f min=%.3f max=%.3f (floor=%.2f, clamp=%.1f)",
            len(rows), arr.mean(), arr.std(), arr.min(), arr.max(),
            weight_floor, weight_max,
        )
    else:
        logger.info("Loaded %d chains | no assistant turns found", len(rows))

    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=test_split, seed=seed)
    logger.info("Split: train=%d, test=%d", len(split["train"]), len(split["test"]))
    return split


def load_from_config(config: dict) -> DatasetDict:
    data_cfg = config["data"]
    return load_sft_dataset(
        data_dir=data_cfg["sft_chain_dir"],
        test_split=data_cfg.get("test_split", 0.05),
        seed=data_cfg.get("seed", 42),
        min_chain_depth=data_cfg.get("min_chain_depth", 3),
        min_total_score=data_cfg.get("min_total_score", 0.0),
        weight_beta=data_cfg.get("weight_beta", 0.3),
        weight_mode=data_cfg.get("weight_mode", "gate_amplify"),
        weight_floor=data_cfg.get("weight_floor", 0.1),
        weight_max=data_cfg.get("weight_max", 3.0),
    )
