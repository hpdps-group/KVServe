# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from .config import MooncakeCompressionConfig


@dataclass(frozen=True)
class HeadSelection:
    low_counts: dict[int, int]
    local_orders: dict[int, tuple[int, ...]]


def _load_scores(path: str) -> np.ndarray:
    score_path = Path(path)
    if not score_path.is_file():
        raise ValueError(f"Compression score file does not exist: {path}")
    suffix = score_path.suffix.lower()
    if suffix == ".npy":
        values = np.load(score_path)
    elif suffix in (".pt", ".pth"):
        loaded = torch.load(score_path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            loaded = loaded.get("scores", loaded.get("head_scores"))
        if not isinstance(loaded, torch.Tensor):
            raise ValueError("Torch score file must contain a tensor or scores key.")
        values = loaded.float().numpy()
    elif suffix == ".json":
        import json

        values = np.asarray(json.loads(score_path.read_text()), dtype=np.float32)
    else:
        values = np.genfromtxt(score_path, delimiter=",", dtype=np.float32)
        if values.ndim == 2:
            values = values[~np.all(np.isnan(values), axis=1)]
            values = values[:, ~np.all(np.isnan(values), axis=0)]
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"Compression scores must be a finite 2-D matrix: {path}")
    return values


def build_head_selection(
    specs: dict[str, KVCacheSpec],
    config: MooncakeCompressionConfig,
    tp_rank: int,
    tp_size: int,
) -> HeadSelection:
    fa_specs = [
        (name, spec)
        for name, spec in specs.items()
        if isinstance(spec, FullAttentionSpec)
    ]
    if config.split_type == "layer":
        return HeadSelection(
            low_counts={extract_layer_index(name): 0 for name, _ in fa_specs},
            local_orders={
                extract_layer_index(name): tuple(range(spec.num_kv_heads))
                for name, spec in fa_specs
            },
        )

    assert config.score_path is not None
    scores = _load_scores(config.score_path)
    mask = np.zeros(scores.size, dtype=np.bool_)
    candidate_offsets: list[int] = []
    for layer_name, base_spec in fa_specs:
        spec = base_spec
        assert isinstance(spec, FullAttentionSpec)
        layer_index = extract_layer_index(layer_name)
        if layer_index >= scores.shape[0]:
            raise ValueError(
                f"Score matrix has {scores.shape[0]} layers but layer "
                f"{layer_index} is required."
            )
        local_heads = spec.num_kv_heads
        if scores.shape[1] not in (local_heads, local_heads * tp_size):
            raise ValueError(
                "Score head count does not match the local or TP-global KV "
                f"head count for {layer_name}: scores={scores.shape[1]}, "
                f"local={local_heads}, tp={tp_size}."
            )
        row_start = layer_index * scores.shape[1]
        candidate_offsets.extend(range(row_start, row_start + scores.shape[1]))
    candidate_array = np.asarray(candidate_offsets, dtype=np.int64)
    low_count = round(candidate_array.size * config.hybrid_ratio)
    if low_count:
        flat_scores = scores.reshape(-1)
        selected = np.argsort(flat_scores[candidate_array], kind="stable")[:low_count]
        mask[candidate_array[selected]] = True
    mask = mask.reshape(scores.shape)

    low_counts: dict[int, int] = {}
    local_orders: dict[int, tuple[int, ...]] = {}
    for layer_name, base_spec in fa_specs:
        spec = base_spec
        assert isinstance(spec, FullAttentionSpec)
        layer_index = extract_layer_index(layer_name)
        local_heads = spec.num_kv_heads
        if mask.shape[1] == local_heads:
            local_mask = mask[layer_index]
        elif mask.shape[1] == local_heads * tp_size:
            start = tp_rank * local_heads
            local_mask = mask[layer_index, start : start + local_heads]
        else:
            raise ValueError(
                "Score head count does not match the local or TP-global KV "
                f"head count for {layer_name}: scores={mask.shape[1]}, "
                f"local={local_heads}, tp={tp_size}."
            )
        low = np.flatnonzero(local_mask).tolist()
        high = np.flatnonzero(~local_mask).tolist()
        low_counts[layer_index] = len(low)
        local_orders[layer_index] = tuple(low + high)
    return HeadSelection(low_counts=low_counts, local_orders=local_orders)
