#!/usr/bin/env python3
"""Assemble ProfileLibrary JSONs from measured profiles + speed CSVs.

Reads profiles/raw/ and writes:
  profiles/libraries/{model}/{dataset}/{machine}.json
  profiles/catalog.json
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kvserve_v1.compression.controller.speed_table import speed_csv_for

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "profiles" / "raw"
OUT = ROOT / "profiles" / "libraries"
CATALOG = ROOT / "profiles" / "catalog.json"

INPUT_LENGTH = 4096
ACC_EDGES = [0.0, 0.96, 0.99, 1.20]
MAX_CANDIDATES = 3
GB_TO_MB = 1024.0

CODEC_MAP = {
    "ans": "ans",
    "bitcomp": "bitcomp",
    "lz4": "lz4",
    "nvcomp": "ans",
}
QUANT_MAP = {"head": "quantizer", "layer": "cachegen"}
AXIS_NAME = {1: "token", 2: "channel", 3: "token"}


def _axis_to_name(axis: Any, default: str) -> str:
    if isinstance(axis, str):
        return axis
    if isinstance(axis, int):
        return AXIS_NAME.get(axis, default)
    if isinstance(axis, list) and axis:
        return AXIS_NAME.get(int(axis[0]), default)
    return default


def load_speed_table(csv_path: Path) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """type -> {length: (prefill_mbps, decode_mbps)}."""
    table: Dict[str, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ctype = row["type"].strip().lower()
            try:
                length = int(float(row["input_length"]))
                pre = float(row["prefill_throughput(GB/s)"]) * GB_TO_MB
                dec = float(row["decode_throughput(GB/s)"]) * GB_TO_MB
            except (TypeError, ValueError):
                continue
            table[ctype][length] = (pre, dec)
    return table


def nearest_length(lengths: List[int], target: int) -> Optional[int]:
    if not lengths:
        return None
    return min(lengths, key=lambda x: (abs(x - target), x))


def component_harmonic(table, ctype: str, length: int) -> Optional[float]:
    data = table.get(ctype.lower())
    if not data:
        return None
    nearest = nearest_length(list(data), length)
    if nearest is None:
        return None
    pre, dec = data[nearest]
    if pre > 0 and dec > 0:
        return 2.0 / (1.0 / pre + 1.0 / dec)
    return pre or dec or None


def pipeline_harmonic(table, components: List[str], length: int) -> Optional[float]:
    inv = 0.0
    for c in components:
        s = component_harmonic(table, c, length)
        if s is None or s <= 0:
            return None
        inv += 1.0 / s
    return 1.0 / inv if inv else None


def pipeline_components(raw: Dict[str, Any]) -> List[str]:
    comps: List[str] = []
    tf = (raw.get("transformer_config") or {}).get("transform_type")
    if tf and tf != "none":
        comps.append("hadamard" if tf.lower() == "hadamard" else tf.lower())
    quant = raw.get("quantizer_config") or {}
    if quant:
        comps.append(QUANT_MAP.get(quant.get("split_type", "head"), "quantizer"))
    codec = raw.get("codec_config") or {}
    algo = (codec.get("nvcomp_algorithm") or codec.get("codec_type") or "")
    mapped = CODEC_MAP.get(str(algo).lower()) if algo else None
    if mapped:
        comps.append(mapped)
    return comps


def compression_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    tf = dict(raw.get("transformer_config") or {})
    quant = dict(raw.get("quantizer_config") or {})
    codec = dict(raw.get("codec_config") or {})
    pipeline = []
    if tf.get("transform_type") and tf.get("transform_type") != "none":
        pipeline.append("transformer")
    if quant:
        pipeline.append("quantizer")
    if codec:
        pipeline.append("codec")
    seed = tf.get("seed")
    if isinstance(seed, str):
        try:
            tf["seed"] = int(seed, 0)
        except ValueError:
            tf["seed"] = 0xC0FEBABE
    return {
        "enabled": True,
        "pipeline": pipeline,
        "transformer_config": tf or None,
        "quantizer_config": quant or None,
        "codec_config": codec or None,
        "min_compress_size": 0,
    }


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


def build_profile(
    raw: Dict[str, Any],
    *,
    model: str,
    dataset: str,
    machine: str,
    table: Dict[str, Dict[int, Tuple[float, float]]],
    source: str,
) -> Optional[Dict[str, Any]]:
    comps = pipeline_components(raw)
    speed = pipeline_harmonic(table, comps, INPUT_LENGTH)
    if speed is None:
        return None
    acc_pct = float(raw.get("accuracy", 100.0))
    if acc_pct > 110:
        return None
    acc = acc_pct / 100.0
    cr = float(raw.get("compression_ratio") or raw.get("cr") or 1.0)
    if cr <= 1.0:
        return None
    bcrit = (1.0 - 1.0 / cr) * speed
    config_id = raw.get("config_id", 0)
    pid = f"{slug(model)}_{slug(dataset)}_{slug(machine)}_c{config_id}"
    return {
        "profile_id": pid,
        "metadata": {
            "model_name": model,
            "dataset": dataset,
            "machine": machine,
            "config_id": config_id,
            "source": source,
            "input_length": INPUT_LENGTH,
            "pipeline_components": comps,
        },
        "compression_config": compression_config(raw),
        "performance_metrics": {
            "compression_ratio": cr,
            "harmonic_speed_mbps": speed,
            "critical_bandwidth_mbps": bcrit,
            "encode_speed_mbps": speed,
            "decode_speed_mbps": speed,
        },
        "quality_metrics": {
            "accuracy": acc,
            "metric_name": "relative_task_accuracy",
        },
    }


def pareto_keep(profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept = []
    for p in profiles:
        cr, s = p["performance_metrics"]["compression_ratio"], p["performance_metrics"]["harmonic_speed_mbps"]
        dominated = False
        for q in profiles:
            if q is p:
                continue
            qcr, qs = q["performance_metrics"]["compression_ratio"], q["performance_metrics"]["harmonic_speed_mbps"]
            if (qcr >= cr and qs >= s) and (qcr > cr or qs > s):
                dominated = True
                break
        if not dominated:
            kept.append(p)
    kept.sort(key=lambda p: -p["performance_metrics"]["critical_bandwidth_mbps"])
    return kept


def make_intervals(pareto: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not pareto:
        return []
    crits = [p["performance_metrics"]["critical_bandwidth_mbps"] for p in pareto]
    # High-B → low-B boundaries: inf, crits..., 0
    bounds = [math.inf] + crits + [0.0]
    intervals = []
    for i in range(len(bounds) - 1):
        b_high, b_low = bounds[i], bounds[i + 1]
        if b_high == b_low:
            continue
        x_low = 0.0 if math.isinf(b_high) else 1.0 / b_high
        x_high = 1.0 if b_low <= 0 else 1.0 / b_low
        if x_high <= x_low:
            continue
        # theoretical optimum: the profile whose B_crit equals b_high, else nearest lower-B profile
        if i == 0:
            opt = min(pareto, key=lambda p: p["performance_metrics"]["compression_ratio"])
        else:
            opt = pareto[i - 1]
        # candidates: opt plus neighbors in the B_crit-sorted list
        idx = pareto.index(opt)
        cand_idx = sorted({max(0, idx - 1), idx, min(len(pareto) - 1, idx + 1)})[:MAX_CANDIDATES]
        candidates = [pareto[j]["profile_id"] for j in cand_idx]
        intervals.append({
            "interval_id": len(intervals),
            "x_range": [x_low, x_high],
            "B_range": [None if math.isinf(b_high) else b_high, b_low],
            "B_range_mbps": [
                None if math.isinf(b_high) else round(b_high, 3),
                round(b_low, 3),
            ],
            "model_optimal_profile_id": opt["profile_id"],
            "candidate_profile_ids": candidates,
        })
        # drop the redundant B_range with None for JSON cleanliness
        intervals[-1].pop("B_range", None)
    return intervals


def bucketize(profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets = []
    for i, (lo, hi) in enumerate(zip(ACC_EDGES, ACC_EDGES[1:])):
        members = [
            p for p in profiles
            if lo <= p["quality_metrics"]["accuracy"] < hi
        ]
        pareto = pareto_keep(members)
        if not pareto:
            continue
        buckets.append({
            "bucket_id": len(buckets),
            "acc_range": [lo, hi],
            "pareto_profiles": pareto,
            "bandwidth_intervals": make_intervals(pareto),
            "n_raw_in_bucket": len(members),
        })
    return buckets


def write_library(
    profiles: List[Dict[str, Any]],
    *,
    model: str,
    dataset: str,
    machine: str,
    source: str,
) -> Optional[Path]:
    if not profiles:
        return None
    buckets = bucketize(profiles)
    if not buckets:
        return None
    payload = {
        "library_metadata": {
            "model_name": model,
            "dataset": dataset,
            "machine": machine,
            "source": source,
            "input_length": INPUT_LENGTH,
            "num_profiles": sum(len(b["pareto_profiles"]) for b in buckets),
            "num_raw_profiles": len(profiles),
            "description": (
                "Measured compression configs + component speeds, "
                "assembled into a fused ProfileLibrary "
                "(accuracy buckets, 1/B intervals, Pareto candidates)."
            ),
            "units": {
                "B_mbps": "MB/s (controller argument)",
                "harmonic_speed_mbps": "MB/s",
                "accuracy": "relative to uncompressed baseline, 0-1",
            },
        },
        "accuracy_buckets": buckets,
    }
    path = OUT / model / dataset / f"{machine}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def fused_bo_raws() -> List[Dict[str, Any]]:
    src = RAW / "fused" / "param_search_Qwen2.5-7B-Instruct" / "tolerance_3_results.json"
    if not src.exists():
        return []
    out = []
    for row in json.loads(src.read_text()):
        out.append({
            "config_id": row.get("config_id"),
            "transformer_config": {
                "transform_type": row.get("transform_type") or "hadamard",
                "seed": 0x3333,
            },
            "quantizer_config": {
                "model_name": "Qwen2.5-7B-Instruct",
                "hybrid_ratio": row.get("heads_selection", 0.5),
                "high_key_max_value": row.get("high_key_max_value"),
                "high_value_max_value": row.get("high_value_max_value"),
                "low_key_max_value": row.get("low_key_max_value"),
                "low_value_max_value": row.get("low_value_max_value"),
                "axis_key": _axis_to_name(row.get("axis_key"), "channel"),
                "axis_value": _axis_to_name(row.get("axis_value"), "token"),
                "split_type": "head",
            },
            "codec_config": {
                "codec_type": "nvcomp",
                "nvcomp_algorithm": "ANS",
                "data_type": "|u1",
            },
            "accuracy": row.get("accuracy", 100.0),
            "compression_ratio": row.get("cr"),
        })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    skipped = []

    profile_root = RAW / "main" / "profiles"
    models = sorted(p.name for p in profile_root.iterdir() if p.is_dir())
    machines = ["5090", "H100", "6000P", "4090"]
    datasets = ["qasper", "gsm8k", "humaneval", "multi_news"]

    for model in models:
        for dataset in datasets:
            cfg_path = profile_root / model / f"{dataset}.json"
            if not cfg_path.exists():
                continue
            raws = json.loads(cfg_path.read_text())
            for machine in machines:
                csv_path = speed_csv_for(machine, model)
                if csv_path is None:
                    skipped.append((model, dataset, machine, "no_speed_csv"))
                    continue
                table = load_speed_table(csv_path)
                built = []
                for raw in raws:
                    p = build_profile(
                        raw, model=model, dataset=dataset, machine=machine,
                        table=table, source=f"main:{cfg_path.name}+{csv_path.name}",
                    )
                    if p:
                        built.append(p)
                path = write_library(
                    built, model=model, dataset=dataset, machine=machine, source="main",
                )
                if path is None:
                    skipped.append((model, dataset, machine, f"no_profiles nraw={len(raws)}"))
                    continue
                meta = json.loads(path.read_text())["library_metadata"]
                entries.append({
                    "model": model,
                    "dataset": dataset,
                    "machine": machine,
                    "source": "main",
                    "n_raw": meta["num_raw_profiles"],
                    "n_profiles": meta["num_profiles"],
                    "path": str(path.relative_to(ROOT / "profiles")),
                })

    # fused BO (Qwen2.5-7B), speeds from main 5090 / fused csv fallback
    fused_raws = fused_bo_raws()
    for machine, csv_path in [
        ("5090", speed_csv_for("5090", "Qwen2.5-7B-Instruct")),
        ("5090_fused_speed", RAW / "fused" / "Qwen2.5-7B-Instruct_speed.csv"),
    ]:
        if not csv_path or not csv_path.exists():
            continue
        # only emit one fused_bo library on 5090 using main 5090 speeds;
        # fused speed csv is a fallback if main csv missing
        if machine != "5090" and speed_csv_for("5090", "Qwen2.5-7B-Instruct"):
            continue
        table = load_speed_table(csv_path)
        built = []
        for raw in fused_raws:
            p = build_profile(
                raw, model="Qwen2.5-7B-Instruct", dataset="fused_bo",
                machine="5090", table=table, source="fused:tolerance_3_results.json",
            )
            if p:
                built.append(p)
        path = write_library(
            built, model="Qwen2.5-7B-Instruct", dataset="fused_bo",
            machine="5090", source="fused",
        )
        if path:
            meta = json.loads(path.read_text())["library_metadata"]
            entries.append({
                "model": "Qwen2.5-7B-Instruct",
                "dataset": "fused_bo",
                "machine": "5090",
                "source": "fused",
                "n_raw": meta["num_raw_profiles"],
                "n_profiles": meta["num_profiles"],
                    "path": str(path.relative_to(ROOT / "profiles")),
            })

    default = next(
        (e for e in entries
         if e["model"] == "Qwen2.5-7B-Instruct" and e["dataset"] == "qasper" and e["machine"] == "5090"),
        entries[0] if entries else None,
    )
    catalog = {
        "default": default,
        "how_to_load": (
            "from kvserve_v1.compression.controller.profile_catalog import load; "
            "lib = load('Qwen2.5-7B-Instruct', 'qasper', '5090')"
        ),
        "controller": (
            "DynamicOnlineController.from_library_json(path, machine, model)"
        ),
        "n_libraries": len(entries),
        "entries": entries,
        "skipped": [{"model": a, "dataset": b, "machine": c, "reason": d} for a, b, c, d in skipped],
    }
    CATALOG.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"wrote {len(entries)} libraries -> {OUT}")
    print(f"catalog -> {CATALOG}")
    if default:
        print(f"default -> {default['path']} ({default['n_profiles']} pareto profiles)")


if __name__ == "__main__":
    main()
