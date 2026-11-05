# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the KVServe project
"""Small CUDA correctness check for configurable SGLang codec profiles.

No service or transport is started. Each case encodes into a leased bank,
copies that payload to a receive slot, and restores both K and V. The default
fused or nondefault C++ path is compared with the same standard Python profile.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

from kvserve_v1.compression.runtime import CacheSpec, CompressionConfig, build_memory_plan
from kvserve_v1.compression.runtime.fused_runtime import create_runtime


PROFILES = (
    ("default-fused", None, "auto"),
    ("quantizer-codec", ("quantizer", "codec"), "standard"),
    ("transformer-quantizer", ("transformer", "quantizer"), "standard"),
    ("codec-only", ("codec",), "standard"),
)


def make_caches(specs, capacity, device):
    caches = {}
    for name, spec in specs.items():
        caches[name] = tuple(
            torch.randn(
                (2 * capacity, spec.num_kv_heads, 1, spec.head_size),
                dtype=torch.bfloat16,
                device=device,
            )
            for _ in (0, 1)
        )
    return caches


def round_trip(runtime, caches, source, target, part, count, capacity):
    for pair in caches.values():
        pair[part][capacity:].fill_(17)
    source_before = {name: pair[part][:capacity].clone() for name, pair in caches.items()}
    other_before = {name: pair[1 - part].clone() for name, pair in caches.items()}
    output, auxiliary = runtime.output_slots[0]
    encoded = runtime.encode_into(
        0,
        [source],
        part,
        valid_tokens_by_group={0: count},
        output=output,
        auxiliary=auxiliary,
    )
    encoded.event.synchronize()
    payload_bytes = encoded.codec_bytes
    aux_bytes = runtime.aux_bytes(part, {0: count})
    if not 0 < payload_bytes <= output.numel() or encoded.raw_tail_bytes:
        raise AssertionError("Invalid encoded size or unexpected raw tail")
    slot = runtime.slots[0]
    slot.arena_a[:payload_bytes].copy_(output[:payload_bytes])
    slot.aux[:aux_bytes].copy_(auxiliary[:aux_bytes])
    torch.cuda.synchronize()
    runtime.decode(
        0,
        [target],
        part,
        payload_bytes,
        encoded.u8_bytes,
        valid_tokens_by_group={0: count},
    ).synchronize()
    for name, pair in caches.items():
        if not torch.equal(pair[part][:capacity], source_before[name]):
            raise AssertionError("Decode changed source pages")
        if not torch.equal(pair[1 - part], other_before[name]):
            raise AssertionError("Decode changed the other K/V part")
        if not torch.all(pair[part][capacity + count :] == 17).item():
            raise AssertionError("Decode changed an unrelated destination page")
    return payload_bytes, aux_bytes


def run_case(name, stages, backend, device, overrides=None):
    capacity, layers, heads = 64, 2, 2
    values = {
        "enable-compression-pipeline": True,
        "logical_blocks_per_chunk": capacity,
        "slot_count": 1,
    }
    if stages is not None:
        values["pipeline"] = stages
    values.update(overrides or {})
    config = CompressionConfig.from_extra_config({"compression": values})
    specs = {
        f"layer.{index}": CacheSpec(index, 1, heads, 128, torch.bfloat16)
        for index in range(layers)
    }
    originals = make_caches(specs, capacity, device)
    candidate_caches = {name: tuple(t.clone() for t in pair) for name, pair in originals.items()}
    reference_caches = {name: tuple(t.clone() for t in pair) for name, pair in originals.items()}
    plan = build_memory_plan(
        config, specs, blocks_per_logical=1, num_layers=layers, device=device,
        warm_cpp_backend=True,
    )
    runtimes = []
    try:
        candidate = create_runtime(
            plan, "HND", 1, device, backend=backend, output_slots=1,
            standard_backend="cpp",
        )
        runtimes.append(candidate)
        reference = create_runtime(
            plan, "HND", 1, device, backend="standard", output_slots=1,
            standard_backend="python",
        )
        runtimes.append(reference)
        candidate.bind_kv_caches(candidate_caches, dict.fromkeys(specs, 0))
        reference.bind_kv_caches(reference_caches, dict.fromkeys(specs, 0))
        candidate_kind = type(candidate).__name__
        if backend == "auto":
            if candidate_kind != "FusedCompressionRuntime":
                raise AssertionError(f"Expected fused execution, got {candidate_kind}")
        elif candidate_kind != "CompressionRuntime" or not all(
            pipeline.pipeline_backend == "cpp" and pipeline._use_cpp()
            for pipeline in candidate.pipelines
        ):
            raise AssertionError("Nondefault profile did not use standard C++ execution")
        if not all(
            pipeline.pipeline_backend == "python" and not pipeline._use_cpp()
            for pipeline in reference.pipelines
        ):
            raise AssertionError("Reference did not use the standard Python pipeline")
        if name == "head-score-levels":
            if plan.head_selection.local_orders != {0: (1, 0), 1: (0, 1)}:
                raise AssertionError("Head score fixture did not exercise head reordering")
            if any(len(part.sections) != 2 for part in plan.layout.layer_parts):
                raise AssertionError("Head score fixture did not exercise both quantization sections")
        ptrs, sizes = candidate.registered_memory()
        if len(ptrs) != len(sizes) or any(not ptr or size <= 0 for ptr, size in zip(ptrs, sizes)):
            raise AssertionError("Runtime exposed an empty registration")

        cases = []
        for count in (17, capacity):
            source = list(range(count))
            target = list(range(capacity, capacity + count))
            for part in (0, 1):
                candidate_bytes, candidate_aux = round_trip(
                    candidate, candidate_caches, source, target, part, count, capacity
                )
                reference_bytes, reference_aux = round_trip(
                    reference, reference_caches, source, target, part, count, capacity
                )
                max_reference_difference = 0.0
                exact_reference = True
                exact_original = True
                for layer in specs:
                    actual = candidate_caches[layer][part][target]
                    expected = reference_caches[layer][part][target]
                    original = originals[layer][part][source]
                    exact_reference &= torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                    exact_original &= torch.equal(actual.view(torch.uint8), original.view(torch.uint8))
                    max_reference_difference = max(
                        max_reference_difference, (actual.float() - expected.float()).abs().max().item()
                    )
                    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.01, atol=0.01)
                if name == "codec-only" and not exact_original:
                    raise AssertionError("Codec-only restoration must preserve BF16 bytes")
                cases.append({
                    "tokens": count,
                    "part": "K" if part == 0 else "V",
                    "candidate_payload_bytes": candidate_bytes,
                    "reference_payload_bytes": reference_bytes,
                    "candidate_aux_bytes": candidate_aux,
                    "reference_aux_bytes": reference_aux,
                    "reference_exact_bf16": exact_reference,
                    "max_reference_abs_difference": max_reference_difference,
                    "codec_only_exact_original": exact_original if name == "codec-only" else None,
                    "source_other_part_and_unrelated_destination_unchanged": True,
                })
        return {
            "name": name,
            "pipeline": config.pipeline,
            "effective_config": asdict(config),
            "section_levels": {
                "K" if part == 0 else "V": [
                    [section.num_levels for section in layer.sections]
                    for layer in plan.layout.parts(part)
                ]
                for part in (0, 1)
            },
            "candidate": candidate_kind,
            "candidate_pipeline_backend": backend if backend == "auto" else "cpp",
            "reference": "standard-python",
            "chunk_tokens": capacity,
            "case_results": cases,
        }
    finally:
        for runtime in reversed(runtimes):
            runtime.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-set", choices=("baseline", "extra"), default="baseline")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.manual_seed(2186)
    backend_before = os.environ.get("KVSERVE_COMPRESSION_BACKEND")
    result = {
        "passed": False,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "profiles": [],
    }
    try:
        if args.profile_set == "extra":
            scores_path = args.output.with_name(args.output.stem + "-head-scores.json")
            if scores_path.exists():
                raise FileExistsError(scores_path)
            scores_path.write_text(json.dumps([[3.0, 0.0], [1.0, 2.0]]) + "\n")
            levels = {
                "high_key_num_levels": 16,
                "low_key_num_levels": 8,
                "high_value_num_levels": 16,
                "low_value_num_levels": 8,
            }
            profiles = (
                ("quantizer-only", ("quantizer",), "standard", {}),
                ("transformer-codec", ("transformer", "codec"), "standard", {}),
                ("quantizer-codec-levels", ("quantizer", "codec"), "standard", levels),
                ("head-score-levels", ("quantizer", "codec"), "standard", {
                    **levels, "split_type": "head", "score_path": str(scores_path),
                    "hybrid_ratio": 0.5,
                }),
            )
        else:
            profiles = tuple((*profile, {}) for profile in PROFILES)
        for name, stages, backend, overrides in profiles:
            row = run_case(name, stages, backend, device, overrides)
            result["profiles"].append(row)
            print(f"{name}: {len(row['case_results'])} checks passed", flush=True)
        if os.environ.get("KVSERVE_COMPRESSION_BACKEND") != backend_before:
            raise AssertionError("Execution mutated the process-wide pipeline backend")
        result["passed"] = True
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
