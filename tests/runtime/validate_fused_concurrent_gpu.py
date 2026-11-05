# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the KVServe project
"""Two-instance/two-slot GPU regression with independent standard-codec oracle."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['KVSERVE_COMPRESSION_BACKEND'] = 'cpp'
import torch
from kvserve_v1.compression.runtime import (
    CacheSpec, CompressionConfig, CompressionRuntime, build_memory_plan,
)
from kvserve_v1.compression.runtime.fused_runtime import create_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.manual_seed(3817)
    results = []
    instances = []
    try:
        for heads in (2, 8):
            capacity, layers = 1024, 3
            config = CompressionConfig.from_extra_config({'compression': {
                'enable-compression-pipeline': True, 'slot_count': 2,
                'logical_blocks_per_chunk': capacity,
            }})
            specs = {f'layer.{i}': CacheSpec(i, 1, heads, 128, torch.bfloat16)
                     for i in range(layers)}
            caches = {name: tuple(torch.randn(
                (capacity * 4, heads, 1, 128), dtype=torch.bfloat16, device=device)
                for _ in (0, 1)) for name in specs}
            plan = build_memory_plan(config, specs, blocks_per_logical=1,
                                     num_layers=layers, device=device)
            runtime = create_runtime(plan, 'HND', 1, device, output_slots=2)
            oracle = CompressionRuntime(plan, 'HND', 1, device)
            for rt in (runtime, oracle):
                rt.bind_kv_caches(caches, {name: 0 for name in specs})
            instances.append((runtime, oracle, caches, heads))

        def frame(instance, slot, part, count):
            runtime, oracle, caches, heads = instance
            torch.cuda.set_device(device)
            src = list(range(slot * 1024, slot * 1024 + count))
            dst = list(range(2048 + slot * 1024, 2048 + slot * 1024 + count))
            before = [c[part][:2048].clone() for c in caches.values()]
            encoded = oracle.encode(slot, [src], part, valid_tokens_by_group={0: count})
            encoded.event.synchronize()
            oracle.decode(slot, [dst], part, encoded.codec_bytes, encoded.u8_bytes,
                          valid_tokens_by_group={0: count}).synchronize()
            expected = [c[part][dst].clone() for c in caches.values()]
            for c in caches.values():
                c[part][dst] = 0
            torch.cuda.synchronize(device)
            payload, auxiliary = runtime.output_slots[slot]
            encoded = runtime.encode_into(slot, [src], part,
                valid_tokens_by_group={0: count}, output=payload, auxiliary=auxiliary)
            encoded.event.synchronize()
            size = encoded.codec_bytes
            aux_bytes = runtime.aux_bytes(part, {0: count})
            runtime.slots[slot].arena_a[:size].copy_(payload[:size])
            runtime.slots[slot].aux[:aux_bytes].copy_(auxiliary[:aux_bytes])
            torch.cuda.synchronize(device)
            runtime.decode(slot, [dst], part, size, encoded.u8_bytes,
                           valid_tokens_by_group={0: count}).synchronize()
            for c, restored, source in zip(caches.values(), expected, before):
                assert torch.equal(c[part][dst].view(torch.uint8), restored.view(torch.uint8))
                assert torch.equal(c[part][:2048], source), 'source changed'
            return {'heads': heads, 'slot': slot, 'part': part, 'tokens': count,
                    'restored_exact': True, 'source_unchanged': True}

        # KK, KV, VV across both Runtime instances at the same time. Full frames
        # replay prebuilt graphs; 17/513 exercise compressed eager tails.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for parts in ((0, 0), (0, 1), (1, 1)):
                for count in (17, 513, 1024):
                    futures = [pool.submit(frame, instance, slot, part, count)
                               for instance in instances for slot, part in enumerate(parts)]
                    results.extend(f.result() for f in futures)
    finally:
        for runtime, oracle, _, _ in instances:
            runtime.shutdown()
            oracle.shutdown()
        args.output.write_text(json.dumps({'passed': len(results) == 36,
                                          'cases': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
