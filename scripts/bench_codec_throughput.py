#!/usr/bin/env python3
"""Measure compress/decompress throughput and theoretical transfer times."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kvserve_v1.compression.manager import KVCompressionAdapter
from kvserve_v1.compression.wire import build_wire

METHODS = [
    ("original_nvcomp", "configs/compression/original_top_nvcomp.json"),
    ("original_lc", "configs/compression/original_top_lc.json"),
    ("fused_nvcomp", "configs/compression/fused_top_nvcomp.json"),
    ("fused_lc", "configs/compression/fused_top_lc.json"),
]

BANDWIDTHS_GBPS = [20, 50, 100, 200]


def load_cfg(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    # Prefer local Llama-3 scores alias used in prior HotpotQA runs.
    qc = cfg.get("quantizer_config") or {}
    if qc.get("model_name") == "Llama-3.1-8B-Instruct":
        scores = (
            ROOT
            / "kvserve_v1/compression/config/duo_config"
            / "Llama-3.1-8B-Instruct_scores.csv"
        )
        if not scores.exists():
            qc["model_name"] = "Llama-3-8B-Instruct"
            cfg["quantizer_config"] = qc
    return cfg


def make_kv(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    device: str,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    # Slightly structured noise so ANS/LC see compressible uint8 after quant.
    # TileLang fused kernels expect bf16 (matches Llama KV cache dtype).
    base = torch.randn(
        num_layers, 2, num_blocks, block_size, num_kv_heads, head_size,
        generator=g, device=device, dtype=torch.bfloat16,
    )
    return (base * 0.35).to(torch.bfloat16).contiguous()


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def timed(fn, warmup: int, iters: int, device: str) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / iters


def bench_one(
    name: str,
    cfg_path: Path,
    kv: torch.Tensor,
    device: str,
    warmup: int,
    iters: int,
) -> dict:
    cfg = load_cfg(cfg_path)
    adapter = KVCompressionAdapter(
        cfg,
        num_kv_heads=kv.shape[4],
        head_size=kv.shape[5],
        model_name=cfg.get("quantizer_config", {}).get("model_name"),
    )

    # JIT / first-touch outside timed region.
    compressed = adapter.compress(kv, request_id=f"warmup-{name}")
    if compressed is None:
        raise RuntimeError(f"{name}: compress returned None")
    _ = adapter.decompress(compressed)
    sync(device)

    def do_compress():
        return adapter.compress(kv, request_id=f"bench-{name}")

    t_comp = timed(do_compress, warmup, iters, device)
    compressed = do_compress()
    sync(device)

    def do_decomp():
        return adapter.decompress(compressed)

    t_decomp = timed(do_decomp, warmup, iters, device)
    out = do_decomp()
    if out is None:
        raise RuntimeError(f"{name}: decompress returned None")

    wire = build_wire(compressed, max_chunk_bytes=1 << 30)
    original_bytes = kv.numel() * kv.element_size()
    wire_bytes = wire.nbytes
    # Manager-reported codec payload (may exclude aux); wire is transfer size.
    codec_bytes = int(getattr(compressed, "compressed_size", 0) or 0)
    if codec_bytes <= 0 and compressed.compressed_tensor is not None:
        t = compressed.compressed_tensor
        codec_bytes = t.numel() * t.element_size()

    ratio = original_bytes / wire_bytes
    # Throughput vs original KV volume (GB/s, decimal).
    gb = original_bytes / 1e9
    comp_gbs = gb / t_comp
    decomp_gbs = gb / t_decomp
    # Harmonic speed S for T_codec = V/S (same convention as AnalyticalModel).
    # 1/S = 1/S_comp + 1/S_decomp
    harm_gbs = 1.0 / (1.0 / comp_gbs + 1.0 / decomp_gbs)

    return {
        "method": name,
        "original_bytes": original_bytes,
        "wire_bytes": wire_bytes,
        "codec_bytes": codec_bytes,
        "ratio": ratio,
        "t_compress_ms": t_comp * 1e3,
        "t_decompress_ms": t_decomp * 1e3,
        "compress_GBps": comp_gbs,
        "decompress_GBps": decomp_gbs,
        "harmonic_GBps": harm_gbs,
        "shape": list(kv.shape),
    }


def theoretical_rows(stats: list[dict]) -> list[dict]:
    rows = []
    for s in stats:
        v = s["original_bytes"]
        c = s["wire_bytes"]
        t_codec_ms = s["t_compress_ms"] + s["t_decompress_ms"]
        for bw in BANDWIDTHS_GBPS:
            # Gbps -> bytes/s: bw * 1e9 / 8
            t_xfer_ms = (c * 8.0 / (bw * 1e9)) * 1e3
            t_raw_ms = (v * 8.0 / (bw * 1e9)) * 1e3
            t_total_ms = t_codec_ms + t_xfer_ms
            rows.append({
                "method": s["method"],
                "bandwidth_Gbps": bw,
                "t_compress_ms": s["t_compress_ms"],
                "t_decompress_ms": s["t_decompress_ms"],
                "t_codec_ms": t_codec_ms,
                "t_transfer_ms": t_xfer_ms,
                "t_total_ms": t_total_ms,
                "t_raw_transfer_ms": t_raw_ms,
                "speedup_vs_raw": t_raw_ms / t_total_ms if t_total_ms > 0 else float("nan"),
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-layers", type=int, default=32)
    p.add_argument("--num-blocks", type=int, default=243)  # ~HotpotQA first req
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument(
        "--methods",
        default="original_nvcomp,original_lc,fused_nvcomp,fused_lc",
        help="Comma-separated method names",
    )
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "sim_outputs" / "bench_codec_throughput"),
    )
    args = p.parse_args()

    want = {m.strip() for m in args.methods.split(",") if m.strip()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"KV shape=[{args.num_layers}, 2, {args.num_blocks}, {args.block_size}, "
        f"{args.num_kv_heads}, {args.head_size}] device={args.device} "
        f"warmup={args.warmup} iters={args.iters}"
    )
    kv = make_kv(
        args.num_layers, args.num_blocks, args.block_size,
        args.num_kv_heads, args.head_size, args.device, seed=42,
    )
    orig_mb = kv.numel() * kv.element_size() / 1e6
    print(f"Original KV: {orig_mb:.1f} MB\n")

    stats = []
    for name, rel in METHODS:
        if name not in want:
            continue
        cfg_path = ROOT / rel
        print(f"=== {name} ===")
        s = bench_one(name, cfg_path, kv, args.device, args.warmup, args.iters)
        stats.append(s)
        print(
            f"  ratio={s['ratio']:.2f}x  wire={s['wire_bytes']/1e6:.1f} MB  "
            f"comp={s['t_compress_ms']:.2f} ms ({s['compress_GBps']:.2f} GB/s)  "
            f"decomp={s['t_decompress_ms']:.2f} ms ({s['decompress_GBps']:.2f} GB/s)  "
            f"harmonic={s['harmonic_GBps']:.2f} GB/s"
        )

    theo = theoretical_rows(stats)

    thr_path = out_dir / "throughput.csv"
    with thr_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method", "original_bytes", "wire_bytes", "codec_bytes", "ratio",
                "t_compress_ms", "t_decompress_ms",
                "compress_GBps", "decompress_GBps", "harmonic_GBps", "shape",
            ],
        )
        w.writeheader()
        for s in stats:
            row = dict(s)
            row["shape"] = json.dumps(s["shape"])
            w.writerow(row)

    theo_path = out_dir / "theoretical_transfer.csv"
    with theo_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(theo[0].keys()) if theo else [])
        w.writeheader()
        w.writerows(theo)

    summary = {"throughput": stats, "theoretical": theo, "bandwidths_Gbps": BANDWIDTHS_GBPS}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown tables
    lines = []
    lines.append("# Codec throughput + theoretical transfer\n")
    lines.append(
        f"KV: {args.num_layers}x2x{args.num_blocks}x{args.block_size}x"
        f"{args.num_kv_heads}x{args.head_size} bf16 on {args.device}; "
        f"warmup={args.warmup}, iters={args.iters}\n"
    )
    lines.append("## Compress / decompress throughput\n")
    lines.append(
        "| method | ratio | wire MB | compress ms | compress GB/s | "
        "decompress ms | decompress GB/s | harmonic GB/s |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in stats:
        lines.append(
            f"| {s['method']} | {s['ratio']:.2f}x | {s['wire_bytes']/1e6:.1f} | "
            f"{s['t_compress_ms']:.2f} | {s['compress_GBps']:.2f} | "
            f"{s['t_decompress_ms']:.2f} | {s['decompress_GBps']:.2f} | "
            f"{s['harmonic_GBps']:.2f} |"
        )

    lines.append("\n## Theoretical time: compress + decompress + transfer\n")
    lines.append(
        "Transfer uses `wire_bytes` at stated link rate (Gbps, bit-rate). "
        "`speedup_vs_raw` = raw_transfer / (codec + compressed_transfer).\n"
    )
    header = (
        "| method | 20 Gbps total ms | 50 Gbps | 100 Gbps | 200 Gbps | "
        "codec ms | wire MB |"
    )
    lines.append(header)
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    by_m = {s["method"]: s for s in stats}
    for s in stats:
        cells = []
        for bw in BANDWIDTHS_GBPS:
            r = next(
                x for x in theo
                if x["method"] == s["method"] and x["bandwidth_Gbps"] == bw
            )
            cells.append(f"{r['t_total_ms']:.2f}")
        codec = s["t_compress_ms"] + s["t_decompress_ms"]
        lines.append(
            f"| {s['method']} | " + " | ".join(cells) +
            f" | {codec:.2f} | {s['wire_bytes']/1e6:.1f} |"
        )

    lines.append("\n### Breakdown by bandwidth\n")
    for bw in BANDWIDTHS_GBPS:
        lines.append(f"\n#### {bw} Gbps\n")
        lines.append(
            "| method | compress | decompress | transfer | total | "
            "raw transfer | speedup vs raw |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in theo:
            if r["bandwidth_Gbps"] != bw:
                continue
            lines.append(
                f"| {r['method']} | {r['t_compress_ms']:.2f} | "
                f"{r['t_decompress_ms']:.2f} | {r['t_transfer_ms']:.2f} | "
                f"{r['t_total_ms']:.2f} | {r['t_raw_transfer_ms']:.2f} | "
                f"{r['speedup_vs_raw']:.2f}x |"
            )

    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {thr_path}")
    print(f"Wrote {theo_path}")
    print(f"Wrote {md_path}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
