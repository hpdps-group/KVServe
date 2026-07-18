#!/usr/bin/env python3
"""Smoke test for the in-process GPU LC runtime codec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kvserve_v1.compression.codec.lc_codec import LCCodec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta",
        default=str(PROJECT_ROOT / "build" / "lc_runtime" / "lc_runtime_meta.json"),
    )
    parser.add_argument("--algorithm", default="TUPL8_1 BIT_8 RZE_2")
    parser.add_argument("--nbytes", type=int, default=4 * 1024 * 1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    codec = LCCodec(lc_algorithm=args.algorithm, lc_meta_path=args.meta)
    x = torch.randint(0, 12, (args.nbytes,), dtype=torch.uint8, device="cuda")
    y = codec.encode(x)
    z = codec.decode(y, "uint8", [args.nbytes], "cuda")
    torch.cuda.synchronize()

    ok = torch.equal(x, z)
    ratio = x.numel() / max(1, y.numel())
    print(f"input={x.numel()} bytes compressed={y.numel()} bytes ratio={ratio:.3f}x ok={ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
