#!/usr/bin/env python3
"""Launch one KVServe remote benchmark replica per local GPU.

Run this on decode first, then on prefill with the same ports, dataset, and
compression settings. Independent PD replicas share the host's IB link, making
link contention visible without changing the network configuration.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["prefill", "decode"], required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument(
        "--gpu-groups",
        default=None,
        help=(
            "Semicolon-separated TP groups, e.g. '0,1' for one TP=2 replica "
            "or '0,1;2,3' for two TP=2 replicas. Overrides --gpus."
        ),
    )
    parser.add_argument("--kv-ip", required=True)
    parser.add_argument("--base-kv-port", type=int, default=25400)
    parser.add_argument("--base-sync-port", type=int, default=26400)
    parser.add_argument("--ib-device", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--mode", default="none")
    parser.add_argument("--compression-config")
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-mem-util", type=float, default=0.6)
    parser.add_argument("--max-inflight-gib", type=float, default=4.0)
    parser.add_argument("--sync-timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--async-send",
        action="store_true",
        help="Overlap send queues with later prefill steps (off for pressure runs).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    test = root / "tests" / "test_kvserve_remote.py"
    if not test.exists():
        parser.error(f"missing benchmark: {test}")
    if args.gpu_groups:
        gpu_groups = [
            group.strip() for group in args.gpu_groups.split(";")
            if group.strip()
        ]
    else:
        gpu_groups = [
            gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()
        ]
    if not gpu_groups:
        parser.error("--gpus/--gpu-groups must contain at least one GPU")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NCCL_IB_DISABLE": "0",
        "NCCL_NET": "IB",
        "NCCL_IB_HCA": args.ib_device,
        "PYTHONPATH": str(root) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        "PYTHONUNBUFFERED": "1",
    })
    env.pop("NCCL_SOCKET_IFNAME", None)
    env.pop("NCCL_SOCKET_FAMILY", None)
    env.pop("NCCL_IB_GID_INDEX", None)

    processes = []
    for index, gpu_group in enumerate(gpu_groups):
        label = f"{args.run_label}_s{index}"
        replica_out = out / f"{label}_{args.role}"
        replica_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(test),
            "--role", args.role,
            "--gpus", gpu_group,
            "--kv-ip", args.kv_ip,
            "--kv-port", str(args.base_kv_port + index),
            "--sync-port", str(args.base_sync_port + index),
            "--sync-timeout-s", str(args.sync_timeout_s),
            "--ib-device", args.ib_device,
            "--model", args.model,
            "--data-path", args.data_path,
            "--output-dir", str(replica_out),
            "--run-label", label,
            "--transfer-prefix", label,
            "--mode", args.mode,
            "--num-requests", str(args.num_requests),
            "--warmup-requests", str(args.warmup_requests),
            "--max-model-len", str(args.max_model_len),
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            "--max-num-seqs", str(args.max_num_seqs or args.num_requests),
            "--max-tokens", str(args.max_tokens),
            "--gpu-mem-util", str(args.gpu_mem_util),
            "--max-inflight-gib", str(args.max_inflight_gib),
        ]
        if args.compression_config:
            cmd.extend(["--compression-config", args.compression_config])
        if args.role == "prefill" and args.async_send:
            cmd.append("--async-send")
        log_path = out / f"{label}_{args.role}.log"
        log = log_path.open("w", encoding="utf-8")
        print(f"[launch] gpus={gpu_group} log={log_path}", flush=True)
        process = subprocess.Popen(
            cmd,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((label, process, log))

    failed = False
    for label, process, log in processes:
        returncode = process.wait()
        log.close()
        print(f"[done] {label} returncode={returncode}", flush=True)
        failed |= returncode != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
