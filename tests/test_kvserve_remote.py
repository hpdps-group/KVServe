"""Benchmark KVServe PD separation across two machines.

Start the decode role first, wait for ``[Remote] engine_ready``, then start the
prefill role with identical prompt and transfer arguments.  A warmup batch is
run inside the same vLLM processes before a TCP barrier releases the measured
batch.  This keeps model loading, NCCL initialization, and TileLang JIT
compilation out of the reported benchmark wall time.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

from test_kvserve import (
    DEFAULT_KV_PORT,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MAX_PROMPT_CHARS,
    MODEL_PATH,
    RequestResult,
    build_prompts,
    load_compression_ratios,
    make_compression_spec,
    print_summary,
    save_csv,
)


def _parse_gpu_list(gpus: str) -> list[str]:
    devices = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    if not devices:
        raise ValueError(f"Invalid GPU list: {gpus!r}")
    return devices


def _shutdown_vllm(llm) -> None:
    if llm is None:
        return
    try:
        engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        pass


def _safe_label(value: str) -> str:
    label = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return label.strip("._-") or "remote"


def _compression_stats_path(output_dir: str, run_label: str) -> str:
    return os.path.join(
        output_dir, f"compression_stats_remote_{_safe_label(run_label)}.jsonl")


def _transport_stats_path(output_dir: str, run_label: str, role: str) -> str:
    return os.path.join(
        output_dir,
        f"transport_stats_remote_{_safe_label(run_label)}_{role}.jsonl",
    )


def _transport_summary(path: str, measured_prefix: str) -> dict:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("request_id", "")).startswith(measured_prefix):
                    rows.append(row)
    except OSError:
        return {}
    if not rows:
        return {}

    payload_bytes = sum(int(row.get("payload_bytes", 0)) for row in rows)
    nccl_s = sum(float(row.get("nccl_s", 0.0)) for row in rows)
    queue_waits = sorted(
        float(row.get("queue_wait_s", 0.0)) for row in rows
        if "queue_wait_s" in row
    )
    result = {
        "records": len(rows),
        "payload_bytes": payload_bytes,
        "nccl_service_s": nccl_s,
        "effective_payload_gbps": (
            payload_bytes * 8 / nccl_s / 1e9 if nccl_s > 0 else None
        ),
    }
    if queue_waits:
        result.update({
            "queue_wait_sum_s": sum(queue_waits),
            "queue_wait_max_s": queue_waits[-1],
            "queue_wait_p50_s": queue_waits[len(queue_waits) // 2],
            "queue_wait_p95_s": queue_waits[
                min(len(queue_waits) - 1, int(len(queue_waits) * 0.95))
            ],
        })
    return result


def _validate_model_arg(model: str) -> None:
    """Fail early with a clear error for missing local model mounts."""
    if model.startswith("/") or model.startswith("."):
        config_path = os.path.join(model, "config.json")
        if not os.path.isdir(model) or not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Local model path is not usable: {model!r}. "
                f"Expected directory containing config.json at {config_path!r}. "
                "Pass the real model path inside this machine's container."
            )


def _patch_compression_model(
    compression_spec: object,
    model_path: str,
) -> object:
    """Keep custom/fused configs aligned with the model selected on the CLI."""
    if not isinstance(compression_spec, dict):
        return compression_spec
    spec = json.loads(json.dumps(compression_spec))
    quantizer_cfg = spec.get("quantizer_config")
    if isinstance(quantizer_cfg, dict):
        quantizer_cfg["model_name"] = os.path.basename(model_path.rstrip("/"))
    return spec


def _make_params(
    prompts: list[str],
    max_tokens: int,
    transfer_prefix: str,
):
    from vllm import SamplingParams

    return [
        SamplingParams(
            max_tokens=max_tokens,
            temperature=0,
            extra_args={
                "kv_transfer_params": {
                    "transfer_id": f"{transfer_prefix}-{i}",
                }
            },
        )
        for i in range(len(prompts))
    ]


def _warmup_prompts(prompts: list[str], count: int) -> list[str]:
    if count <= 0:
        return []
    return [prompts[i % len(prompts)] for i in range(count)]


def _connect_measurement_barrier(
    host: str,
    port: int,
    timeout_s: float,
) -> socket.socket:
    """Prefill announces readiness, then waits for decode to release the run."""
    deadline = time.monotonic() + timeout_s
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(min(5.0, timeout_s))
        try:
            sock.connect((host, port))
            sock.sendall(b"READY\n")
            token = sock.recv(16)
            if token.strip() != b"GO":
                raise RuntimeError(f"Unexpected barrier token: {token!r}")
            return sock
        except (ConnectionRefusedError, TimeoutError, OSError) as exc:
            last_error = exc
            sock.close()
            time.sleep(0.2)
    raise TimeoutError(
        f"Timed out connecting to measurement barrier {host}:{port}: {last_error}")


def _accept_measurement_barrier(
    bind_ip: str,
    port: int,
    timeout_s: float,
) -> tuple[socket.socket, socket.socket]:
    """Decode waits until prefill is ready; caller sends GO after starting timer."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((bind_ip, port))
    listener.listen(1)
    listener.settimeout(timeout_s)
    conn, _ = listener.accept()
    conn.settimeout(timeout_s)
    token = conn.recv(16)
    if token.strip() != b"READY":
        conn.close()
        listener.close()
        raise RuntimeError(f"Unexpected barrier token: {token!r}")
    return listener, conn


def _count_tokens(outputs: list[object]) -> tuple[int, int]:
    prompt_tokens = 0
    output_tokens = 0
    for out in outputs:
        prompt_ids = getattr(out, "prompt_token_ids", None)
        if prompt_ids is not None:
            prompt_tokens += len(prompt_ids)
        candidates = getattr(out, "outputs", None) or []
        if candidates:
            output_tokens += len(candidates[0].token_ids)
    return prompt_tokens, output_tokens


def _write_benchmark_summary(
    args,
    role: str,
    engine_init_s: float,
    warmup_s: float,
    measured_s: float,
    prompt_tokens: int,
    output_tokens: int,
    process_total_s: float,
    compression_ratios: list[float] | None = None,
) -> str:
    requests = args.num_requests
    summary = {
        "role": role,
        "run_label": args.run_label,
        "mode": args.mode,
        "compression_config": args.compression_config,
        "model": args.model,
        "requests": requests,
        "warmup_requests": args.warmup_requests,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "async_send": args.async_send,
        "max_inflight_gib": args.max_inflight_gib,
        "max_tokens": args.max_tokens,
        "engine_init_s": engine_init_s,
        "warmup_s": warmup_s,
        "measured_job_time_s": measured_s,
        "request_throughput_req_s": requests / measured_s,
        "prompt_tokens": prompt_tokens,
        "prompt_throughput_tok_s": prompt_tokens / measured_s,
        "output_tokens": output_tokens,
        "output_throughput_tok_s": output_tokens / measured_s,
        "total_tokens": prompt_tokens + output_tokens,
        "total_throughput_tok_s": (prompt_tokens + output_tokens) / measured_s,
        "process_total_s": process_total_s,
        "tilelang_jit_excluded": args.warmup_requests > 0,
        "timestamp_unix_s": time.time(),
    }
    if compression_ratios:
        summary["avg_compression_ratio"] = (
            sum(compression_ratios) / len(compression_ratios))
        summary["compression_ratio_samples"] = len(compression_ratios)
    transport = _transport_summary(
        args.transport_stats_path, f"{args.transfer_prefix}-measure-")
    if transport:
        summary["transport"] = transport

    path = os.path.join(
        args.output_dir,
        f"benchmark_remote_{_safe_label(args.run_label)}_{role}.json",
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        "[Benchmark] "
        f"role={role} run={args.run_label} requests={requests} "
        f"job_time={measured_s:.3f}s "
        f"throughput={requests / measured_s:.3f}req/s "
        f"prompt_tok_s={prompt_tokens / measured_s:.1f} "
        f"output_tok_s={output_tokens / measured_s:.1f}",
        flush=True,
    )
    print(f"[Benchmark] summary={path}", flush=True)
    return path


def _build_llm(args, compression_spec: object, role: str):
    from vllm import LLM
    from vllm.config import KVTransferConfig

    devices = _parse_gpu_list(args.gpus)
    kv_cfg = KVTransferConfig(
        kv_connector="CompressedKVConnector",
        kv_connector_module_path="kvserve_v1.connector.compressed_kv_connector",
        kv_role="kv_producer" if role == "prefill" else "kv_consumer",
        kv_rank=0 if role == "prefill" else 1,
        kv_parallel_size=2,
        kv_ip=args.kv_ip,
        kv_port=args.kv_port,
        kv_connector_extra_config={"compression": compression_spec},
    )
    return LLM(
        model=args.model,
        kv_transfer_config=kv_cfg,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=len(devices),
        enable_prefix_caching=False,
        enforce_eager=True,
        enable_chunked_prefill=False,
        disable_log_stats=False,
    )


def run_prefill(
    args,
    prompts: list[str],
    warmup_prompts: list[str],
    compression_spec: object,
    process_t0: float,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(_parse_gpu_list(args.gpus))
    os.environ["KVSERVE_TRANSPORT_STATS_PATH"] = args.transport_stats_path
    os.environ["KVSERVE_ASYNC_SEND"] = "1" if args.async_send else "0"
    os.environ["KVSERVE_MAX_INFLIGHT_BYTES"] = str(
        int(args.max_inflight_gib * 1024**3))
    if args.compression_stats_path:
        os.environ["KVSERVE_COMPRESSION_STATS_PATH"] = args.compression_stats_path

    init_t0 = time.perf_counter()
    llm = _build_llm(args, compression_spec, "prefill")
    engine_init_s = time.perf_counter() - init_t0
    print(
        f"[Remote] engine_ready role=prefill init={engine_init_s:.3f}s",
        flush=True,
    )
    try:
        warmup_s = 0.0
        if warmup_prompts:
            warmup_t0 = time.perf_counter()
            llm.generate(
                warmup_prompts,
                sampling_params=_make_params(
                    warmup_prompts, 1, f"{args.transfer_prefix}-warmup"),
            )
            warmup_s = time.perf_counter() - warmup_t0
            print(
                f"[Warmup] role=prefill requests={len(warmup_prompts)} "
                f"time={warmup_s:.3f}s (excluded)",
                flush=True,
            )

        barrier = _connect_measurement_barrier(
            args.kv_ip, args.sync_port, args.sync_timeout_s)
        try:
            measured_t0 = time.perf_counter()
            outputs = llm.generate(
                prompts,
                sampling_params=_make_params(
                    prompts, 1, f"{args.transfer_prefix}-measure"),
            )
            measured_s = time.perf_counter() - measured_t0
        finally:
            barrier.close()

        prompt_tokens, output_tokens = _count_tokens(outputs)
        ratios = load_compression_ratios(args.compression_stats_path)
        _write_benchmark_summary(
            args=args,
            role="prefill",
            engine_init_s=engine_init_s,
            warmup_s=warmup_s,
            measured_s=measured_s,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            process_total_s=time.perf_counter() - process_t0,
            compression_ratios=ratios,
        )
    finally:
        _shutdown_vllm(llm)


def run_decode(
    args,
    prompts: list[str],
    warmup_prompts: list[str],
    compression_spec: object,
    process_t0: float,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(_parse_gpu_list(args.gpus))
    os.environ["KVSERVE_TRANSPORT_STATS_PATH"] = args.transport_stats_path

    init_t0 = time.perf_counter()
    llm = _build_llm(args, compression_spec, "decode")
    engine_init_s = time.perf_counter() - init_t0
    print(
        f"[Remote] engine_ready role=decode init={engine_init_s:.3f}s",
        flush=True,
    )
    try:
        warmup_s = 0.0
        if warmup_prompts:
            warmup_t0 = time.perf_counter()
            llm.generate(
                warmup_prompts,
                sampling_params=_make_params(
                    warmup_prompts,
                    args.max_tokens,
                    f"{args.transfer_prefix}-warmup",
                ),
            )
            warmup_s = time.perf_counter() - warmup_t0
            print(
                f"[Warmup] role=decode requests={len(warmup_prompts)} "
                f"time={warmup_s:.3f}s (excluded)",
                flush=True,
            )

        listener, barrier = _accept_measurement_barrier(
            args.kv_ip, args.sync_port, args.sync_timeout_s)
        try:
            measured_t0 = time.perf_counter()
            barrier.sendall(b"GO\n")
            outputs = llm.generate(
                prompts,
                sampling_params=_make_params(
                    prompts,
                    args.max_tokens,
                    f"{args.transfer_prefix}-measure",
                ),
            )
            measured_s = time.perf_counter() - measured_t0
        finally:
            barrier.close()
            listener.close()

        results = []
        for i, out in enumerate(outputs):
            candidate = out.outputs[0]
            results.append(RequestResult(
                request_id=i,
                prompt_chars=len(out.prompt),
                output_text=candidate.text,
                output_tokens=len(candidate.token_ids),
                compression_mode=args.run_label,
            ))
            if args.print_outputs:
                print(
                    f"[Decode] [{i}] {out.prompt[:60]!r}... -> "
                    f"{candidate.text!r}",
                    flush=True,
                )

        prompt_tokens, output_tokens = _count_tokens(outputs)
        print_summary(
            results,
            compression_ratios=None,
            compression_enabled=compression_spec is not None,
        )
        csv_name = (
            f"results_remote_{_safe_label(args.run_label)}_"
            f"{_safe_label(args.lmeval_task or os.path.basename(args.data_path or 'builtin'))}.csv"
        )
        save_csv(results, os.path.join(args.output_dir, csv_name))
        _write_benchmark_summary(
            args=args,
            role="decode",
            engine_init_s=engine_init_s,
            warmup_s=warmup_s,
            measured_s=measured_s,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            process_total_s=time.perf_counter() - process_t0,
        )
    finally:
        _shutdown_vllm(llm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-machine KVServe PD benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--role", choices=["prefill", "decode"], required=True)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--gpus", default="0",
                        help="Comma-separated local GPU ids for this role")
    parser.add_argument(
        "--kv-ip",
        required=True,
        help="Decode machine IP used by both the KV connector and benchmark barrier.",
    )
    parser.add_argument("--kv-port", type=int, default=DEFAULT_KV_PORT)
    parser.add_argument(
        "--sync-port",
        type=int,
        default=None,
        help="TCP measurement-barrier port (default: kv-port + 1000)",
    )
    parser.add_argument("--sync-timeout-s", type=float, default=600.0)
    parser.add_argument("--gpu-mem-util", type=float,
                        default=GPU_MEMORY_UTILIZATION)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="Maximum number of prompt/decode tokens scheduled in one iteration.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=128,
        help="Maximum number of sequences scheduled concurrently.",
    )
    parser.add_argument(
        "--async-send",
        action="store_true",
        help="Allow producer NCCL sends to span scheduler iterations.",
    )
    parser.add_argument(
        "--max-inflight-gib",
        type=float,
        default=4.0,
        help="Async producer send-queue high-water mark in GiB.",
    )
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--transfer-prefix", default="remote")
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--lmeval-task", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--online", action="store_true", default=False)
    parser.add_argument(
        "--mode",
        choices=["none", "default", "custom", "tilelang_lc", "controller"],
        default="none",
    )
    parser.add_argument(
        "--compression-config",
        default=None,
        help="JSON compression config path; overrides the selected mode profile.",
    )
    parser.add_argument("--print-outputs", action="store_true", default=False)
    parser.add_argument("--output-dir", default="./sim_outputs")
    parser.add_argument("--library-path", default=None)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--bandwidth-mbps", type=float, default=1000.0)
    parser.add_argument("--slo-ms", type=float, default=200.0)
    parser.add_argument("--accuracy-req", type=float, default=0.92)
    parser.add_argument("--t-model-ms", type=float, default=0.0)
    args = parser.parse_args()
    process_t0 = time.perf_counter()

    if args.num_requests <= 0:
        parser.error("--num-requests must be positive")
    if args.max_num_batched_tokens <= 0:
        parser.error("--max-num-batched-tokens must be positive")
    if args.max_num_seqs <= 0:
        parser.error("--max-num-seqs must be positive")
    if args.max_inflight_gib <= 0:
        parser.error("--max-inflight-gib must be positive")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests cannot be negative")
    args.sync_port = args.sync_port or args.kv_port + 1000
    args.run_label = args.run_label or (
        os.path.splitext(os.path.basename(args.compression_config))[0]
        if args.compression_config else args.mode
    )

    _validate_model_arg(args.model)
    if args.model != MODEL_PATH:
        import test_kvserve
        test_kvserve.MODEL_PATH = args.model
        test_kvserve.MAX_MODEL_LEN = args.max_model_len
        test_kvserve.MAX_PROMPT_TOKENS = args.max_model_len - 128

    compression_spec = _patch_compression_model(
        make_compression_spec(args), args.model)
    os.makedirs(args.output_dir, exist_ok=True)
    args.compression_stats_path = None
    if compression_spec is not None and args.role == "prefill":
        args.compression_stats_path = _compression_stats_path(
            args.output_dir, args.run_label)
        if os.path.exists(args.compression_stats_path):
            os.remove(args.compression_stats_path)
    args.transport_stats_path = _transport_stats_path(
        args.output_dir, args.run_label, args.role)
    if os.path.exists(args.transport_stats_path):
        os.remove(args.transport_stats_path)

    max_prompt_tokens = max(1, int(args.max_model_len) - 128)
    prompts = build_prompts(
        args.lmeval_task,
        args.num_requests,
        offline=not args.online,
        max_prompt_chars=MAX_PROMPT_CHARS,
        data_path=args.data_path,
        max_prompt_tokens=max_prompt_tokens,
    )
    warmup_prompts = _warmup_prompts(prompts, args.warmup_requests)

    print("\n" + "=" * 64)
    print("KVServe remote PD benchmark")
    print("=" * 64)
    print(f"  Role             : {args.role}")
    print(f"  Model            : {args.model}")
    print(f"  GPUs             : {args.gpus} (TP={len(_parse_gpu_list(args.gpus))})")
    print(f"  KV endpoint      : {args.kv_ip}:{args.kv_port}")
    print(f"  Barrier endpoint : {args.kv_ip}:{args.sync_port}")
    print(f"  Run label        : {args.run_label}")
    print(f"  Compression      : {args.mode}")
    print(f"  Config           : {args.compression_config or '<mode profile>'}")
    print(f"  Requests         : {len(prompts)}")
    print(f"  Warmup requests  : {len(warmup_prompts)} (excluded from timing)")
    print(f"  Batched tokens   : {args.max_num_batched_tokens}")
    print(f"  Max sequences    : {args.max_num_seqs}")
    print(
        f"  Async send       : {args.async_send} "
        f"(limit={args.max_inflight_gib:.2f} GiB)")
    print(f"  Prefix           : {args.transfer_prefix}")
    print("=" * 64 + "\n", flush=True)

    if args.role == "prefill":
        run_prefill(
            args, prompts, warmup_prompts, compression_spec, process_t0)
    else:
        run_decode(
            args, prompts, warmup_prompts, compression_spec, process_t0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
