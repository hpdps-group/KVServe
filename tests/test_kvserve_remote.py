"""Run KVServe PD separation across two machines.

Start the decode side first on the receiver machine, then start the prefill side
on the sender machine. Both sides must use the same prompt arguments so their
transfer_id order matches.
"""

import argparse
import os
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
    extract_decode_metrics,
    get_output_transfer_id,
    load_compression_ratios,
    load_transfer_latencies,
    make_compression_spec,
    merge_request_timings,
    print_summary,
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


def _compression_stats_path(output_dir: str, mode: str) -> str:
    return os.path.join(output_dir, f"compression_stats_remote_{mode}.jsonl")


def _transfer_stats_path(output_dir: str, mode: str) -> str:
    return os.path.join(output_dir, f"transfer_stats_remote_{mode}.jsonl")


def _validate_model_arg(model: str) -> None:
    """Fail early with a clear error for missing local model mounts."""
    if model.startswith("/") or model.startswith("."):
        config_path = os.path.join(model, "config.json")
        if not os.path.isdir(model) or not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Local model path is not usable: {model!r}. "
                f"Expected directory containing config.json at {config_path!r}. "
                "On a remote machine, either mount/copy the model to this exact "
                "path or pass that machine's real model path with --model."
            )


def run_prefill(args, prompts: list[str], compression_spec: object) -> None:
    devices = _parse_gpu_list(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    if args.compression_stats_path:
        os.environ["KVSERVE_COMPRESSION_STATS_PATH"] = args.compression_stats_path

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    kv_cfg = KVTransferConfig(
        kv_connector="CompressedKVConnector",
        kv_connector_module_path="kvserve_v1.connector.compressed_kv_connector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
        kv_ip=args.kv_ip,
        kv_port=args.kv_port,
        kv_connector_extra_config={"compression": compression_spec},
    )
    llm = LLM(
        model=args.model,
        kv_transfer_config=kv_cfg,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        tensor_parallel_size=len(devices),
        enable_prefix_caching=False,
        enforce_eager=True,
        enable_chunked_prefill=False,
        disable_log_stats=False,
    )
    try:
        submit_ts_ns = time.time_ns()
        params = [
            SamplingParams(
                max_tokens=1,
                temperature=0,
                extra_args={
                    "kv_transfer_params": {
                        "transfer_id": f"{args.transfer_prefix}-{i}",
                        "prefill_submit_ts_ns": submit_ts_ns,
                    }
                },
            )
            for i in range(len(prompts))
        ]
        print(
            f"[Prefill] sending {len(prompts)} requests to {args.kv_ip}:{args.kv_port}",
            flush=True,
        )
        llm.generate(prompts, sampling_params=params)
        print("[Prefill] Done - KV sent.", flush=True)
    finally:
        _shutdown_vllm(llm)


def run_decode(args, prompts: list[str], compression_spec: object) -> None:
    devices = _parse_gpu_list(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    if args.transfer_stats_path:
        os.environ["KVSERVE_TRANSFER_STATS_PATH"] = args.transfer_stats_path

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    kv_cfg = KVTransferConfig(
        kv_connector="CompressedKVConnector",
        kv_connector_module_path="kvserve_v1.connector.compressed_kv_connector",
        kv_role="kv_consumer",
        kv_rank=1,
        kv_parallel_size=2,
        kv_ip=args.kv_ip,
        kv_port=args.kv_port,
        kv_connector_extra_config={"compression": compression_spec},
    )
    llm = LLM(
        model=args.model,
        kv_transfer_config=kv_cfg,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        tensor_parallel_size=len(devices),
        enable_prefix_caching=False,
        enforce_eager=True,
        enable_chunked_prefill=False,
        disable_log_stats=False,
    )
    try:
        params = [
            SamplingParams(
                max_tokens=args.max_tokens,
                temperature=0,
                extra_args={
                    "kv_transfer_params": {
                        "transfer_id": f"{args.transfer_prefix}-{i}"
                    }
                },
            )
            for i in range(len(prompts))
        ]
        print(
            f"[Decode] ready on port {args.kv_port}; waiting for prefill KV...",
            flush=True,
        )
        outputs = llm.generate(prompts, sampling_params=params)
        transfer_latencies = load_transfer_latencies(args.transfer_stats_path)

        results = []
        for i, out in enumerate(outputs):
            text = out.outputs[0].text
            phase_prefill_ms, decode_ms, phase_job_ms = extract_decode_metrics(out)
            transfer_id = get_output_transfer_id(
                out, fallback=f"{args.transfer_prefix}-{i}")
            transfer = transfer_latencies.get(transfer_id)
            prefill_ms, transfer_ms, transfer_queue_ms, kv_bytes, job_ms = (
                merge_request_timings(phase_prefill_ms, decode_ms, phase_job_ms, transfer)
            )
            results.append(RequestResult(
                request_id=i,
                prompt_chars=len(out.prompt),
                output_text=text,
                output_tokens=len(out.outputs[0].token_ids),
                compression_mode=args.mode,
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                transfer_ms=transfer_ms,
                transfer_queue_ms=transfer_queue_ms,
                kv_bytes=kv_bytes,
                job_ms=job_ms,
            ))
            if args.print_outputs:
                print(f"[Decode] [{i}] {out.prompt[:60]!r}... -> {text!r}",
                      flush=True)

        jobs = [r.job_ms for r in results if r.job_ms is not None]
        avg_s = (
            f"{sum(jobs) / len(jobs):.1f}" if jobs else "N/A (no vLLM metrics)")
        print(
            f"[Decode] Done. per_request_avg_job_ms={avg_s} "
            f"(queued→last token; {len(jobs)}/{len(results)} with metrics)",
            flush=True,
        )
        ratios = load_compression_ratios(args.compression_stats_path)
        print_summary(
            results,
            ratios,
            compression_enabled=compression_spec is not None,
        )
    finally:
        _shutdown_vllm(llm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-machine KVServe PD runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--role", choices=["prefill", "decode"], required=True)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--gpus", default="0",
                        help="Comma-separated local GPU ids for this role")
    parser.add_argument("--kv-ip", required=True,
                        help="Decode machine IP. Producer connects to it; "
                             "consumer binds locally and keeps this for config parity.")
    parser.add_argument("--kv-port", type=int, default=DEFAULT_KV_PORT)
    parser.add_argument("--gpu-mem-util", type=float,
                        default=GPU_MEMORY_UTILIZATION)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--transfer-prefix", default="remote")
    parser.add_argument("--lmeval-task", default=None)
    parser.add_argument("--online", action="store_true", default=False)
    parser.add_argument("--mode",
                        choices=["none", "default", "custom", "controller"],
                        default="none")
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
    job_t0 = time.perf_counter()
    job_ok = False

    _validate_model_arg(args.model)

    if args.model != MODEL_PATH:
        # test_kvserve.build_prompts uses its module-level model path for token
        # truncation; patch it so remote runs honor --model without duplicating
        # prompt-loading code here.
        import test_kvserve
        test_kvserve.MODEL_PATH = args.model
        test_kvserve.MAX_MODEL_LEN = args.max_model_len
        test_kvserve.MAX_PROMPT_TOKENS = args.max_model_len - 128

    compression_spec = make_compression_spec(args)
    args.compression_stats_path = None
    args.transfer_stats_path = _transfer_stats_path(args.output_dir, args.mode)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.role == "decode" and os.path.exists(args.transfer_stats_path):
        os.remove(args.transfer_stats_path)
    if compression_spec is not None:
        args.compression_stats_path = _compression_stats_path(
            args.output_dir, args.mode)
        if args.role == "prefill" and os.path.exists(args.compression_stats_path):
            os.remove(args.compression_stats_path)

    prompts = build_prompts(
        args.lmeval_task,
        args.num_requests,
        offline=not args.online,
        max_prompt_chars=MAX_PROMPT_CHARS,
    )

    print("\n" + "=" * 60)
    print("KVServe remote PD runner")
    print("=" * 60)
    print(f"  Role        : {args.role}")
    print(f"  Model       : {args.model}")
    print(f"  GPUs        : {args.gpus} (TP={len(_parse_gpu_list(args.gpus))})")
    print(f"  KV endpoint : {args.kv_ip}:{args.kv_port}")
    print(f"  Compression : {args.mode}")
    print(f"  Requests    : {len(prompts)}")
    print(f"  Prefix      : {args.transfer_prefix}")
    print("=" * 60 + "\n")

    try:
        if args.role == "prefill":
            run_prefill(args, prompts, compression_spec)
        else:
            run_decode(args, prompts, compression_spec)
        job_ok = True
    finally:
        total_s = time.perf_counter() - job_t0
        status = "OK" if job_ok else "FAIL"
        print(
            f"[Total] status={status} role={args.role} mode={args.mode} "
            f"requests={len(prompts)} total_job_time={total_s:.3f}s",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
