"""KVServe end-to-end test with optional KV compression and lm-eval prompts.

This runs real vLLM V1 PD separation through CompressedKVConnector and NCCL
transport.

USAGE
=====
  python tests/test_kvserve.py                           # no compression, built-in prompts
  python tests/test_kvserve.py --mode custom             # custom compression config
  python tests/test_kvserve.py --mode tilelang_lc        # TileLang fused quantizer + LC codec
  python tests/test_kvserve.py --mode custom --compression-config configs/compression/fused_top_lc.json
  python tests/test_kvserve.py --mode default            # built-in default config
  python tests/test_kvserve.py --mode controller         # online adaptive (needs --library-path)
      --library-path /path/to/profiles.json
      --bandwidth-mbps 1000 --slo-ms 200 --accuracy-req 0.92
  python tests/test_kvserve.py --lmeval-task wikitext --num-requests 20

CONFIGURATION
=============
Edit the constants block below to change model, GPU memory, ports, etc.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MODEL_PATH = "/data/models/Qwen2.5-7B-Instruct"
GPU_MEMORY_UTILIZATION = 0.6
MAX_MODEL_LEN = 4096
MAX_PROMPT_TOKENS = MAX_MODEL_LEN - 128
DEFAULT_NUM_REQUESTS = 10
DEFAULT_KV_PORT = 25010
OUTPUT_DIR = "./sim_outputs"
MAX_PROMPT_CHARS = 40_000
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_LC_META = os.path.join(
    _REPO_ROOT, "build", "lc_runtime", "lc_runtime_meta.json"
)

CUSTOM_COMPRESSION_CFG = {
    "enabled": True,
    "pipeline": ["quantizer", "codec"],
    "quantizer_config": {
        "model_name": "Qwen2.5-7B-Instruct",
        "hybrid_ratio": 0.5,
        "high_key_max_value": 12,
        "high_value_max_value": 8,
        "low_key_max_value": 6,
        "low_value_max_value": 4,
        "axis_key": "channel",
        "axis_value": "token",
        "split_type": "head",
    },
    "codec_config": {
        "codec_type": "nvcomp",
        "nvcomp_algorithm": "ANS",
        "data_type": "|u1",
    },
    "min_compress_size": 0,
}

TILELANG_LC_COMPRESSION_CFG = {
    "enabled": True,
    "pipeline": ["quantizer", "codec"],
    "quantizer_config": {
        "impl": "tilelang_fused",
        "model_name": "Qwen2.5-7B-Instruct",
        "quant_type": "minmax",
        "base_seed": 3237919422,
        "hybrid_ratio": 0.5,
        "high_key_max_value": 12,
        "high_value_max_value": 8,
        "low_key_max_value": 6,
        "low_value_max_value": 4,
        "axis_key": "channel",
        "axis_value": "token",
        "split_type": "head",
    },
    "codec_config": {
        "codec_type": "lc",
        "lc_algorithm": "TUPL8_1 BIT_8 RZE_2",
        "lc_meta_path": _DEFAULT_LC_META,
    },
    "min_compress_size": 0,
}

# Built-in fallback prompts used when --lmeval-task is not given.
_BUILTIN_PROMPTS = [
    (
        "Climate change represents one of the most pressing challenges facing humanity "
        "in the 21st century, requiring comprehensive and coordinated efforts across "
        "scientific research, technological innovation, policy development, and global "
        "cooperation. The accumulation of greenhouse gases in the Earth's atmosphere, "
        "primarily carbon dioxide from fossil fuel combustion and deforestation, has "
        "led to unprecedented warming trends. Renewable energy technologies such as "
        "solar panels, wind turbines, and geothermal installations offer promising "
        "pathways toward decarbonization. In summary, the key actions needed to "
        "address climate change are"
    ),
    (
        "The field of artificial intelligence has undergone remarkable transformations "
        "over the past several decades, evolving from simple rule-based systems to "
        "sophisticated neural networks capable of understanding and generating human-like "
        "text. Machine learning algorithms, particularly deep learning models, have "
        "demonstrated exceptional capabilities in tasks ranging from image recognition "
        "and natural language processing to autonomous decision-making and creative "
        "content generation. The development of transformer architectures has "
        "revolutionized how we approach sequence-to-sequence problems. In conclusion, "
        "the future of artificial intelligence will likely involve"
    ),
]

# ---------------------------------------------------------------------------
# lm-eval prompt loading  (lm-eval-harness >= 0.4)
# ---------------------------------------------------------------------------

def load_lmeval_prompts(task_name: str, num_requests: int,
                        offline: bool = True,
                        max_prompt_chars: int = MAX_PROMPT_CHARS) -> list:
    """Load prompts from an lm-eval 0.4 task.

    Tries test -> validation -> train splits in order; uses doc_to_text to
    convert each document to a plain string prompt.

    Args:
        offline: If True (default), force HuggingFace datasets to use only
                 local cache and skip Hub version checks.  Set to False if you
                 want to allow downloading the dataset on first run.
    """
    try:
        from lm_eval.tasks import TaskManager, get_task_dict
    except ImportError:
        raise RuntimeError("lm-eval not installed. Run: pip install lm-eval")

    if offline:
        # Force all HuggingFace libraries to use local cache only.
        # HF_HUB_OFFLINE  — huggingface_hub (used by lm-eval task loading)
        # HF_DATASETS_OFFLINE — datasets library
        # TRANSFORMERS_OFFLINE — transformers
        # Must be set before TaskManager() / get_task_dict() are called.
        for _var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ[_var] = "1"

    task_manager = TaskManager()
    task_dict = get_task_dict([task_name], task_manager)

    if task_name not in task_dict:
        sample = sorted(task_manager.all_tasks)[:20]
        raise ValueError(
            f"Task '{task_name}' not found in lm-eval registry.\n"
            f"Sample available tasks: {sample}\n"
            f"Full list: python -m lm_eval --tasks list"
        )

    task = task_dict[task_name]

    docs = None
    for split_getter in ("test_docs", "validation_docs", "train_docs"):
        if hasattr(task, split_getter):
            try:
                docs = list(getattr(task, split_getter)())
                if docs:
                    break
            except Exception:
                continue

    if not docs:
        raise RuntimeError(f"Task '{task_name}' returned no documents from any split.")

    prompts = []
    for doc in docs:
        if len(prompts) >= num_requests:
            break
        text = ""
        # doc_to_text returns empty for perplexity tasks (e.g. wikitext); fall back
        for getter in (
            lambda: task.doc_to_text(doc),
            lambda: task.doc_to_target(doc),
            lambda: doc.get("text", "") if isinstance(doc, dict) else str(doc),
        ):
            try:
                candidate = getter()
                if isinstance(candidate, str) and candidate.strip():
                    text = candidate
                    break
            except Exception:
                continue
        if text:
            if max_prompt_chars > 0 and len(text) > max_prompt_chars:
                text = text[:max_prompt_chars]
            prompts.append(text)

    if not prompts:
        raise RuntimeError(f"Task '{task_name}': could not convert any document to text.")
    return prompts


def load_jsonl_prompts(data_path: str, num_requests: int,
                       max_prompt_chars: int = MAX_PROMPT_CHARS) -> list:
    """Load prompts from a LongBench-style JSON/JSONL file."""
    path = os.path.abspath(data_path)
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        else:
            obj = json.load(f)
            if isinstance(obj, list):
                records = obj
            elif isinstance(obj, dict):
                for key in ("data", "examples", "records"):
                    if isinstance(obj.get(key), list):
                        records = obj[key]
                        break
    if not records:
        raise RuntimeError(f"No records loaded from {path}")

    prompts: list[str] = []
    for record in records:
        if len(prompts) >= num_requests:
            break
        if "prompt" in record and str(record["prompt"]).strip():
            prompt = str(record["prompt"])
        else:
            context = str(record.get("context", record.get("passage", "")))
            question = str(record.get("input", record.get("question", "")))
            # Local LongBench cache often already embeds the instruction template.
            if context.lstrip().startswith("Answer the question"):
                prompt = context
            else:
                prompt = (
                    "Answer the question based on the given context.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n"
                    "Answer:"
                )
        if max_prompt_chars > 0 and len(prompt) > max_prompt_chars:
            prompt = prompt[:max_prompt_chars]
        if prompt.strip():
            prompts.append(prompt)
    if not prompts:
        raise RuntimeError(f"No usable prompts in {path}")
    return prompts


def build_prompts(lmeval_task: Optional[str], num_requests: int,
                  offline: bool = True,
                  max_prompt_chars: int = MAX_PROMPT_CHARS,
                  data_path: Optional[str] = None,
                  max_prompt_tokens: int = MAX_PROMPT_TOKENS) -> list:
    if data_path:
        prompts = load_jsonl_prompts(
            data_path, num_requests, max_prompt_chars=max_prompt_chars)
        print(f"[Prompts] Loaded {len(prompts)} docs from {data_path}")
    elif lmeval_task:
        prompts = load_lmeval_prompts(
            lmeval_task,
            num_requests,
            offline=offline,
            max_prompt_chars=max_prompt_chars,
        )
        print(f"[Prompts] Loaded {len(prompts)} docs from lm-eval '{lmeval_task}'")
    else:
        prompts = [_BUILTIN_PROMPTS[i % len(_BUILTIN_PROMPTS)] for i in range(num_requests)]
        print(f"[Prompts] Using {len(prompts)} built-in prompts")
    if prompts:
        lengths = [len(p) for p in prompts]
        print(
            f"[Prompts] chars min/avg/max = "
            f"{min(lengths)}/{sum(lengths)/len(lengths):.1f}/{max(lengths)}"
        )
    prompts = _truncate_prompts_by_tokens(prompts, max_prompt_tokens)
    return prompts


def _parse_gpu_list(gpus: str) -> list[str]:
    devices = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    if not devices:
        raise ValueError(f"Invalid GPU list: {gpus!r}")
    return devices


def _truncate_prompts_by_tokens(prompts: list[str], max_prompt_tokens: int) -> list[str]:
    """Bound prompt token length to avoid very slow rendering/tokenization."""
    if not prompts or max_prompt_tokens <= 0:
        return prompts

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=False,
        )
    except Exception as e:
        print(f"[Prompts] WARN: tokenizer unavailable, skip token truncation ({e})")
        return prompts

    truncated_prompts: list[str] = []
    token_lengths: list[int] = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) > max_prompt_tokens:
            token_ids = token_ids[:max_prompt_tokens]
            prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
        if prompt and prompt.strip():
            truncated_prompts.append(prompt)
            token_lengths.append(len(token_ids))

    if not truncated_prompts:
        raise RuntimeError("All prompts became empty after token truncation.")

    print(
        f"[Prompts] tokens min/avg/max = "
        f"{min(token_lengths)}/{sum(token_lengths)/len(token_lengths):.1f}/{max(token_lengths)} "
        f"(cap={max_prompt_tokens})"
    )
    return truncated_prompts


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    request_id: int
    prompt_chars: int
    output_text: str
    output_tokens: int
    compression_mode: str


# ---------------------------------------------------------------------------
# Compression spec builder
# ---------------------------------------------------------------------------

def make_compression_spec(args) -> object:
    if getattr(args, "compression_config", None):
        with open(args.compression_config, "r", encoding="utf-8") as f:
            return json.load(f)
    if args.mode == "default":
        return "default"
    if args.mode == "controller":
        if not args.library_path:
            raise ValueError("--library-path is required for controller mode")
        return {
            "mode": "controller",
            "library_path": args.library_path,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "service_config": {
                "bandwidth_mbps": args.bandwidth_mbps,
                "slo_ms": args.slo_ms,
                "accuracy_requirement": args.accuracy_req,
                "t_model_ms": args.t_model_ms,
            },
        }
    if args.mode == "custom":
        return CUSTOM_COMPRESSION_CFG
    if args.mode == "tilelang_lc":
        return TILELANG_LC_COMPRESSION_CFG
    return None  # none


# ---------------------------------------------------------------------------
# Worker processes
# ---------------------------------------------------------------------------

def run_prefill(model, prefill_gpus, kv_port, gpu_mem_util,
                compression_spec, prompts, compression_stats_path,
                max_model_len=MAX_MODEL_LEN,
                kv_buffer_size=1_000_000_000):
    prefill_devices = _parse_gpu_list(str(prefill_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(prefill_devices)
    if compression_stats_path:
        os.environ["KVSERVE_COMPRESSION_STATS_PATH"] = compression_stats_path
    tp_size = len(prefill_devices)

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    kv_cfg = KVTransferConfig(
        kv_connector="CompressedKVConnector",
        kv_connector_module_path="kvserve_v1.connector.compressed_kv_connector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
        kv_ip="127.0.0.1",
        kv_port=kv_port,
        kv_buffer_size=kv_buffer_size,
        kv_connector_extra_config={"compression": compression_spec},
    )
    llm = LLM(
        model=model,
        kv_transfer_config=kv_cfg,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        tensor_parallel_size=tp_size,
        enable_prefix_caching=False,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )
    prefill_params = [
        SamplingParams(
            max_tokens=1,
            temperature=0,
            extra_args={"kv_transfer_params": {"transfer_id": f"sim-{i}"}},
        )
        for i in range(len(prompts))
    ]
    llm.generate(prompts, sampling_params=prefill_params)
    print("[Prefill] Done - KV sent.", flush=True)


def run_decode(model, decode_gpus, kv_port, result_queue, gpu_mem_util,
               compression_spec, prompts, max_tokens, mode_label,
               print_outputs, max_model_len=MAX_MODEL_LEN,
               kv_buffer_size=1_000_000_000):
    decode_devices = _parse_gpu_list(str(decode_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(decode_devices)
    tp_size = len(decode_devices)

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    kv_cfg = KVTransferConfig(
        kv_connector="CompressedKVConnector",
        kv_connector_module_path="kvserve_v1.connector.compressed_kv_connector",
        kv_role="kv_consumer",
        kv_rank=1,
        kv_parallel_size=2,
        kv_ip="127.0.0.1",
        kv_port=kv_port,
        kv_buffer_size=kv_buffer_size,
        kv_connector_extra_config={"compression": compression_spec},
    )
    llm = LLM(
        model=model,
        kv_transfer_config=kv_cfg,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        tensor_parallel_size=tp_size,
        enable_prefix_caching=False,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )

    print("[Decode] Engine ready, starting decode requests...", flush=True)

    t_start = time.monotonic()
    decode_params = [
        SamplingParams(
            max_tokens=max_tokens,
            temperature=0,
            extra_args={"kv_transfer_params": {"transfer_id": f"sim-{i}"}},
        )
        for i in range(len(prompts))
    ]
    outputs = llm.generate(prompts, sampling_params=decode_params)
    total_ms = (time.monotonic() - t_start) * 1e3

    results = []
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        r = RequestResult(
            request_id=i,
            prompt_chars=len(out.prompt),
            output_text=text,
            output_tokens=len(out.outputs[0].token_ids),
            compression_mode=mode_label,
        )
        results.append(r)
        if print_outputs:
            print(f"[Decode] [{i}] {out.prompt[:60]!r}... -> {text!r}", flush=True)

    result_queue.put(results)
    print(f"[Decode] Done. total_wall={total_ms:.0f}ms", flush=True)


# ---------------------------------------------------------------------------
# CSV export + summary
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "request_id", "compression_mode", "prompt_chars",
    "output_tokens", "output_text",
]


def save_csv(results: list, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["output_text"] = row["output_text"][:200]
            writer.writerow({k: row[k] for k in _CSV_FIELDS})
    print(f"[Results] Saved {len(results)} rows -> {path}")


def _compression_stats_path(output_dir: str, mode: str) -> str:
    return os.path.join(output_dir, f"compression_stats_{mode}.jsonl")


def load_compression_ratios(path: str | None) -> list[float]:
    if not path or not os.path.exists(path):
        return []

    ratios: list[float] = []
    with open(path, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            original = float(row.get("original_bytes", 0))
            compressed = float(row.get("compressed_bytes", 0))
            if original > 0 and compressed > 0:
                ratios.append(original / compressed)
    return ratios


def print_summary(
    results: list,
    compression_ratios: list[float] | None = None,
    compression_enabled: bool = False,
) -> None:
    n = len(results)
    if n == 0:
        return
    avg_tok = sum(r.output_tokens for r in results) / n
    success = n
    print(f"\n{'='*60}")
    print(f"SUMMARY  (n={n}, mode={results[0].compression_mode})")
    print(f"{'='*60}")
    print(f"  Avg output tokens/req  : {avg_tok:.1f}")
    print(f"  Successful requests    : {success}/{n}")
    if compression_enabled and compression_ratios:
        avg_ratio = sum(compression_ratios) / len(compression_ratios)
        print(f"  Avg compression ratio  : {avg_ratio:.2f}x")
    elif compression_enabled:
        print("  Avg compression ratio  : N/A")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PD separation test (real NCCL transport)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Prompt source
    parser.add_argument("--lmeval-task", default=None,
                        help="lm-eval task name (e.g. longbench_qasper, gsm8k). "
                             "Omit to use built-in long prompts.")
    parser.add_argument("--data-path", default=None,
                        help="JSON/JSONL prompt file (e.g. LongBench hotpotqa.jsonl). "
                             "Overrides --lmeval-task when set.")
    parser.add_argument("--online", action="store_true", default=False,
                        help="Allow HuggingFace Hub access when loading lm-eval datasets. "
                             "By default datasets are loaded from local cache only.")
    parser.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS)
    parser.add_argument("--max-tokens", type=int, default=30,
                        help="Max new tokens per decode request")
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN,
                        help="vLLM max_model_len / prompt token cap")

    # Hardware
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--prefill-gpu", type=int, default=0)
    parser.add_argument("--decode-gpu", type=int, default=1)
    parser.add_argument("--prefill-gpus", default=None,
                        help="Comma-separated GPU list for prefill. "
                             "Overrides --prefill-gpu and enables TP by list length.")
    parser.add_argument("--decode-gpus", default=None,
                        help="Comma-separated GPU list for decode. "
                             "Overrides --decode-gpu and enables TP by list length.")
    parser.add_argument("--kv-port", type=int, default=DEFAULT_KV_PORT)
    parser.add_argument("--gpu-mem-util", type=float, default=GPU_MEMORY_UTILIZATION)
    parser.add_argument(
        "--kv-buffer-gib",
        type=float,
        default=1.0,
        help="Per-channel decode staging capacity enforced by receiver credits.",
    )

    # Compression mode
    parser.add_argument("--mode",
                        choices=["none", "default", "custom", "tilelang_lc", "controller"],
                        default="none",
                        help="Compression mode")
    parser.add_argument("--compression-config", default=None,
                        help="JSON compression config path (overrides --mode profile)")
    parser.add_argument("--print-outputs", action="store_true", default=False,
                        help="Print per-request decoded text. Disabled by default.")

    # Controller-only options
    parser.add_argument("--library-path", default=None,
                        help="[controller] Profile library JSON path")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="[controller] epsilon-greedy exploration rate")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="[controller] EWMA learning rate")
    parser.add_argument("--bandwidth-mbps", type=float, default=1000.0,
                        help="[controller] Estimated bandwidth in MB/s")
    parser.add_argument("--slo-ms", type=float, default=200.0,
                        help="[controller] SLO budget in ms")
    parser.add_argument("--accuracy-req", type=float, default=0.92,
                        help="[controller] Required accuracy 0-1")
    parser.add_argument("--t-model-ms", type=float, default=0.0,
                        help="[controller] Estimated model compute latency in ms")

    # Output
    parser.add_argument("--output-dir", default=OUTPUT_DIR)

    args = parser.parse_args()
    if args.kv_buffer_gib <= 0:
        parser.error("--kv-buffer-gib must be positive")

    prefill_gpus = args.prefill_gpus or str(args.prefill_gpu)
    decode_gpus = args.decode_gpus or str(args.decode_gpu)
    prefill_tp = len(_parse_gpu_list(prefill_gpus))
    decode_tp = len(_parse_gpu_list(decode_gpus))
    if prefill_tp != decode_tp:
        raise ValueError(
            "CompressedKVConnector currently supports homogeneous TP only: "
            f"prefill_tp={prefill_tp}, decode_tp={decode_tp}")

    compression_spec = make_compression_spec(args)
    max_prompt_tokens = max(1, int(args.max_model_len) - 128)
    prompts = build_prompts(
        args.lmeval_task,
        args.num_requests,
        offline=not args.online,
        max_prompt_chars=MAX_PROMPT_CHARS,
        data_path=args.data_path,
        max_prompt_tokens=max_prompt_tokens,
    )

    print(f"\n{'='*60}")
    print("PD SEPARATION TEST")
    print(f"{'='*60}")
    print(f"  Model        : {args.model}")
    if args.data_path:
        src = "jsonl:" + args.data_path
    elif args.lmeval_task:
        src = "lm-eval:" + args.lmeval_task
    else:
        src = "built-in"
    print(f"  Prompts      : {len(prompts)} ({src})")
    print(f"  Compression  : {args.mode}")
    if args.compression_config:
        print(f"  Config       : {args.compression_config}")
    print(f"  max_model_len: {args.max_model_len}")
    print(f"  Prefill GPUs : {prefill_gpus} (TP={prefill_tp})")
    print(f"  Decode GPUs  : {decode_gpus} (TP={decode_tp})")
    kv_ports = (
        str(args.kv_port) if prefill_tp == 1
        else f"{args.kv_port}-{args.kv_port + prefill_tp - 1}"
    )
    print(f"  KV port(s)   : {kv_ports}")
    print(f"  Receive stage: {args.kv_buffer_gib:.2f} GiB/channel")
    print(f"{'='*60}\n")

    compression_stats_path = None
    if compression_spec is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        mode_label = args.mode
        if args.compression_config:
            mode_label = os.path.splitext(os.path.basename(args.compression_config))[0]
        compression_stats_path = _compression_stats_path(args.output_dir, mode_label)
        if os.path.exists(compression_stats_path):
            os.remove(compression_stats_path)
    else:
        mode_label = args.mode

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_queue = manager.Queue()

    p_prefill = mp.Process(
        target=run_prefill,
        args=(args.model, prefill_gpus, args.kv_port, args.gpu_mem_util,
              compression_spec, prompts, compression_stats_path,
              args.max_model_len, int(args.kv_buffer_gib * 1024**3)),
    )
    p_decode = mp.Process(
        target=run_decode,
        args=(args.model, decode_gpus, args.kv_port, result_queue,
              args.gpu_mem_util, compression_spec, prompts, args.max_tokens,
              mode_label, args.print_outputs, args.max_model_len,
              int(args.kv_buffer_gib * 1024**3)),
    )

    # Start decode first so the consumer transport is ready to receive as
    # prefill begins producing KV. The connector handles per-request waiting.
    p_decode.start()
    p_prefill.start()

    results = None
    # LongBench-scale prompts need a much larger wall budget than short builtins.
    deadline = time.time() + max(600, int(args.num_requests) * 120)
    while time.time() < deadline:
        try:
            results = result_queue.get(timeout=1.0)
            break
        except Exception:
            if not p_decode.is_alive():
                # Drain once more in case the item arrived as the child exited.
                try:
                    results = result_queue.get_nowait()
                except Exception:
                    print("[Main] Decode process exited before returning results.",
                          flush=True)
                break

    for proc in (p_prefill, p_decode):
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)

    try:
        manager.shutdown()
    except Exception:
        pass

    if not results:
        print("FAIL: no results received", flush=True)
        os._exit(1)

    print_summary(
        results,
        load_compression_ratios(compression_stats_path),
        compression_enabled=compression_spec is not None,
    )

    src_tag = "hotpotqa" if args.data_path else (args.lmeval_task or "builtin")
    csv_name = f"results_{mode_label}_{src_tag}.csv"
    save_csv(results, os.path.join(args.output_dir, csv_name))

    n, expected = len(results), len(prompts)
    if n == expected:
        print(f"PASS: {n}/{expected} requests completed")
        os._exit(0)
    else:
        print(f"FAIL: {n}/{expected} completed")
        os._exit(1)


if __name__ == "__main__":
    main()
