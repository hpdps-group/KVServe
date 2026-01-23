"""PD Separation Simulator Test Script

USAGE EXAMPLES:
===============

# Run both prefill and decode with TP=1 (default mode)
python test/test_simulator.py

# Run with TP=2
python test/test_simulator.py tp2

# Run both TP=1 and TP=2
python test/test_simulator.py both

# Custom number of requests and rate
python test/test_simulator.py tp1 --num-requests 50 --request-rate 10.0

# Use lm-eval task for prompts
python test/test_simulator.py tp1 --lmeval-task longbench_qasper

# Run only prefill stage (save KV to directory)
python test/test_simulator.py prefill-only --kv-dir ./my_kv_cache --prefill-results-dir ./sim_timestamps

# Run only decode stage (load KV from directory + prefill timestamp file)
python test/test_simulator.py decode-only --kv-dir ./my_kv_cache --prefill-results-file ./sim_timestamps/prefill_output.pkl

CONFIGURATION:
==============
Modify constants at the top of this file to change:
- Compression mode (custom/default/controller)
- Engine parameters (GPU memory, batch size, etc.)
- Network/IO simulation parameters
"""

import argparse
import asyncio
import os
import sys
import csv
import subprocess
import pickle
from typing import Optional

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Set environment for vLLM
os.environ['VLLM_USE_V1'] = '0'
os.environ['VLLM_USE_CUSTOM_ALLREDUCE'] = '0'

from kvserve.simulator import SimulatorBackend
from kvserve.engine.service_config import ServiceConfig
from kvserve.controller import OnlineController
from kvserve.controller.dynamic_online_controller import DynamicOnlineController

# ============================================================================
# ENGINE CONFIGURATION - CHANGE HERE TO ADJUST SIMULATOR SETTINGS
# ============================================================================

# Model path
MODEL_PATH = "/root/ssd/Llama3.1-8B-Instruct"

# Prefill engine settings
PREFILL_GPU_MEMORY_UTILIZATION = 0.75
PREFILL_MAX_MODEL_LEN = 10000
PREFILL_MAX_BATCH_SIZE = 4

# Decode engine settings
DECODE_GPU_MEMORY_UTILIZATION = 0.75
DECODE_MAX_MODEL_LEN = 10000
DECODE_MAX_BATCH_SIZE = 4
DECODE_MAX_OUTPUT_LEN = 128

# Common settings
DTYPE = "bfloat16"
BLOCK_SIZE = 16

# Network simulation parameters (for KV transfer timing)
NETWORK_GBPS = 10.0
MAX_CONCURRENT_TRANSFERS_PREFILL = 2
MAX_CONCURRENT_TRANSFERS_DECODE = 2
NETWORK_EFFICIENCY = 0.8
NETWORK_JITTER_MS = 1.0

# IO simulation parameters (PCIe/storage)
PCIE_GBPS = 5.0
IO_JITTER_MS = 0.5

# Logging
LOG_LEVEL = "INFO"

# KV cache storage directory (for simulation mode)
KV_STORAGE_DIR = "./simulation_kv"

# Output directory for csv artifacts
OUTPUT_DIR = "./sim_outputs"

# Timestamp directories for prefill/decode pkl artifacts
PREFILL_RESULTS_DIR = "./sim_timestamps"
DECODE_RESULTS_DIR = "./sim_timestamps"


# ============================================================================
# COMPRESSION CONFIGURATION - CHANGE HERE TO SWITCH MODES
# ============================================================================

# Select compression mode: "none", "custom", "default", "controller", "cachegen", "kivi"
COMPRESSION_MODE = "controller"  # <-- CHANGE THIS TO SWITCH MODES

# -------- CUSTOM MODE CONFIG --------
CUSTOM_COMPRESSION_CONFIG = {
    "enabled": True,
    "pipeline": ["quantizer", "codec"],
    "impl": "kvserve",
    "quantizer_config": {
        "model_name": "Llama-3.1-8B-Instruct",
        "hybrid_ratio": 0.8,
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
    "min_compress_size": 1024,
}

# -------- CACHEGEN MODE CONFIG --------
CACHEGEN_COMPRESSION_CONFIG = {
    "enabled": True,
    "pipeline": ["quantizer", "codec"],
    "impl": "cachegen",
    "quantizer_config": {
        "model_name": "Llama-3.1-8B-Instruct",
        "quantization_level": 2,
        "high_max_value": 32,
        "mid_max_value": 16,
        "low_max_value": 12,
    },
    "codec_config": {
        "codec_type": "torchac",
    },
    "min_compress_size": 1024,
}

# -------- KIVI MODE CONFIG --------
KIVI_COMPRESSION_CONFIG = {
    "enabled": True,
    "pipeline": ["quantizer", "codec"],
    "impl": "kivi",
    "quantizer_config": {
        "model_name": "Llama-3.1-8B-Instruct",
        "nbits": 2,
        "axis_key": "channel",
        "axis_value": "token",
        "group_size": 32,
    },
    "codec_config": {
        "codec_type": "bitpacking",
    },
    "min_compress_size": 1024,
}

# -------- CONTROLLER MODE CONFIG --------
CONTROLLER_PROFILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "/root/lzd/kvserve_project/profiles/Llama-3.1-8B-Instruct/qasper.json"
)
CONTROLLER_PREFILL_MACHINE = "5090"
CONTROLLER_DECODE_MACHINE = "5090"
CONTROLLER_DATASET = "qasper"


# ============================================================================
# SERVICE CONFIGURATION - CHANGE HERE TO SWITCH SERVICES
# ============================================================================

SERVICE_CONFIG = ServiceConfig(
    bandwidth_gbps=NETWORK_GBPS,  # Network bandwidth in Gbps (bits/s)
    network_efficiency=NETWORK_EFFICIENCY,
    slo_ms=5000.0,  # Service Level Objective in milliseconds
    accuracy_requirement=0.95,  # Minimum accuracy requirement
    model_name="Llama-3.1-8B-Instruct",
    dataset="longbench_qasper",
)

# ============================================================================
# END OF CONFIGURATION
# ============================================================================


def load_lmeval_prompts(task_name: str, num_requests: int) -> list[str]:
    """Load prompts from an lm-eval-harness task (validation/test split)."""
    try:
        from lm_eval import tasks as lm_tasks
        try:
            # Newer lm-eval-harness API: get_task_dict returns registry of task classes
            from lm_eval.tasks import get_task_dict
            try:
                registry = get_task_dict([task_name])  # some versions require list arg
            except TypeError:
                registry = get_task_dict()
            
            # Handle different return types
            if isinstance(registry, dict):
                registry_dict = registry
            elif isinstance(registry, (list, tuple)):
                # registry may be a tuple (dict, list); take dict
                registry_dict = registry[0] if len(registry) > 0 else {}
            else:
                registry_dict = {}
            
            if task_name not in registry_dict:
                raise KeyError(f"Task '{task_name}' not found in registry")
            
            task_obj = registry_dict[task_name]
            # Check if it's already an instance or needs instantiation
            # ConfigurableTask objects are already instances, not classes
            if isinstance(task_obj, type):
                # It's a class, instantiate it
                task = task_obj()
            else:
                # Already an instance (like ConfigurableTask)
                task = task_obj
        except Exception:
            # Fallback: older API
            if hasattr(lm_tasks, "get_task"):
                task = lm_tasks.get_task(task_name)
            else:
                raise
    except Exception as e:
        if "not found" in str(e).lower() or "keyerror" in str(type(e).__name__).lower():
            raise RuntimeError(
                f"lm-eval task '{task_name}' not found.\n"
                f"Error: {type(e).__name__}: {e}\n"
                f"\nTip: Check task name spelling. Some tasks use colons (e.g., 'longbench:qasper').\n"
                f"List available tasks: python -m lm_eval --tasks list"
            ) from e
        else:
            raise RuntimeError(
                f"Failed to load lm-eval task '{task_name}': {type(e).__name__}: {e}\n"
                f"lm-eval-harness is required; install via `pip install lm-eval`."
            ) from e
    # Get dataset - try different API methods for compatibility
    dataset = None
    last_error = None
    
    # Method 1: Try new API - eval_docs or test_docs
    for doc_method_name in ["eval_docs", "test_docs", "validation_docs", "train_docs"]:
        if hasattr(task, doc_method_name):
            try:
                doc_method = getattr(task, doc_method_name)
                if callable(doc_method):
                    dataset = list(doc_method("test"))  # Try with split arg
                else:
                    dataset = list(doc_method)  # Direct attribute
                if dataset:
                    break
            except Exception as e:
                try:
                    # Try without split arg
                    if callable(doc_method):
                        dataset = list(doc_method())
                    if dataset:
                        break
                except Exception:
                    last_error = e
                    continue
    
    # Method 2: Try old API - get_dataset
    if not dataset:
        for split in ["validation", "test", "train"]:
            try:
                if hasattr(task, 'get_dataset'):
                    dataset = list(task.get_dataset(split=split))
                    if dataset:
                        break
            except Exception as e:
                last_error = e
                continue
    
    # Method 3: Try direct dataset attribute
    if not dataset and hasattr(task, 'dataset'):
        try:
            dataset = list(task.dataset)
        except Exception as e:
            last_error = e

    if not dataset:
        error_msg = f"No available dataset for lm-eval task '{task_name}'"
        if last_error:
            error_msg += f"\nLast error: {type(last_error).__name__}: {last_error}"
        error_msg += f"\n\nTip: Some tasks may need additional setup or different naming."
        error_msg += f"\nTry: python -m lm_eval --model hf --tasks {task_name} --limit 1"
        raise RuntimeError(error_msg)

    prompts: list[str] = []
    for doc in dataset:
        if len(prompts) >= num_requests:
            break
        try:
            # Try doc_to_text method
            if hasattr(task, 'doc_to_text'):
                prompts.append(task.doc_to_text(doc))
            elif isinstance(doc, dict) and 'text' in doc:
                prompts.append(doc['text'])
            elif isinstance(doc, dict) and 'input' in doc:
                prompts.append(doc['input'])
            elif isinstance(doc, str):
                prompts.append(doc)
            else:
                # Try to convert to string
                prompts.append(str(doc))
        except Exception as e:
            # Fallback: try raw string if possible
            if isinstance(doc, str):
                prompts.append(doc)
            else:
                continue  # Skip this doc instead of failing
    
    if not prompts:
        raise RuntimeError(f"lm-eval task '{task_name}' produced no prompts (got {len(dataset)} docs but could not convert to text)")
    return prompts


async def worker_prefill(
    tp: int,
    intermediate_file: str,
    num_requests: int,
    request_rate_rps: float,
    lmeval_task: Optional[str] = None,
    compression_config = None,
    service_config = None,
    kv_dir: str = None,
):
    """Worker: run prefill stage"""
    sim = SimulatorBackend(
        model_path=MODEL_PATH,
        tensor_parallel_size=tp,
        gpu_memory_utilization=PREFILL_GPU_MEMORY_UTILIZATION,
        max_model_len=PREFILL_MAX_MODEL_LEN,
        max_batch_size=PREFILL_MAX_BATCH_SIZE,
        dtype=DTYPE,
        block_size=BLOCK_SIZE,
        network_gbps=NETWORK_GBPS,
        max_concurrent_transfers=MAX_CONCURRENT_TRANSFERS_PREFILL,
        network_efficiency=NETWORK_EFFICIENCY,
        network_jitter_ms=NETWORK_JITTER_MS,
        pcie_gbps=PCIE_GBPS,
        io_jitter_ms=IO_JITTER_MS,
        log_level=LOG_LEVEL,
        compression_config=compression_config,
        service_config=service_config,
        simulation_kv_dir=kv_dir or KV_STORAGE_DIR,
    )
    
    await sim.initialize()
    
    # Load prompts
    if lmeval_task:
        prompts = load_lmeval_prompts(lmeval_task, num_requests)
    else:
        # Test prompts: 2 long sentences (~1K chars each) for larger KV cache
        segment_1 = "Climate change represents one of the most pressing challenges facing humanity in the 21st century, requiring comprehensive and coordinated efforts across scientific research, technological innovation, policy development, and global cooperation. The accumulation of greenhouse gases in the Earth's atmosphere, primarily carbon dioxide from fossil fuel combustion and deforestation, has led to unprecedented warming trends that manifest in rising global temperatures, melting polar ice caps, shifting precipitation patterns, and increasing frequency of extreme weather events. Scientists have documented clear evidence of climate change through temperature records, ice core samples, satellite observations, and ecological studies that reveal species migration and ecosystem disruption. Renewable energy technologies such as solar panels, wind turbines, hydroelectric systems, and geothermal installations offer promising pathways toward decarbonization, though they require substantial infrastructure investments and grid modernization to fully replace fossil fuel-based power generation. "
        segment_2 = "The field of artificial intelligence has undergone remarkable transformations over the past several decades, evolving from simple rule-based systems to sophisticated neural networks capable of understanding and generating human-like text. Machine learning algorithms, particularly deep learning models, have demonstrated exceptional capabilities in tasks ranging from image recognition and natural language processing to autonomous decision-making and creative content generation. The development of transformer architectures has revolutionized how we approach sequence-to-sequence problems, enabling models to process and understand context across vast amounts of data with unprecedented accuracy. Large language models trained on extensive corpora have shown emergent abilities such as reasoning, in-context learning, and multi-step problem solving that were previously thought to require explicit programming. "
        base_prompts = [
            (segment_1 * 7)[:7000],
            (segment_2 * 7)[:7000],
        ]
        prompts = [base_prompts[i % len(base_prompts)] for i in range(num_requests)]
    
    results = await sim.run_prefill_only(
        prompts=prompts,
        max_tokens=1000,  # simulate longer sequences
        temperature=0.0,
        request_rate_rps=request_rate_rps,
        output_file=intermediate_file,
    )
    
    print(f"\n[Worker] Prefill complete, saved to {intermediate_file}")
    return results


async def worker_decode(tp: int, intermediate_file: str, final_file: str, compression_config=None, service_config=None, kv_dir: str = None, max_output_len: int = None):
    """Worker: run decode stage"""
    sim = SimulatorBackend(
        model_path=MODEL_PATH,
        tensor_parallel_size=tp,
        gpu_memory_utilization=DECODE_GPU_MEMORY_UTILIZATION,
        max_model_len=DECODE_MAX_MODEL_LEN,
        max_batch_size=DECODE_MAX_BATCH_SIZE,
        dtype=DTYPE,
        block_size=BLOCK_SIZE,
        network_gbps=NETWORK_GBPS,
        max_concurrent_transfers=MAX_CONCURRENT_TRANSFERS_DECODE,
        network_efficiency=NETWORK_EFFICIENCY,
        network_jitter_ms=NETWORK_JITTER_MS,
        pcie_gbps=PCIE_GBPS,
        io_jitter_ms=IO_JITTER_MS,
        log_level=LOG_LEVEL,
        compression_config=compression_config,
        service_config=service_config,
        simulation_kv_dir=kv_dir or KV_STORAGE_DIR,
    )
    
    await sim.initialize()
    
    results = await sim.run_decode_only(
        input_file=intermediate_file,
        output_file=final_file,
        max_output_len=max_output_len,
    )
    
    # Print stats
    sim.print_stats(results)
    
    print(f"\n[Worker] Decode complete, saved to {final_file}")
    return results


def save_results_to_csv(results_file: str, csv_path: str, tp: int):
    """Load results from pickle and save to CSV"""
    with open(results_file, 'rb') as f:
        data = pickle.load(f)
    
    events = data['events'].values()
    
    fieldnames = [
        "tp", "request_id", "total_latency_ms", "compute_only_ms",
        "prefill_compute_ms", "decode_compute_ms",
        "network_queue_ms", "network_transfer_ms",
        "io_pack_ms", "io_unpack_ms", "kv_size_bytes",
        "compression_time_ms", "decompression_time_ms",
        "original_kv_size_bytes", "compression_ratio",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in events:
            writer.writerow({
                "tp": tp,
                "request_id": r.request_id,
                "total_latency_ms": r.total_latency_ms,
                "compute_only_ms": r.compute_only_ms,
                "prefill_compute_ms": r.prefill_compute_ms,
                "decode_compute_ms": r.decode_compute_ms,
                "network_queue_ms": r.network_queue_ms,
                "network_transfer_ms": r.network_transfer_ms,
                "io_pack_ms": r.io_pack_ms,
                "io_unpack_ms": r.io_unpack_ms,
                "kv_size_bytes": r.kv_size_bytes,
                "compression_time_ms": getattr(r, "compression_time_ms", 0.0),
                "decompression_time_ms": getattr(r, "decompression_time_ms", 0.0),
                "original_kv_size_bytes": getattr(r, "original_kv_size_bytes", 0),
                "compression_ratio": getattr(r, "compression_ratio", 1.0),
            })
    print(f"✓ Saved results to {csv_path}")


def run_stage_in_subprocess(
    stage: str,
    tp: int,
    gpus: str,
    compression_mode: str,
    kv_dir: str = None,
    *args
) -> int:
    """Run a stage (prefill or decode) in a separate subprocess"""
    script_path = os.path.abspath(__file__)
    python_exec = sys.executable
    
    env = os.environ.copy()
    # Respect parent CUDA_VISIBLE_DEVICES mapping if present
    def _map_visible_gpus(parent_visible: str, requested: str) -> str:
        parent_list = [p.strip() for p in parent_visible.split(',') if p.strip()]
        req_list = [r.strip() for r in requested.split(',') if r.strip()]
        if parent_list and req_list and all(r.isdigit() for r in req_list):
            idxs = [int(r) for r in req_list]
            if all(i < len(parent_list) for i in idxs):
                return ",".join(parent_list[i] for i in idxs)
        return requested

    parent_visible = env.get('CUDA_VISIBLE_DEVICES')
    mapped_gpus = _map_visible_gpus(parent_visible, gpus) if parent_visible else gpus
    env['CUDA_VISIBLE_DEVICES'] = mapped_gpus
    env['VLLM_USE_V1'] = '0'
    env['VLLM_USE_CUSTOM_ALLREDUCE'] = '0'
    # Force vLLM worker pool to use spawn to avoid CUDA re-init issues in forked procs
    env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    # Torch multiprocessing default to spawn as well
    env.setdefault('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', '1')
    
    cmd = [python_exec, script_path, '--worker', stage, str(tp)] + list(args)
    if compression_mode:
        cmd.extend(['--compression-mode', compression_mode])
    if kv_dir:
        cmd.extend(['--kv-dir', kv_dir])
    
    print(f"\n[Subprocess] Launching {stage.upper()} (TP={tp}) on GPU(s) {mapped_gpus}")
    print(f"[Subprocess] Compression mode: {compression_mode}")
    if kv_dir:
        print(f"[Subprocess] KV directory: {kv_dir}")
    print(f"[Subprocess] Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, env=env)
    return result.returncode


async def main():
    """Main entry point"""
    # Worker subprocess handling - must be before argparse to avoid parsing errors
    if '--worker' in sys.argv:
        # Parse compression mode for worker
        compression_mode = None
        if '--compression-mode' in sys.argv:
            mode_idx = sys.argv.index('--compression-mode')
            compression_mode = sys.argv[mode_idx + 1]
        
        # Parse KV directory
        kv_dir = None
        if '--kv-dir' in sys.argv:
            kv_dir_idx = sys.argv.index('--kv-dir')
            kv_dir = sys.argv[kv_dir_idx + 1]
        
        # Resolve compression config based on mode
        compression_config = None
        service_config = None
        
        if compression_mode == "custom":
            compression_config = CUSTOM_COMPRESSION_CONFIG
        elif compression_mode == "cachegen":
            compression_config = CACHEGEN_COMPRESSION_CONFIG
        elif compression_mode == "kivi":
            compression_config = KIVI_COMPRESSION_CONFIG
        elif compression_mode == "default":
            compression_config = "default"  # Worker will use default config
        elif compression_mode == "controller":
            # Create controller for this worker
            compression_config = DynamicOnlineController(
                model_name=SERVICE_CONFIG.model_name,
                dataset=CONTROLLER_DATASET,
                prefill_machine=CONTROLLER_PREFILL_MACHINE,
                decode_machine=CONTROLLER_DECODE_MACHINE,
                min_compression_ratio=7.0  # Filter out low compression ratio configs
            )
            service_config = SERVICE_CONFIG
        elif compression_mode == "none":
            compression_config = {"enabled": False}
        
        worker_idx = sys.argv.index('--worker')
        stage = sys.argv[worker_idx + 1]  # 'prefill' or 'decode'
        tp = int(sys.argv[worker_idx + 2])
        
        if stage == 'prefill':
            intermediate_file = sys.argv[worker_idx + 3]
            num_requests = int(sys.argv[worker_idx + 4])
            request_rate = float(sys.argv[worker_idx + 5])
            lmeval_task = sys.argv[worker_idx + 6] if len(sys.argv) > worker_idx + 6 and not sys.argv[worker_idx + 6].startswith('--') else None
            print(f"\n[Worker] Running Prefill (TP={tp}), output={intermediate_file}")
            print(f"[Worker] Compression mode: {compression_mode}")
            print(f"[Worker] KV directory: {kv_dir or KV_STORAGE_DIR}")
            await worker_prefill(
                tp,
                intermediate_file,
                num_requests=num_requests,
                request_rate_rps=request_rate,
                lmeval_task=lmeval_task,
                compression_config=compression_config,
                service_config=service_config,
                kv_dir=kv_dir,
            )
        elif stage == 'decode':
            intermediate_file = sys.argv[worker_idx + 3]
            final_file = sys.argv[worker_idx + 4]
            print(f"\n[Worker] Running Decode (TP={tp}), input={intermediate_file}, output={final_file}")
            print(f"[Worker] Compression mode: {compression_mode}")
            print(f"[Worker] KV directory: {kv_dir or KV_STORAGE_DIR}")
            await worker_decode(
                tp,
                intermediate_file,
                final_file,
                compression_config=compression_config,
                service_config=service_config,
                kv_dir=kv_dir,
                max_output_len=DECODE_MAX_OUTPUT_LEN,
            )
        return
    
    # Main process: parse arguments with argparse
    parser = argparse.ArgumentParser(description="PD separation simulator test")
    parser.add_argument("mode", nargs="?", default="both", help="tp1|tp2|both|prefill-only|decode-only")
    parser.add_argument("--num-requests", type=int, default=20, help="Number of requests")
    parser.add_argument("--request-rate", type=float, default=5.0, help="Request rate (RPS)")
    parser.add_argument("--lmeval-task", type=str, default=None, help="lm-eval-harness task name for prompts")
    parser.add_argument("--kv-dir", type=str, default=None, help="KV cache storage directory")
    parser.add_argument("--compression", action="store_true", help="Enable KV compression (use COMPRESSION_MODE)")
    parser.add_argument(
        "--compression-mode",
        type=str,
        default=None,
        choices=["none", "custom", "default", "controller", "cachegen", "kivi"],
        help="Override compression mode for this run",
    )
    parser.add_argument("--prefill-results-dir", type=str, default=None, help="Directory to save prefill timestamp pkl files")
    parser.add_argument("--decode-results-dir", type=str, default=None, help="Directory to save decode timestamp pkl files")
    parser.add_argument("--prefill-results-file", type=str, default=None, help="Prefill timestamp pkl file to use for decode")
    args = parser.parse_args()
    
    # Resolve compression config based on CLI flag + global mode
    compression_config = None
    service_config = None
    compression_mode = "none"

    if args.compression_mode:
        compression_mode = args.compression_mode
    elif args.compression and COMPRESSION_MODE != "none":
        compression_mode = COMPRESSION_MODE
    
    if compression_mode == "custom":
        compression_config = CUSTOM_COMPRESSION_CONFIG
        print(f"\n{'='*60}")
        print("COMPRESSION: Custom Mode")
        print(f"{'='*60}")
        print(f"Pipeline: {compression_config['pipeline']}")
    elif compression_mode == "cachegen":
        compression_config = CACHEGEN_COMPRESSION_CONFIG
        print(f"\n{'='*60}")
        print("COMPRESSION: CacheGen Mode")
        print(f"{'='*60}")
        print(f"Pipeline: {compression_config['pipeline']}")
    elif compression_mode == "kivi":
        compression_config = KIVI_COMPRESSION_CONFIG
        print(f"\n{'='*60}")
        print("COMPRESSION: KIVI Mode")
        print(f"{'='*60}")
        print(f"Pipeline: {compression_config['pipeline']}")
    elif compression_mode == "default":
        compression_config = "default"
        print(f"\n{'='*60}")
        print("COMPRESSION: Default Mode")
        print(f"{'='*60}")
        print("Using system default compression config")
    elif compression_mode == "controller":
        compression_config = DynamicOnlineController(
            model_name=SERVICE_CONFIG.model_name,
            dataset=CONTROLLER_DATASET,
            prefill_machine=CONTROLLER_PREFILL_MACHINE,
            decode_machine=CONTROLLER_DECODE_MACHINE,
            min_compression_ratio=7.0  # Filter out low compression ratio configs
        )
        service_config = SERVICE_CONFIG
        print(f"\n{'='*60}")
        print("COMPRESSION: Controller Mode (Dynamic)")
        print(f"{'='*60}")
        print(f"Prefill machine: {CONTROLLER_PREFILL_MACHINE}")
        print(f"Decode machine: {CONTROLLER_DECODE_MACHINE}")
        print(f"Dataset: {CONTROLLER_DATASET}")
        print(f"Service config:")
        print(f"  Bandwidth: {SERVICE_CONFIG.bandwidth_gbps} GB/s")
        print(f"  SLO: {SERVICE_CONFIG.slo_ms} ms")
        print(f"  Accuracy: {SERVICE_CONFIG.accuracy_requirement}")
    else:
        print(f"\n{'='*60}")
        print("COMPRESSION: Disabled")
        print(f"{'='*60}")
    
    # Main process: coordinate subprocess launches
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    test_mode = args.mode
    kv_dir = args.kv_dir or KV_STORAGE_DIR
    prefill_results_dir = args.prefill_results_dir or PREFILL_RESULTS_DIR
    decode_results_dir = args.decode_results_dir or DECODE_RESULTS_DIR
    
    print("\n" + "="*80)
    print("PD SEPARATION SIMULATOR TEST")
    print("="*80)
    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(prefill_results_dir, exist_ok=True)
    os.makedirs(decode_results_dir, exist_ok=True)

    print(f"\nModel: {MODEL_PATH}")
    print(f"Test mode: {test_mode}")
    print(f"KV directory: {kv_dir}")
    print(f"Prefill results dir: {prefill_results_dir}")
    print(f"Decode results dir: {decode_results_dir}")
    print(f"Requests: {args.num_requests}, Request rate: {args.request_rate} rps")
    if args.lmeval_task:
        print(f"Prompts: lm-eval task '{args.lmeval_task}'")
    else:
        print("Prompts: built-in long prompts")
    print("="*80 + "\n")
    
    # Prefill-only mode
    if test_mode == "prefill-only":
        print("\n" + "#"*80)
        print("# PREFILL ONLY MODE")
        print("#"*80)
        
        intermediate_file = os.path.join(prefill_results_dir, "prefill_output.pkl")
        
        # Default to TP=1 for single-stage runs
        ret = run_stage_in_subprocess(
            'prefill',
            1,
            '0',
            compression_mode,
            kv_dir,
            intermediate_file,
            str(args.num_requests),
            str(args.request_rate),
            *( [args.lmeval_task] if args.lmeval_task else [] ),
        )
        if ret != 0:
            print(f"❌ Prefill failed")
            return
        
        print(f"\n✅ Prefill complete! KV cache saved to: {kv_dir}")
        print(f"   Intermediate file: {intermediate_file}")
        return
    
    # Decode-only mode
    if test_mode == "decode-only":
        print("\n" + "#"*80)
        print("# DECODE ONLY MODE")
        print("#"*80)
        
        intermediate_file = args.prefill_results_file or os.path.join(prefill_results_dir, "prefill_output.pkl")
        final_file = os.path.join(decode_results_dir, "decode_output.pkl")
        csv_file = os.path.join(OUTPUT_DIR, "sim_results_decode_only.csv")
        
        if not os.path.exists(intermediate_file):
            print(f"❌ Intermediate file not found: {intermediate_file}")
            print(f"   Please run prefill-only mode first or specify --prefill-results-file")
            return
        
        # Default to TP=1 for single-stage runs
        ret = run_stage_in_subprocess(
            'decode',
            1,
            '1',
            compression_mode,
            kv_dir,
            intermediate_file,
            final_file
        )
        if ret != 0:
            print(f"❌ Decode failed")
            return
        
        # Save to CSV
        save_results_to_csv(final_file, csv_file, tp=1)
        print(f"\n✅ Decode complete! Results saved to: {csv_file}")
        return
    
    # Test TP=1
    if test_mode in ["both", "tp1", "1"]:
        print("\n" + "#"*80)
        print("# TEST 1: TP=1 (Baseline)")
        print("#"*80)
        
        intermediate_file = os.path.join(prefill_results_dir, "prefill_tp1.pkl")
        final_file = os.path.join(decode_results_dir, "decode_tp1.pkl")
        csv_file = os.path.join(OUTPUT_DIR, "sim_results_tp1.csv")
        
        # Run prefill (GPU 0)
        ret = run_stage_in_subprocess(
            'prefill',
            1,
            '0',
            compression_mode,
            kv_dir,
            intermediate_file,
            str(args.num_requests),
            str(args.request_rate),
            *( [args.lmeval_task] if args.lmeval_task else [] ),
        )
        if ret != 0:
            print(f"❌ TP=1 Prefill failed")
            return
        
        # Run decode (GPU 1)
        ret = run_stage_in_subprocess(
            'decode',
            1,
            '1',
            compression_mode,
            kv_dir,
            intermediate_file,
            final_file
        )
        if ret != 0:
            print(f"❌ TP=1 Decode failed")
            return
        
        # Save to CSV
        save_results_to_csv(final_file, csv_file, tp=1)
    
    # Test TP=2
    if test_mode in ["both", "tp2", "2"]:
        print("\n" + "#"*80)
        print("# TEST 2: TP=2 (Distributed)")
        print("#"*80)
        
        intermediate_file = os.path.join(prefill_results_dir, "prefill_tp2.pkl")
        final_file = os.path.join(decode_results_dir, "decode_tp2.pkl")
        csv_file = os.path.join(OUTPUT_DIR, "sim_results_tp2.csv")
        
        # Run prefill (GPU 0,1)
        ret = run_stage_in_subprocess(
            'prefill',
            2,
            '0,1',
            compression_mode,
            kv_dir,
            intermediate_file,
            str(args.num_requests),
            str(args.request_rate),
            *( [args.lmeval_task] if args.lmeval_task else [] ),
        )
        if ret != 0:
            print(f"❌ TP=2 Prefill failed")
            return
        
        # Run decode (GPU 0,1 - reuse same GPUs, time-multiplexed)
        ret = run_stage_in_subprocess(
            'decode',
            2,
            '0,1',
            compression_mode,
            kv_dir,
            intermediate_file,
            final_file
        )
        if ret != 0:
            print(f"❌ TP=2 Decode failed")
            return
        
        # Save to CSV
        save_results_to_csv(final_file, csv_file, tp=2)
    
    # Compare results
    if test_mode == "both" and os.path.exists(os.path.join(OUTPUT_DIR, "sim_results_tp1.csv")) and os.path.exists(os.path.join(OUTPUT_DIR, "sim_results_tp2.csv")):
        print("\n" + "="*80)
        print("COMPARISON: TP=1 vs TP=2")
        print("="*80)
        
        with open(os.path.join(OUTPUT_DIR, "sim_results_tp1.csv"), "r") as f:
            reader = csv.DictReader(f)
            results_tp1 = list(reader)
        
        with open(os.path.join(OUTPUT_DIR, "sim_results_tp2.csv"), "r") as f:
            reader = csv.DictReader(f)
            results_tp2 = list(reader)
        
        avg_latency_tp1 = sum(float(r['total_latency_ms']) for r in results_tp1) / len(results_tp1)
        avg_latency_tp2 = sum(float(r['total_latency_ms']) for r in results_tp2) / len(results_tp2)
        
        avg_compute_tp1 = sum(float(r['compute_only_ms']) for r in results_tp1) / len(results_tp1)
        avg_compute_tp2 = sum(float(r['compute_only_ms']) for r in results_tp2) / len(results_tp2)
        
        speedup = avg_latency_tp1 / avg_latency_tp2 if avg_latency_tp2 > 0 else 0
        compute_speedup = avg_compute_tp1 / avg_compute_tp2 if avg_compute_tp2 > 0 else 0
        
        print(f"\n📊 Average Total Latency:")
        print(f"  TP=1: {avg_latency_tp1:.2f} ms")
        print(f"  TP=2: {avg_latency_tp2:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        print(f"\n📊 Average Compute Time:")
        print(f"  TP=1: {avg_compute_tp1:.2f} ms")
        print(f"  TP=2: {avg_compute_tp2:.2f} ms")
        print(f"  Speedup: {compute_speedup:.2f}x")
    
    print("\n" + "="*80)
    print("✓ Simulation complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set in parent process; safe to ignore
        pass
    asyncio.run(main())
