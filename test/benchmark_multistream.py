#!/usr/bin/env python3
"""
Multi-Stream Performance Benchmark
Compare single-stream vs multi-stream performance
"""

import asyncio
import os
import sys
import time
import json
import logging
from typing import List, Dict
from dataclasses import dataclass, asdict
import ray
from transformers import AutoTokenizer

# ✅ CRITICAL: Prevent GPU memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Enable logging for important events only
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from kvserve.engine.backend import PDBackend
from kvserve.engine.utils import Request
from kvserve.engine.logger import set_log_level, LogLevel


@dataclass
class BenchmarkResult:
    """Benchmark result for one configuration"""
    mode: str  # "single-stream" or "multi-stream"
    num_requests: int
    total_time: float  # seconds
    throughput: float  # requests/second
    avg_latency: float  # seconds per request
    p50_latency: float
    p90_latency: float
    p99_latency: float
    avg_prefill_time: float
    avg_decode_time: float
    avg_kv_transfer_time: float
    total_input_tokens: int
    total_output_tokens: int
    input_tokens_per_sec: float
    output_tokens_per_sec: float


async def run_benchmark(
    model_path: str,
    tokenizer_path: str,
    num_requests: int = 100,
    enable_multi_stream: bool = False,
    prompt_length: int = 100,  # Medium length
    max_output_tokens: int = 50,
) -> BenchmarkResult:
    """
    Run benchmark with given configuration
    
    Args:
        model_path: Path to model
        tokenizer_path: Path to tokenizer
        num_requests: Number of test requests
        enable_multi_stream: Whether to enable multi-stream optimization
        prompt_length: Target prompt length in tokens
        max_output_tokens: Max output tokens per request
    """
    mode = "multi-stream" if enable_multi_stream else "single-stream"
    print(f"\n{'='*80}")
    print(f"🚀 Running benchmark: {mode.upper()}")
    print(f"{'='*80}")
    print(f"  Requests: {num_requests}")
    print(f"  Prompt length: ~{prompt_length} tokens")
    print(f"  Max output: {max_output_tokens} tokens")
    print()
    
    # Enable kvserve internal logs at DEBUG level (includes multi-stream diagnostics)
    set_log_level(LogLevel.INFO)
    
    # Initialize Ray
    if not ray.is_initialized():
        vllm_path = "/root/lzd/vllm-0.10.1"
        env_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath_parts = [vllm_path, project_root]
        if env_pythonpath:
            pythonpath_parts.append(env_pythonpath)
        os.environ["PYTHONPATH"] = ":".join(pythonpath_parts)
        # Propagate kvserve log level to worker processes
        os.environ["KVSERVE_LOG_LEVEL"] = "DEBUG"
        
        ray.init(
            ignore_reinit_error=True,
            num_gpus=2,
            runtime_env={"env_vars": {
                "PYTHONPATH": os.environ["PYTHONPATH"],
                "KVSERVE_LOG_LEVEL": "WARNING",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # ✅ Prevent fragmentation
            }}
        )
    
    # Create backend
    print("📦 Creating backend...")
    # NOTE: Do NOT set max_num_gpu_blocks - let system auto-profile after loading model
    # The profiling will calculate optimal blocks based on available GPU memory
    
    backend = PDBackend(
        model_path=model_path,
        num_prefill_workers=1,
        num_decoding_workers=1,
        block_size=16,
        max_model_len=4500,  # Support 4000-token prompts + 100 output tokens
        max_batch_size=10,  # Increased from 5 to 10 for better throughput
        # max_num_gpu_blocks: Let profiling auto-calculate ✅
        # max_num_cpu_blocks: Let profiling auto-calculate ✅
        dtype="float16",
        gpu_memory_utilization=0.8,
        kv_transfer_method="nccl",
        nccl_init_method="tcp://localhost:29500",
        enable_multi_stream=enable_multi_stream,
    )
    print(f"   Max model len: 4500 tokens")
    print(f"   GPU blocks: Will be auto-profiled after model load")
    
    # Initialize backend
    print("🔧 Initializing backend (loading models)...")
    await backend.initialize()
    await backend.start()
    print("✅ Backend ready\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Generate test prompts with target length
    print(f"📝 Generating {num_requests} test prompts...")
    base_texts = [
        "Explain the theory of quantum mechanics and its applications in modern physics. ",
        "Describe the process of photosynthesis in plants and its importance to life on Earth. ",
        "Discuss the history of artificial intelligence from its origins to present day. ",
        "Analyze the impact of climate change on global ecosystems and biodiversity. ",
        "Explain the principles of machine learning and deep neural networks in detail. ",
    ]
    
    # Pre-compute token lengths for each base text (only once)
    base_text_token_lengths = []
    base_text_token_ids = []
    for base_text in base_texts:
        token_ids = tokenizer.encode(base_text, add_special_tokens=False)
        base_text_token_lengths.append(len(token_ids))
        base_text_token_ids.append(token_ids)
    
    requests = []
    for i in range(num_requests):
        # Select base text
        base_idx = i % len(base_texts)
        base_token_length = base_text_token_lengths[base_idx]
        base_token_ids = base_text_token_ids[base_idx]
        
        # Calculate how many repetitions needed
        num_repeats = (prompt_length + base_token_length - 1) // base_token_length  # Ceiling division
        
        # Build token_ids directly (much faster than encoding text repeatedly)
        prompt_token_ids = (base_token_ids * num_repeats)[:prompt_length]
        
        # Decode only once to get the prompt text
        prompt = tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
        
        request = Request(
            request_id=f"bench_req_{i:04d}",
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_output_tokens,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
        )
        requests.append(request)
    
    print(f"✅ Generated {len(requests)} requests")
    print(f"   Avg prompt length: {sum(len(r.prompt_token_ids) for r in requests) / len(requests):.1f} tokens\n")
    
    # Submit requests with rate control to avoid memory pressure
    print("🚀 Submitting requests with rate control...")
    start_time = time.time()
    
    # Configuration: submit in batches to avoid memory deadlock
    batch_size = 10  # CRITICAL: For 4000-token prompts, submit 3 at a time (match max_batch_size)
    submit_delay = 1.2  # Wait 1s between batches to let prefill complete
    
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i+batch_size]
        for req in batch:
            await backend.add_request(req)
        
        submitted = min(i + batch_size, len(requests))
        print(f"   Submitted {submitted}/{len(requests)} requests...")
        
        # Small delay to let system process
        if submitted < len(requests):
            await asyncio.sleep(submit_delay)
    
    print(f"✅ All {len(requests)} requests submitted\n")
    
    # Wait for completion
    print("⏳ Waiting for completion...")
    completed = 0
    last_print = time.time()
    
    while completed < num_requests:
        await asyncio.sleep(0.1)
        completed = len(backend._request_outputs)
        
        # Print progress every second
        now = time.time()
        if now - last_print >= 1.0:
            elapsed = now - start_time
            throughput = completed / elapsed if elapsed > 0 else 0
            print(f"   Progress: {completed}/{num_requests} ({completed*100//num_requests}%) "
                  f"| Elapsed: {elapsed:.1f}s | Throughput: {throughput:.2f} req/s")
            last_print = now
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n✅ All requests completed in {total_time:.2f}s\n")
    
    # Collect statistics
    print("📊 Collecting statistics...")
    latencies = []
    prefill_times = []
    decode_times = []
    kv_transfer_times = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    for req_id, outputs in backend._request_outputs.items():
        if not outputs:
            continue
        
        # Get request
        request = None
        for r in requests:
            if r.request_id == req_id:
                request = r
                break
        
        if not request:
            continue
        
        # Calculate latency (from arrival to completion)
        if request.arrival_time and request.decoding_end_time:
            latency = request.decoding_end_time - request.arrival_time
            latencies.append(latency)
        
        # Prefill time
        if request.prefill_start_time and request.prefill_end_time:
            prefill_time = request.prefill_end_time - request.prefill_start_time
            prefill_times.append(prefill_time)
        
        # Decode time
        if request.decoding_start_time and request.decoding_end_time:
            decode_time = request.decoding_end_time - request.decoding_start_time
            decode_times.append(decode_time)
        
        # KV transfer time
        if request.kv_transfer_time:
            kv_transfer_times.append(request.kv_transfer_time)
        
        # Token counts
        total_input_tokens += len(request.prompt_token_ids) if request.prompt_token_ids else 0
        total_output_tokens += len(request.output_token_ids) if request.output_token_ids else 0
    
    # Calculate percentiles
    latencies.sort()
    n = len(latencies)
    p50_latency = latencies[n // 2] if n > 0 else 0
    p90_latency = latencies[int(n * 0.9)] if n > 0 else 0
    p99_latency = latencies[int(n * 0.99)] if n > 0 else 0
    
    # Calculate averages
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_prefill_time = sum(prefill_times) / len(prefill_times) if prefill_times else 0
    avg_decode_time = sum(decode_times) / len(decode_times) if decode_times else 0
    avg_kv_transfer_time = sum(kv_transfer_times) / len(kv_transfer_times) if kv_transfer_times else 0
    
    # Calculate throughput
    throughput = num_requests / total_time if total_time > 0 else 0
    input_tokens_per_sec = total_input_tokens / total_time if total_time > 0 else 0
    output_tokens_per_sec = total_output_tokens / total_time if total_time > 0 else 0
    
    # Get multi-stream stats if enabled
    if enable_multi_stream:
        ms_stats = backend.decode_engine.get_multistream_stats()
        if ms_stats.get("enabled"):
            print(f"\n📈 Multi-stream statistics (TRUE ASYNC):")
            print(f"   Mode: {ms_stats.get('mode', 'UNKNOWN')}")
            print(f"   Transfer queue: {ms_stats.get('transfer_queue_size', 0)}")
            print(f"   Pending transfers: {ms_stats.get('pending_transfers', 0)}")
            print(f"   Ready queue: {ms_stats.get('ready_queue_size', 0)}")
            print(f"   Total transfers: {ms_stats.get('total_transfers', 0)}")
            print(f"   Successful: {ms_stats.get('successful_transfers', 0)}")
            print(f"   Failed: {ms_stats.get('failed_transfers', 0)}")
            
            # ✅ Key metrics for true async
            print(f"\n   ⚡ Async Performance:")
            print(f"   Avg transfer time: {ms_stats.get('avg_transfer_time_ms', 0):.2f} ms")
            print(f"   Avg wait time: {ms_stats.get('avg_wait_time_ms', 0):.2f} ms")
            print(f"   Avg overlap time: {ms_stats.get('avg_overlap_time_ms', 0):.2f} ms")
            print(f"   Overlap ratio: {ms_stats.get('overlap_ratio', 0):.2%} (1.0 = perfect)")
            
            # Explanation
            overlap_ratio = ms_stats.get('overlap_ratio', 0)
            if overlap_ratio > 0.8:
                print(f"   ✅ Excellent overlap! Transfer mostly hidden by computation.")
            elif overlap_ratio > 0.5:
                print(f"   🟡 Good overlap. Some waiting still occurs.")
            elif overlap_ratio > 0.2:
                print(f"   🟠 Moderate overlap. Transfer visible in latency.")
            else:
                print(f"   ❌ Poor overlap. Requests waiting for transfers.")
    
    # Stop backend (but don't delete it yet - let main function handle cleanup)
    print("\n🛑 Stopping backend...")
    await backend.stop()
    
    # Clear PyTorch CUDA cache in worker processes
    # Note: This needs to be done in the worker processes, not here
    # The main cleanup will happen after returning
    
    # Create result
    result = BenchmarkResult(
        mode=mode,
        num_requests=num_requests,
        total_time=total_time,
        throughput=throughput,
        avg_latency=avg_latency,
        p50_latency=p50_latency,
        p90_latency=p90_latency,
        p99_latency=p99_latency,
        avg_prefill_time=avg_prefill_time,
        avg_decode_time=avg_decode_time,
        avg_kv_transfer_time=avg_kv_transfer_time,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        input_tokens_per_sec=input_tokens_per_sec,
        output_tokens_per_sec=output_tokens_per_sec,
    )
    
    return result, backend


def print_comparison(results: List[BenchmarkResult]):
    """Print comparison of results"""
    print("\n" + "="*100)
    print("📊 BENCHMARK RESULTS COMPARISON")
    print("="*100)
    
    if len(results) < 2:
        print("⚠️  Need at least 2 results to compare")
        return
    
    baseline = results[0]
    optimized = results[1]
    
    # Print table header
    print(f"\n{'Metric':<35} {'Single-Stream':<20} {'Multi-Stream':<20} {'Improvement':<15}")
    print("-" * 100)
    
    # Throughput
    throughput_improvement = (optimized.throughput - baseline.throughput) / baseline.throughput * 100
    print(f"{'Throughput (req/s)':<35} {baseline.throughput:<20.2f} {optimized.throughput:<20.2f} {throughput_improvement:>+14.1f}%")
    
    # Total time
    time_improvement = (baseline.total_time - optimized.total_time) / baseline.total_time * 100
    print(f"{'Total Time (s)':<35} {baseline.total_time:<20.2f} {optimized.total_time:<20.2f} {time_improvement:>+14.1f}%")
    
    # Latencies
    lat_improvement = (baseline.avg_latency - optimized.avg_latency) / baseline.avg_latency * 100
    print(f"{'Avg Latency (s)':<35} {baseline.avg_latency:<20.3f} {optimized.avg_latency:<20.3f} {lat_improvement:>+14.1f}%")
    
    p50_improvement = (baseline.p50_latency - optimized.p50_latency) / baseline.p50_latency * 100
    print(f"{'P50 Latency (s)':<35} {baseline.p50_latency:<20.3f} {optimized.p50_latency:<20.3f} {p50_improvement:>+14.1f}%")
    
    p90_improvement = (baseline.p90_latency - optimized.p90_latency) / baseline.p90_latency * 100
    print(f"{'P90 Latency (s)':<35} {baseline.p90_latency:<20.3f} {optimized.p90_latency:<20.3f} {p90_improvement:>+14.1f}%")
    
    p99_improvement = (baseline.p99_latency - optimized.p99_latency) / baseline.p99_latency * 100
    print(f"{'P99 Latency (s)':<35} {baseline.p99_latency:<20.3f} {optimized.p99_latency:<20.3f} {p99_improvement:>+14.1f}%")
    
    print("-" * 100)
    
    # Stage breakdown
    print(f"{'Avg Prefill Time (s)':<35} {baseline.avg_prefill_time:<20.3f} {optimized.avg_prefill_time:<20.3f}")
    print(f"{'Avg Decode Time (s)':<35} {baseline.avg_decode_time:<20.3f} {optimized.avg_decode_time:<20.3f}")
    print(f"{'Avg KV Transfer Time (s)':<35} {baseline.avg_kv_transfer_time:<20.3f} {optimized.avg_kv_transfer_time:<20.3f}")
    
    print("-" * 100)
    
    # Token throughput
    in_tok_improvement = (optimized.input_tokens_per_sec - baseline.input_tokens_per_sec) / baseline.input_tokens_per_sec * 100
    print(f"{'Input Tokens/s':<35} {baseline.input_tokens_per_sec:<20.1f} {optimized.input_tokens_per_sec:<20.1f} {in_tok_improvement:>+14.1f}%")
    
    out_tok_improvement = (optimized.output_tokens_per_sec - baseline.output_tokens_per_sec) / baseline.output_tokens_per_sec * 100
    print(f"{'Output Tokens/s':<35} {baseline.output_tokens_per_sec:<20.1f} {optimized.output_tokens_per_sec:<20.1f} {out_tok_improvement:>+14.1f}%")
    
    print("-" * 100)
    print(f"{'Total Input Tokens':<35} {baseline.total_input_tokens:<20} {optimized.total_input_tokens:<20}")
    print(f"{'Total Output Tokens':<35} {baseline.total_output_tokens:<20} {optimized.total_output_tokens:<20}")
    
    print("\n" + "="*100)
    print(f"🎯 Overall Performance Improvement: {throughput_improvement:+.1f}%")
    print("="*100 + "\n")


async def main():
    """Main benchmark function"""
    # Import configuration
    try:
        from benchmark_configs import RECOMMENDED_CONFIG as config
        print(f"\n📋 Using configuration: {config['name']}")
        print(f"   Description: {config['description']}")
    except ImportError:
        # Fallback to default
        config = {
            "num_requests": 200,
            "prompt_length": 8000,  # Medium prompts for multi-stream benefit
            "max_output_tokens": 100,
            "name": "long_context",
            "description": "Long context for multi-stream optimization",
        }
    
    # Configuration
    model_path = "/root/lzd/model/qwen2.5-VL"
    tokenizer_path = model_path
    num_requests = config["num_requests"]
    prompt_length = config["prompt_length"]
    max_output_tokens = config["max_output_tokens"]
    
    print("\n" + "="*100)
    print("🚀 MULTI-STREAM PERFORMANCE BENCHMARK")
    print("="*100)
    print(f"Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Requests: {num_requests}")
    print(f"  Prompt length: ~{prompt_length} tokens")
    print(f"  Max output: {max_output_tokens} tokens")
    print("="*100)
    
    results = []
    
    # Run single-stream baseline
    try:
        print("\n🔵 Phase 1: Single-Stream (Baseline)")
        result, backend = await run_benchmark(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            num_requests=num_requests,
            enable_multi_stream=False,
            prompt_length=prompt_length,
            max_output_tokens=max_output_tokens,
        )
        results.append(result)
        
        # Clean up: shutdown Ray to release all actors and GPU memory
        print("\n🧹 Cleaning up single-stream backend...")
        
        # First, try to explicitly clean up workers
        try:
            if backend and hasattr(backend, 'prefill_engine') and backend.prefill_engine:
                print("  Cleaning prefill workers...")
                for worker in backend.prefill_engine.workers:
                    try:
                        ray.get(worker.destruct.remote(['model', 'kv']), timeout=5)
                    except:
                        # Force kill if destruct fails
                        try:
                            ray.kill(worker, no_restart=True)
                        except:
                            pass
            if backend and hasattr(backend, 'decode_engine') and backend.decode_engine:
                print("  Cleaning decode workers...")
                for worker in backend.decode_engine.workers:
                    try:
                        ray.get(worker.destruct.remote(['model', 'kv']), timeout=5)
                    except:
                        # Force kill if destruct fails
                        try:
                            ray.kill(worker, no_restart=True)
                        except:
                            pass
        except Exception as e:
            print(f"⚠️  Worker cleanup warning: {e}")
        
        # Delete backend object
        if backend:
            del backend
            import gc
            gc.collect()
        
        # Force garbage collection before shutting down Ray
        import gc
        gc.collect()
        
        print("🛑 Shutting down Ray (this will release all GPU memory)...")
        ray.shutdown()
        
        # Wait a bit for Ray processes to fully exit
        await asyncio.sleep(3)
        
        # Force garbage collection again
        gc.collect()
        
        # Check for zombie processes using GPU (using nvidia-smi to avoid creating CUDA context)
        print("🔍 Checking for GPU processes using nvidia-smi...")
        import subprocess
        try:
            # Use nvidia-smi to check GPU processes
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                print("⚠️  Found processes still using GPU:")
                print(result.stdout)
                # Try to kill them
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            pid = int(line.split(',')[0].strip())
                            if pid != os.getpid():  # Don't kill ourselves
                                print(f"  Attempting to kill process {pid}...")
                                os.kill(pid, 9)  # SIGKILL
                        except:
                            pass
                await asyncio.sleep(2)
            else:
                print("✅ No GPU processes found")
        except Exception as e:
            print(f"  Could not check GPU processes: {e}")
        
        # Check GPU memory using nvidia-smi (avoid PyTorch to prevent CUDA context creation)
        print("\n📊 GPU Memory Status (via nvidia-smi):")
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n')[:2]:  # Only show GPU 0 and 1
                    parts = line.split(',')
                    gpu_id = parts[0].strip()
                    mem_used = float(parts[1].strip())
                    mem_total = float(parts[2].strip())
                    mem_free = mem_total - mem_used
                    print(f"  GPU {gpu_id}: {mem_used:.0f} MiB used, {mem_free:.0f} MiB free / {mem_total:.0f} MiB total")
        except Exception as e:
            print(f"  Could not query GPU memory: {e}")
        
        # Cool down to ensure GPU memory is fully released
        print("\n⏸️  Cooling down for 10 seconds (waiting for GPU memory release)...")
        await asyncio.sleep(10)
        
        # Check again
        print("\n📊 GPU Memory Status after cooldown:")
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n')[:2]:  # Only show GPU 0 and 1
                    parts = line.split(',')
                    gpu_id = parts[0].strip()
                    mem_used = float(parts[1].strip())
                    mem_total = float(parts[2].strip())
                    mem_free = mem_total - mem_used
                    print(f"  GPU {gpu_id}: {mem_used:.0f} MiB used, {mem_free:.0f} MiB free / {mem_total:.0f} MiB total")
        except Exception as e:
            print(f"  Could not query GPU memory: {e}")
        
        # Re-initialize Ray for next test
        print("🔄 Re-initializing Ray for multi-stream test...")
        vllm_path = "/root/lzd/vllm-0.10.1"
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath_parts = [vllm_path, project_root]
        if env_pythonpath:
            pythonpath_parts.append(env_pythonpath)
        os.environ["PYTHONPATH"] = ":".join(pythonpath_parts)
        
        ray.init(
            ignore_reinit_error=True,
            num_gpus=2,
            runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}}
        )
        print("✅ Ray re-initialized")
        
    except Exception as e:
        print(f"\n❌ Single-stream benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run multi-stream optimized
    try:
        print("\n🟢 Phase 2: Multi-Stream (Optimized)")
        result, backend = await run_benchmark(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            num_requests=num_requests,
            enable_multi_stream=True,
            prompt_length=prompt_length,
            max_output_tokens=max_output_tokens,
        )
        results.append(result)
        
        # Clean up multi-stream backend
        if backend:
            del backend
            import gc
            gc.collect()
        
    except Exception as e:
        print(f"\n❌ Multi-stream benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        ray.shutdown()
    
    # Print comparison
    print_comparison(results)
    
    # Save results
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"📁 Results saved to: {output_file}\n")


if __name__ == "__main__":
    asyncio.run(main())

