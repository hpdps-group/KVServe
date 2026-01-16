"""
Simulator backend for PD separation testing with real kvserve engines.

Usage:
    # Prefill stage (separate process)
    sim = SimulatorBackend(model_path, tensor_parallel_size=2, compression_config={...})
    await sim.run_prefill_only(prompts, output_file="prefill_results.pkl")
    
    # Decode stage (separate process, after prefill completes)
    sim = SimulatorBackend(model_path, tensor_parallel_size=2, compression_config={...})
    await sim.run_decode_only(input_file="prefill_results.pkl", output_file="decode_results.pkl")
"""

import asyncio
import os
import pickle
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from kvserve.simulator.network_simulator import NetworkSimulator, TransferTask


def log_info(msg: str):
    print(f"[INFO] {msg}")


def log_debug(msg: str):
    print(f"[DEBUG] {msg}")


def log_warning(msg: str):
    print(f"[WARNING] {msg}")


def log_error(msg: str):
    print(f"[ERROR] {msg}")


@dataclass
class RequestEvent:
    """Event tracking for a single request"""
    request_id: str
    prompt_token_ids: List[int]
    max_tokens: int
    t0_arrival: float
    t_p_end: float = 0.0
    t_net_arrive: float = 0.0
    t_d_submit: float = 0.0
    t_d_end: float = 0.0
    prefill_compute_ms: float = 0.0
    decode_compute_ms: float = 0.0
    compression_time_ms: float = 0.0
    decompression_time_ms: float = 0.0
    extract_time_ms: float = 0.0
    write_time_ms: float = 0.0
    network_queue_ms: float = 0.0
    network_transfer_ms: float = 0.0
    io_pack_ms: float = 0.0
    io_unpack_ms: float = 0.0
    kv_size_bytes: int = 0
    original_kv_size_bytes: int = 0
    compression_ratio: float = 1.0
    num_output_tokens: int = 0
    prefill_done: bool = False
    decode_done: bool = False
    
    @property
    def total_latency_ms(self) -> float:
        return self.t_d_end - self.t0_arrival
    
    @property
    def compute_only_ms(self) -> float:
        return self.prefill_compute_ms + self.decode_compute_ms


class SimulatorBackend:
    """
    Simulator for PD separation using real kvserve engines.
    
    For TP>1, Prefill and Decode must run in separate processes.
    Use run_prefill_only() and run_decode_only() with intermediate files.
    """
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.75,
        max_model_len: int = 16384,
        max_batch_size: int = 32,
        dtype: str = "float16",
        network_gbps: float = 80.0,
        max_concurrent_transfers: int = 2,
        network_efficiency: float = 0.9,
        network_jitter_ms: float = 0.0,
        pcie_gbps: float = 32.0,
        io_jitter_ms: float = 0.0,
        compression_config: Optional[Dict[str, Any]] = None,
        log_level: str = "INFO",
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_batch_size = max_batch_size
        self.dtype = dtype
        self.compression_config = compression_config
        self.log_level = log_level
        
        # Network simulator
        self.network_sim = NetworkSimulator(
            throughput_gbps=network_gbps,
            max_concurrent=max_concurrent_transfers,
            efficiency=network_efficiency,
            jitter_ms=network_jitter_ms
        )
        
        # PCIe parameters for IO simulation
        self.pcie_gbps = pcie_gbps
        self.io_jitter_ms = io_jitter_ms
        
        # Engines (initialized in run_prefill_only / run_decode_only)
        self.prefill_engine = None
        self.decode_engine = None
    
    def _estimate_io_time(self, size_bytes: int) -> float:
        """Estimate IO time for KV transfer (simplified from KVStorage)"""
        size_gb = size_bytes / (1024**3)
        base_time_ms = (size_gb / self.pcie_gbps) * 1000.0
        if self.io_jitter_ms > 0:
            jitter = random.uniform(-self.io_jitter_ms, self.io_jitter_ms)
            base_time_ms += jitter
        return max(0.0, base_time_ms)
    
    async def initialize(self):
        """Initialize"""
        log_info(f"[Simulator] Simulator initialized\n  Model: {self.model_path}\n  TP: {self.tensor_parallel_size}")
    
    async def run_prefill_only(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.0,
        request_rate_rps: float = 1.0,
        output_file: str = "prefill_results.pkl",
    ):
        """
        Run ONLY Prefill stage using kvserve engine, save results to file.
        This should be called in a separate process for TP>1.
        """
        from kvserve.engine.stage_engine import PrefillEngine
        from kvserve.engine.kv_transfer import KVTransferManager, TransferMethod
        from kvserve.engine.utils import Request, EngineStage
        from transformers import AutoTokenizer
        import ray
        
        log_info(f"[Simulator] Phase 1: Running Prefill for {len(prompts)} requests...")
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_gpus=self.tensor_parallel_size)
        
        # Generate arrival times (Poisson process)
        inter_arrival_s = 1.0 / request_rate_rps if request_rate_rps > 0 else 0.0
        arrivals_ms = []
        current_time_ms = 0.0
        for _ in range(len(prompts)):
            arrivals_ms.append(current_time_ms)
            if inter_arrival_s > 0:
                current_time_ms += random.expovariate(1.0 / inter_arrival_s) * 1000.0
        
        # Create events
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        events = {}
        for i, (prompt, arrival_time) in enumerate(zip(prompts, arrivals_ms)):
            request_id = f"req_{i}"
            prompt_token_ids = tokenizer.encode(prompt)
            event = RequestEvent(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                max_tokens=max_tokens,
                t0_arrival=arrival_time,
            )
            events[request_id] = event
        
        # Setup simulation directory
        sim_dir = "./simulation_kv"
        os.makedirs(sim_dir, exist_ok=True)
        
        # Create KV transfer manager
        kv_transfer_manager = KVTransferManager(
            transfer_method=TransferMethod.SIMULATION,
            simulation_dir=sim_dir
        )
        
        # Calculate max_num_gpu_blocks dynamically
        block_size = 16
        max_blocks_per_req = (self.max_model_len + block_size - 1) // block_size
        max_num_gpu_blocks = self.max_batch_size * max_blocks_per_req
        
        log_info(f"[Simulator] Creating PrefillEngine (TP={self.tensor_parallel_size})...")
        log_info(f"[Simulator] Calculated max_num_gpu_blocks={max_num_gpu_blocks} (batch={self.max_batch_size}, max_len={self.max_model_len}, blocks_per_req={max_blocks_per_req})")
        
        # Create prefill engine
        prefill_queue = asyncio.Queue()
        self.prefill_engine = PrefillEngine(
            prefill_decode_bridge_queue=prefill_queue,
            model_path=self.model_path,
            num_workers=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            block_size=block_size,
            max_model_len=self.max_model_len,
            max_num_gpu_blocks=max_num_gpu_blocks,
            dtype=self.dtype,
            kv_transfer_manager=kv_transfer_manager,
            nccl_init_method="tcp://localhost:29500",
            nccl_world_size=self.tensor_parallel_size,
        )
        
        await self.prefill_engine.initialize()
        
        # Add all requests
        for request_id, event in events.items():
            request = Request(
                request_id=request_id,
                prompt=None,
                prompt_token_ids=event.prompt_token_ids,
                max_tokens=event.max_tokens,
                arrival_time=event.t0_arrival,
            )
            self.prefill_engine.scheduler.add_request(request)
        
        # Run prefill
        log_info(f"[Simulator] Running prefill for {len(events)} requests...")
        t_start_global = time.time()
        
        while (self.prefill_engine.scheduler.waiting_queue or 
               self.prefill_engine.scheduler.running_requests):
            t_prefill_start = time.time()
            await self.prefill_engine.step()
            t_prefill_end = time.time()
            
            # Update events with prefill timing
            for req_id, event in events.items():
                if req_id in self.prefill_engine.scheduler.running_requests or not event.prefill_done:
                    if not event.prefill_done:
                        event.prefill_compute_ms = (t_prefill_end - t_prefill_start) * 1000.0
                        event.t_p_end = (t_prefill_end - t_start_global) * 1000.0 + event.t0_arrival
                        event.prefill_done = True
            
            await asyncio.sleep(0.01)
        
        prefill_duration_sec = time.time() - t_start_global
        log_info(f"[Simulator] Prefill complete: {len(events)} requests migrated in {prefill_duration_sec:.2f} seconds")
        
        # Get migration manifest
        manifest = self.prefill_engine.kv_transfer_manager.get_simulation_manifest()
        
        # Aggregate manifest per request (handle TP>1 shards)
        aggregated_manifest = {}
        for shard_req_id, info in manifest.items():
            # Extract base request ID (remove _tp0, _tp1, etc.)
            base_req_id = shard_req_id.rsplit('_tp', 1)[0] if '_tp' in shard_req_id else shard_req_id
            
            if base_req_id not in aggregated_manifest:
                aggregated_manifest[base_req_id] = {
                    'kv_bytes_total': 0,
                    'kv_bytes_max_shard': 0,
                    'original_bytes_total': 0,
                    'compression_time_ms': 0.0,
                    'extract_time_ms': 0.0,
                    'shard_files': []
                }
            
            aggregated_manifest[base_req_id]['kv_bytes_total'] += info.get('kv_bytes', 0)
            aggregated_manifest[base_req_id]['kv_bytes_max_shard'] = max(
                aggregated_manifest[base_req_id]['kv_bytes_max_shard'],
                info.get('kv_bytes', 0)
            )
            aggregated_manifest[base_req_id]['original_bytes_total'] += info.get('original_bytes', 0)
            aggregated_manifest[base_req_id]['compression_time_ms'] = max(
                aggregated_manifest[base_req_id]['compression_time_ms'],
                info.get('compression_time_ms', 0.0)
            )
            aggregated_manifest[base_req_id]['extract_time_ms'] = max(
                aggregated_manifest[base_req_id]['extract_time_ms'],
                info.get('extract_time_ms', 0.0)
            )
            aggregated_manifest[base_req_id]['shard_files'].append(shard_req_id)
        
        # Simulate network transfers
        log_info(f"[Simulator] Simulating network transfers...")
        for req_id, event in events.items():
            if req_id in aggregated_manifest:
                info = aggregated_manifest[req_id]
                
                # Use max shard size for network simulation (realistic for TP>1)
                network_sim_size = info['kv_bytes_max_shard']
                kv_mb = network_sim_size / (1024**2)
                log_info(f"[Simulator] {req_id}: Using max shard size {kv_mb:.2f} MB for network sim (total {info['kv_bytes_total']/(1024**2):.2f} MB)")
                
                # Network transfer simulation
                task = TransferTask(
                    request_id=req_id,
                    size_bytes=network_sim_size,
                    submit_time_ms=event.t_p_end,  # Exclude io_pack_ms
                )
                completed_task = self.network_sim.simulate_transfer(task)
                
                event.network_queue_ms = completed_task.queue_time_ms
                event.network_transfer_ms = completed_task.transfer_time_ms
                event.t_net_arrive = completed_task.complete_time_ms
                
                # KV size and compression
                event.kv_size_bytes = network_sim_size
                event.original_kv_size_bytes = info['original_bytes_total']
                event.compression_time_ms = info['compression_time_ms']
                event.extract_time_ms = info['extract_time_ms']
                
                if event.original_kv_size_bytes > 0:
                    event.compression_ratio = event.kv_size_bytes / event.original_kv_size_bytes
                else:
                    event.compression_ratio = 1.0
        
        # Save results
        with open(output_file, 'wb') as f:
            pickle.dump({'events': events, 'manifest': manifest}, f)
        
        manifest_file = output_file.replace('.pkl', '_manifest.pkl')
        self.prefill_engine.kv_transfer_manager.save_simulation_manifest(manifest_file)
        
        log_info(f"[Simulator] Cleaning up Prefill engine...")
        await self.prefill_engine.shutdown()
        
        log_info(f"[Simulator] ✓ Prefill complete, results saved to {output_file}")
        return list(events.values())
    
    async def run_decode_only(
        self,
        input_file: str = "prefill_results.pkl",
        output_file: str = "decode_results.pkl",
    ):
        """
        Run ONLY Decode stage using kvserve engine, load prefill results from file.
        This should be called in a separate process for TP>1.
        """
        from kvserve.engine.stage_engine import DecodeEngine
        from kvserve.engine.utils import Request, MigratingRequest, EngineStage
        import ray
        
        log_info(f"[Simulator] Phase 2: Running Decode stage...")
        
        # Load prefill results
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        events: Dict[str, RequestEvent] = data['events']
        manifest = data['manifest']
        
        log_info(f"[Simulator] Loaded {len(events)} events from prefill")
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_gpus=self.tensor_parallel_size)
        
        # Create decode engine
        bridge_queue = asyncio.Queue()
        
        # Calculate max_num_gpu_blocks
        block_size = 16
        max_blocks_per_req = (self.max_model_len + block_size - 1) // block_size
        max_num_gpu_blocks = self.max_batch_size * max_blocks_per_req
        
        log_info(f"[Simulator] Creating DecodeEngine (TP={self.tensor_parallel_size})...")
        
        self.decode_engine = DecodeEngine(
            prefill_decode_bridge_queue=bridge_queue,
            model_path=self.model_path,
            num_workers=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            block_size=block_size,
            max_model_len=self.max_model_len,
            max_num_gpu_blocks=max_num_gpu_blocks,
            max_batch_size=self.max_batch_size,
            dtype=self.dtype,
            nccl_init_method="tcp://localhost:29600",
            nccl_world_size=self.tensor_parallel_size,
        )
        
        await self.decode_engine.initialize()
        
        # Set output callback to capture outputs
        decode_outputs = []
        
        def on_output(output):
            decode_outputs.append(output)
        
        self.decode_engine.set_output_callback(on_output)
        
        # Submit requests and run decode loop in parallel
        request_objects = {}  # Store request objects for later access
        
        async def submit_requests():
            """Submit requests to decode engine based on arrival time"""
            log_info(f"[Simulator] Submitting requests to decode based on arrival time...")
            sorted_requests = sorted(events.items(), key=lambda x: x[1].t_net_arrive)
            
            for req_id, event in sorted_requests:
                # Wait until network arrival time
                current_time_ms = (time.time() * 1000.0)
                if event.t_net_arrive > current_time_ms:
                    wait_time_ms = event.t_net_arrive - current_time_ms
                    await asyncio.sleep(wait_time_ms / 1000.0)
                
                # Find shard files for this request
                shard_req_id = f"{req_id}_tp0"
                if shard_req_id not in manifest:
                    # Try without _tp suffix
                    shard_req_id = req_id
                
                if shard_req_id not in manifest:
                    log_error(f"[Simulator] Manifest entry not found for {req_id} (tried {req_id}_tp0, {req_id})")
                    log_info(f"[Simulator] Available manifest keys: {list(manifest.keys())[:10]}...")
                    continue
                
                manifest_entry = manifest[shard_req_id]
                src_blocks = manifest_entry.get('src_blocks', [])
                
                # Create Request and MigratingRequest
                request = Request(
                    request_id=req_id,
                    prompt=None,
                    prompt_token_ids=event.prompt_token_ids,
                    max_tokens=event.max_tokens,
                )
                migrating_req = MigratingRequest(
                    req=request,
                    kv_block_indexes=src_blocks,
                    output_token_ids=[],
                    source_stage=EngineStage.PREFILL,
                    target_stage=EngineStage.DECODING,
                )
                
                await bridge_queue.put(migrating_req)
                
                # Save request object for later retrieval
                request_objects[req_id] = migrating_req
                
                event.t_d_submit = time.time() * 1000.0
        
        async def run_decode_loop():
            """Run decode loop until all requests complete"""
            max_iterations = 100000
            iteration = 0
            submit_done = asyncio.Event()
            
            # Wait for submission to complete
            asyncio.create_task(submit_requests()).add_done_callback(lambda _: submit_done.set())
            
            while iteration < max_iterations:
                iteration += 1
                
                # Check if we should exit
                if (submit_done.is_set() and 
                    bridge_queue.qsize() == 0 and 
                    not self.decode_engine.scheduler.running_requests and
                    not self.decode_engine.scheduler.waiting_queue):
                    break
                
                await self.decode_engine.step()
                await asyncio.sleep(0.01)
            
            if iteration >= max_iterations:
                log_warning(f"[Simulator] Decode loop reached max iterations, exiting.")
        
        # Run submission and decode loop in parallel
        log_info(f"[Simulator] Running decode...")
        t_decode_start = time.time()
        await asyncio.gather(submit_requests(), run_decode_loop())
        t_decode_end = time.time()
        decode_duration_ms = (t_decode_end - t_decode_start) * 1000.0
        
        log_info(f"[Simulator] Decode done: {len(events)} requests in {decode_duration_ms:.2f} ms total")
        
        # Get total decode compute time from engine
        total_decode_compute_ms = self.decode_engine.get_decode_compute_ms()
        
        # Distribute evenly among requests
        per_request_decode_ms = total_decode_compute_ms / len(events) if len(events) > 0 else 0.0
        
        # Get decompression and write times from manifest (for TP>1, take max across shards)
        updated_manifest = self.decode_engine.kv_transfer_manager.get_simulation_manifest()
        
        for req_id, event in events.items():
            # Find all shards for this request
            shard_keys = [k for k in updated_manifest.keys() if k.startswith(req_id)]
            
            decompression_times = []
            write_times = []
            for shard_key in shard_keys:
                info = updated_manifest.get(shard_key, {})
                decompression_times.append(info.get('decompression_time_ms', 0.0))
                write_times.append(info.get('write_time_ms', 0.0))
            
            event.decompression_time_ms = max(decompression_times) if decompression_times else 0.0
            event.write_time_ms = max(write_times) if write_times else 0.0
            event.decode_compute_ms = per_request_decode_ms
            event.t_d_end = event.t_net_arrive + event.decode_compute_ms
            event.decode_done = True
        
        # Save results
        with open(output_file, 'wb') as f:
            pickle.dump({'events': events}, f)
        
        log_info(f"[Simulator] Cleaning up Decode engine...")
        await self.decode_engine.shutdown()
        
        log_info(f"[Simulator] ✓ Decode complete, results saved to {output_file}")
        return list(events.values())
    
    def print_stats(self, results: List[RequestEvent]):
        """Print detailed statistics"""
        print("=" * 80)
        print("SIMULATION RESULTS")
        print("=" * 80)
        
        print(f"\n Configuration:")
        print(f"  Model: {self.model_path}")
        print(f"  TP: {self.tensor_parallel_size}")
        print(f"  Network: {self.network_sim.throughput_gbps} Gbps (efficiency={self.network_sim.efficiency})")
        print(f"  Max Concurrent: {self.network_sim.max_concurrent}")
        print(f"  PCIe: {self.pcie_gbps} GB/s")
        print(f"  Compression: {'ENABLED' if self.compression_config else 'DISABLED'}")
        
        print(f"\n Per-Request Timeline (ms) and KV Size:")
        header = f"{'req_id':<8} {'t0':<8} {'t_p':<8} {'t_net':<8} {'t_d':<8} {'compress':<10} {'decompress':<11} {'net_q':<7} {'net_tx':<10} {'KV_MB':<12}"
        print(header)
        
        for event in results:
            kv_mb = event.kv_size_bytes / (1024**2)
            print(f"{event.request_id:<8} "
                  f"{event.t0_arrival:>8.2f} "
                  f"{event.t_p_end:>8.2f} "
                  f"{event.t_net_arrive:>8.2f} "
                  f"{event.t_d_end:>8.2f} "
                  f"{event.compression_time_ms:>10.2f} "
                  f"{event.decompression_time_ms:>11.2f} "
                  f"{event.network_queue_ms:>7.2f} "
                  f"{event.network_transfer_ms:>10.2f} "
                  f"{kv_mb:>12.2f}")
        
        # Aggregate statistics
        print(f"\n Aggregate Statistics:")
        print(f"  Total Requests: {len(results)}")
        
        avg_latency = sum(e.total_latency_ms for e in results) / len(results)
        avg_compute = sum(e.compute_only_ms for e in results) / len(results)
        avg_prefill = sum(e.prefill_compute_ms for e in results) / len(results)
        avg_decode = sum(e.decode_compute_ms for e in results) / len(results)
        avg_network = sum(e.network_queue_ms + e.network_transfer_ms for e in results) / len(results)
        avg_io = sum(e.io_pack_ms + e.io_unpack_ms for e in results) / len(results)
        avg_compression = sum(e.compression_time_ms for e in results) / len(results)
        avg_decompression = sum(e.decompression_time_ms for e in results) / len(results)
        
        print(f"  Avg Total Latency: {avg_latency:.2f} ms")
        print(f"  Avg Compute Time: {avg_compute:.2f} ms (Prefill: {avg_prefill:.2f}, Decode: {avg_decode:.2f})")
        print(f"  Avg Network Time: {avg_network:.2f} ms")
        print(f"  Avg IO Time: {avg_io:.2f} ms")
        print(f"  Avg Compression Time: {avg_compression:.2f} ms")
        print(f"  Avg Decompression Time: {avg_decompression:.2f} ms")
        
        if any(e.compression_ratio < 1.0 for e in results):
            avg_ratio = sum(e.compression_ratio for e in results) / len(results)
            print(f"  Avg Compression Ratio: {avg_ratio:.2f}")
        
        # Network calculation
        print(f"\n KV Cache Size & Network Calculation:")
        avg_kv_bytes = sum(e.kv_size_bytes for e in results) / len(results)
        avg_kv_mb = avg_kv_bytes / (1024**2)
        print(f"  Avg KV Size: {avg_kv_mb:.2f} MB ({avg_kv_bytes:.0f} bytes)")
        
        size_gb = avg_kv_bytes / (1024**3)
        size_gb_in_bits = size_gb * 8
        expected_transfer_ms = (size_gb_in_bits / self.network_sim.throughput_gbps / self.network_sim.efficiency) * 1000.0
        
        print(f"  Network Config: {self.network_sim.throughput_gbps} Gbps × efficiency = {self.network_sim.throughput_gbps * self.network_sim.efficiency:.1f} Gbps effective")
        print(f"  Expected Transfer Time: {avg_kv_mb:.6f} GB × 8 / {self.network_sim.throughput_gbps * self.network_sim.efficiency:.1f} Gbps × 1000 = {expected_transfer_ms:.4f} ms (transfer only, excludes queue)")
        print(f"  Actual Avg Network Time: {avg_network:.4f} ms (includes queue + transfer time)")
        
        print("=" * 80)

