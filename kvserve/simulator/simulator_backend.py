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
import time
import pickle
import os
import random
import socket
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .network_simulator import NetworkSimulator, TransferTask

# Simple logging functions
def log_info(msg): print(f"[INFO] {msg}")
def log_debug(msg): pass  # Suppress debug logs in simulator
def log_warning(msg): print(f"[WARNING] {msg}")
def log_error(msg): print(f"[ERROR] {msg}")
def log_debug(msg): print(f"[DEBUG] {msg}")

def pick_free_port(start_port: int, max_tries: int = 50) -> int:
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    log_warning(
        f"No free port found in range {start_port}-{start_port + max_tries - 1}, "
        f"falling back to {start_port}"
    )
    return start_port


@dataclass
class RequestEvent:
    """Event tracking for a single request"""
    request_id: str
    prompt_token_ids: List[int]
    max_tokens: int
    
    # Timestamps (all in milliseconds from start)
    t0_arrival: float = 0.0
    t_p_end: float = 0.0
    t_net_arrive: float = 0.0
    t_d_submit: float = 0.0
    t_d_end: float = 0.0
    
    # Compute times
    prefill_compute_ms: float = 0.0
    decode_compute_ms: float = 0.0
    
    # Compression times (from kvserve internal)
    compression_time_ms: float = 0.0
    decompression_time_ms: float = 0.0
    extract_time_ms: float = 0.0
    write_time_ms: float = 0.0
    
    # Network & IO times
    network_queue_ms: float = 0.0
    network_transfer_ms: float = 0.0
    io_pack_ms: float = 0.0
    io_unpack_ms: float = 0.0
    
    # KV metadata
    kv_size_bytes: int = 0
    original_kv_size_bytes: int = 0
    compression_ratio: float = 1.0
    num_output_tokens: int = 0
    
    # Status
    prefill_done: bool = False
    decode_done: bool = False
    
    @property
    def total_latency_ms(self) -> float:
        return self.t_d_end - self.t0_arrival if self.decode_done else 0.0
    
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
        max_model_len: int = 2048,
        max_batch_size: int = 16,
        block_size: int = 16,
        dtype: str = "float16",
        # Network simulation
        network_gbps: float = 80.0,
        max_concurrent_transfers: int = 2,
        network_efficiency: float = 0.8,
        network_jitter_ms: float = 1.0,
        # IO simulation
        pcie_gbps: float = 24.0,
        io_jitter_ms: float = 0.5,
        # Compression
        compression_config = None,  # Can be Dict, "default", None, or OnlineController instance
        service_config = None,  # Service parameters for controller mode
        # Simulation KV storage
        simulation_kv_dir: str = "./simulation_kv",  # Directory for KV cache files in simulation mode
        # Other
        log_level: str = "INFO",
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_batch_size = max_batch_size
        self.block_size = block_size
        self.dtype = dtype
        self.log_level = log_level
        self.compression_config = compression_config
        self.service_config = service_config
        self.simulation_kv_dir = simulation_kv_dir
        
        # Network simulator (for time calculation only)
        self.network_sim = NetworkSimulator(
            throughput_gbps=network_gbps,
            max_concurrent=max_concurrent_transfers,
            efficiency=network_efficiency,
            jitter_ms=network_jitter_ms,
        )
        
        # IO simulation parameters
        self.pcie_gbps = pcie_gbps
        self.io_jitter_ms = io_jitter_ms
        
        # kvserve engines (created on-demand)
        self.prefill_engine = None
        self.decode_engine = None
    
    def _estimate_io_time(self, size_bytes: int) -> float:
        """Estimate IO time for KV transfer (simplified from KVStorage)"""
        size_gb = size_bytes / (1024 ** 3)
        base_time_ms = (size_gb / self.pcie_gbps) * 1000.0
        jitter = random.uniform(-self.io_jitter_ms, self.io_jitter_ms)
        return max(0.1, base_time_ms + jitter)

    def _compression_enabled(self) -> bool:
        if not self.compression_config:
            return False
        if isinstance(self.compression_config, dict):
            return bool(self.compression_config.get("enabled", True))
        return True
    
    async def initialize(self):
        """Initialize"""
        log_info("[Simulator] Simulator initialized")
        log_info(f"  Model: {self.model_path}")
        log_info(f"  TP: {self.tensor_parallel_size}")
    
    async def update_service_config(self, **kwargs):
        """
        Update service configuration for all engines
        
        Args:
            **kwargs: Service config parameters to update (bandwidth_gbps, slo_ms, etc.)
        """
        if self.service_config:
            self.service_config.update(**kwargs)
        
        # Update prefill engine if exists
        if self.prefill_engine:
            await self.prefill_engine.update_service_config(**kwargs)
        
        # Update decode engine if exists
        if self.decode_engine:
            await self.decode_engine.update_service_config(**kwargs)
        
        log_info(f"[Simulator] Service config updated: {kwargs}")
    
    async def run_prefill_only(
        self,
        prompts: List[str],
        max_tokens: int = 50,
        temperature: float = 0.0,
        request_rate_rps: float = 0.0,
        output_file: str = "prefill_results.pkl",
    ):
        """
        Run ONLY Prefill stage using kvserve engine, save results to file.
        This should be called in a separate process for TP>1.
        """
        import ray
        from kvserve.engine.stage_engine import PrefillEngine
        from kvserve.engine.kv_transfer import KVTransferManager, TransferMethod
        from kvserve.engine.utils import Request, EngineStage
        
        log_info(f"[Simulator] Phase 1: Running Prefill for {len(prompts)} requests...")
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                num_gpus=self.tensor_parallel_size,
            )
        
        # Create arrival times
        arrivals_ms = []
        if request_rate_rps and request_rate_rps > 0:
            import random
            t = 0.0
            for _ in range(len(prompts)):
                inter_arrival_s = random.expovariate(request_rate_rps)
                t += inter_arrival_s * 1000.0
                arrivals_ms.append(t)
        else:
            arrivals_ms = [0.0 for _ in prompts]
        
        # Create events
        events: Dict[str, RequestEvent] = {}
        for i, prompt in enumerate(prompts):
            event = RequestEvent(
                request_id=f"req_{i}",
                prompt_token_ids=[],
                max_tokens=max_tokens,
                t0_arrival=arrivals_ms[i],
            )
            events[event.request_id] = event
        
        # Create simulation directory
        sim_dir = self.simulation_kv_dir
        # Convert to absolute path
        sim_dir = os.path.abspath(sim_dir)
        os.makedirs(sim_dir, exist_ok=True)
        log_info(f"[Simulator] KV cache directory: {sim_dir}")
        
        # Create KV transfer manager in SIMULATION mode
        kv_transfer_manager = KVTransferManager(
            transfer_method=TransferMethod.SIMULATION,
            simulation_dir=sim_dir,
        )
        
        # Create PrefillEngine
        log_info(f"[Simulator] Creating PrefillEngine (TP={self.tensor_parallel_size})...")
        prefill_queue = asyncio.Queue()  # Dummy queue for prefill
        
        # Calculate reasonable max_num_gpu_blocks based on max_batch_size and max_model_len
        # Each request can use up to max_model_len tokens
        # With block_size=16, max blocks per request = ceil(max_model_len / 16)
        block_size = self.block_size
        max_blocks_per_req = (self.max_model_len + block_size - 1) // block_size
        # Total blocks needed = max_batch_size * max_blocks_per_req + 10% buffer
        max_num_gpu_blocks = int(self.max_batch_size * max_blocks_per_req * 1.1)
        log_info(f"[Simulator] Calculated max_num_gpu_blocks={max_num_gpu_blocks} "
                 f"(batch={self.max_batch_size}, max_len={self.max_model_len}, blocks_per_req={max_blocks_per_req})")
        
        self.prefill_engine = PrefillEngine(
            prefill_decode_bridge_queue=prefill_queue,
            model_path=self.model_path,
            num_workers=1,
            tensor_parallel_size=self.tensor_parallel_size,
            block_size=block_size,
            max_num_gpu_blocks=max_num_gpu_blocks,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            kv_transfer_manager=kv_transfer_manager,
            nccl_init_method="tcp://localhost:29500",
            nccl_world_size=self.tensor_parallel_size,  # Only prefill workers
            max_batch_size=self.max_batch_size,
            compression_config=self.compression_config,
            service_config=self.service_config,
        )
        
        await self.prefill_engine.initialize()
        
        # Create tokenizer for encoding prompts
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Add requests
        t_start_global = time.time()
        for i, prompt in enumerate(prompts):
            # Tokenize prompt
            prompt_token_ids = tokenizer.encode(prompt)
            
            request = Request(
                request_id=f"req_{i}",
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                arrival_time=t_start_global + arrivals_ms[i] / 1000.0,
            )
            self.prefill_engine.scheduler.add_request(request)
        
        # Reset network simulator state to avoid cross-experiment contamination
        self.network_sim.reset()
        
        log_info(f"[Simulator] Running prefill for {len(prompts)} requests...")
        t_prefill_start = time.time()
        
        # Run prefill until all requests complete
        while (self.prefill_engine.scheduler.waiting_queue or 
               self.prefill_engine.scheduler.running_requests):
            await self.prefill_engine.step()
            await asyncio.sleep(0.001)
        
        # Wait for all migrations to complete
        await asyncio.sleep(0.1)
        
        # Collect results from prefill_queue (MigratingRequest objects)
        migrating_requests = []
        while not prefill_queue.empty():
            migrating_req = await prefill_queue.get()
            migrating_requests.append(migrating_req)
        
        t_prefill_end = time.time()
        prefill_duration_sec = t_prefill_end - t_prefill_start
        log_info(f"[Simulator] Prefill complete: {len(migrating_requests)} requests migrated in {prefill_duration_sec:.2f} seconds")
        
        # Get transfer manifest (KV files saved by PrefillEngine in SIMULATION mode)
        manifest = kv_transfer_manager.get_simulation_manifest()
        
        # For TP>1: Create aggregated manifest entries (req_id -> sum of all shards)
        if self.tensor_parallel_size > 1:
            aggregated_manifest = {}
            for key, info in manifest.items():
                # Extract base request_id (remove _tpX suffix)
                if '_tp' in key:
                    base_req_id = key.rsplit('_tp', 1)[0]
                    if base_req_id not in aggregated_manifest:
                        aggregated_manifest[base_req_id] = {
                            'kv_bytes_total': 0,  # Total for display
                            'kv_bytes_max_shard': 0,  # Max shard for network simulation
                            'original_bytes_total': 0,  # Total original size
                            'compression_time_ms': 0,
                            'extract_time_ms': 0,
                            'shard_files': []
                        }
                    # Sum up stats from all shards
                    aggregated_manifest[base_req_id]['kv_bytes_total'] += info['kv_bytes']
                    aggregated_manifest[base_req_id]['original_bytes_total'] += info.get('original_bytes', info['kv_bytes'])
                    # Track max shard size (for parallel network transfer simulation)
                    aggregated_manifest[base_req_id]['kv_bytes_max_shard'] = max(
                        aggregated_manifest[base_req_id]['kv_bytes_max_shard'],
                        info['kv_bytes']
                    )
                    aggregated_manifest[base_req_id]['compression_time_ms'] = max(
                        aggregated_manifest[base_req_id]['compression_time_ms'],
                        info.get('compression_time_ms', 0)
                    )
                    aggregated_manifest[base_req_id]['extract_time_ms'] = max(
                        aggregated_manifest[base_req_id]['extract_time_ms'],
                        info.get('extract_time_ms', 0)
                    )
                    aggregated_manifest[base_req_id]['shard_files'].append(key)
            # For backward compatibility, set kv_bytes and original_bytes to total
            for base_req_id in aggregated_manifest:
                aggregated_manifest[base_req_id]['kv_bytes'] = aggregated_manifest[base_req_id]['kv_bytes_total']
                aggregated_manifest[base_req_id]['original_bytes'] = aggregated_manifest[base_req_id]['original_bytes_total']
            # Add aggregated entries to manifest
            manifest.update(aggregated_manifest)
        
        # Update events with prefill results
        for migrating_req in migrating_requests:
            req_id = migrating_req.req.request_id
            if req_id in events:
                event = events[req_id]
                event.prefill_compute_ms = (migrating_req.req.prefill_end_time - migrating_req.req.prefill_start_time) * 1000.0
                event.prefill_done = True
                event.prompt_token_ids = migrating_req.expanded_prompt_token_ids or []
                
                # Get compression info from manifest
                if req_id in manifest:
                    info = manifest[req_id]
                    event.kv_size_bytes = info['kv_bytes']
                    event.original_kv_size_bytes = info.get('original_bytes', info['kv_bytes'])
                    event.compression_time_ms = info.get('compression_time_ms', 0)
                    event.extract_time_ms = info.get('extract_time_ms', 0)
                    
                    # Calculate compression ratio
                    if event.original_kv_size_bytes > 0:
                        event.compression_ratio = event.original_kv_size_bytes / event.kv_size_bytes
                    else:
                        event.compression_ratio = 1.0
                    
                    # Estimate IO pack time
                    event.io_pack_ms = self._estimate_io_time(event.kv_size_bytes)
                
                # CRITICAL FIX: t_p_end must include compression time
                # Timeline: t0 → [prefill] → [compress] → [network] → ...
                event.t_p_end = event.t0_arrival + event.prefill_compute_ms + event.compression_time_ms
        
        # Simulate network transfers and calculate arrival times
        log_info(f"[Simulator] Simulating network transfers...")
        # IMPORTANT: Process events in order of prefill completion time (t_p_end)
        # to correctly simulate network queueing behavior
        sorted_events = sorted(events.items(), key=lambda x: x[1].t_p_end)
        log_info(f"[Simulator] Network simulation order (by t_p_end): {[req_id for req_id, _ in sorted_events[:5]]} ...")
        for req_id, event in sorted_events:
            if event.kv_size_bytes > 0:
                # Use total KV size for network simulation.
                # TP>1 still traverses the same network bandwidth, so total size matters.
                network_sim_size = event.kv_size_bytes
                
                # Network transfer starts immediately after prefill completes (no IO pack in real scenario)
                # IO pack time is simulator-internal and should not affect end-to-end latency
                task = TransferTask(
                    request_id=req_id,
                    size_bytes=network_sim_size,
                    submit_time_ms=event.t_p_end,  # Exclude io_pack_ms from submit time
                )
                completed_task = self.network_sim.simulate_transfer(task)
                event.network_queue_ms = completed_task.queue_time_ms
                event.network_transfer_ms = completed_task.transfer_time_ms
                event.t_net_arrive = completed_task.complete_time_ms
                
                # Update manifest with arrival time (for decode stage)
                if req_id in manifest:
                    manifest[req_id]['arrival_time_ms'] = event.t_net_arrive
        
        # Save manifest for decode stage
        manifest_file = output_file.replace('.pkl', '_manifest.pkl')
        kv_transfer_manager.save_simulation_manifest(manifest_file)
        
        # Save events
        with open(output_file, 'wb') as f:
            pickle.dump({
                'events': events,
                'prompts': prompts,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'manifest_file': manifest_file,
            }, f)
        
        # Shutdown
        log_info(f"[Simulator] Cleaning up Prefill engine...")
        # No explicit shutdown method, just let Ray clean up
        self.prefill_engine = None
        ray.shutdown()
        
        log_info(f"[Simulator] ✓ Prefill complete, results saved to {output_file}")
        return list(events.values())
    
    async def run_decode_only(
        self,
        input_file: str = "prefill_results.pkl",
        output_file: str = "decode_results.pkl",
        max_output_len: int = None,
    ):
        """
        Run ONLY Decode stage using kvserve engine, load prefill results from file.
        This should be called in a separate process for TP>1.
        """
        import ray
        from kvserve.engine.stage_engine import DecodeEngine
        from kvserve.engine.kv_transfer import KVTransferManager, TransferMethod
        from kvserve.engine.utils import Request, MigratingRequest, EngineStage
        
        log_info(f"[Simulator] Phase 2: Running Decode stage...")
        
        # Reset network simulator state to avoid cross-experiment contamination
        self.network_sim.reset()
        
        # Load prefill results
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        
        events: Dict[str, RequestEvent] = data['events']
        prompts = data['prompts']
        max_tokens = data['max_tokens']
        if max_output_len is not None:
            max_tokens = max_output_len
        temperature = data['temperature']
        manifest_file = data['manifest_file']
        
        log_info(f"[Simulator] Loaded {len(events)} events from prefill")
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                num_gpus=self.tensor_parallel_size,
            )
        
        # Create simulation directory
        sim_dir = self.simulation_kv_dir
        # Convert to absolute path
        sim_dir = os.path.abspath(sim_dir)
        log_info(f"[Simulator] KV cache directory: {sim_dir}")
        
        # Create KV transfer manager and load manifest
        kv_transfer_manager = KVTransferManager(
            transfer_method=TransferMethod.SIMULATION,
            simulation_dir=sim_dir,
        )
        kv_transfer_manager.load_simulation_manifest(manifest_file)
        
        # Create bridge queue and prepare migrating requests
        bridge_queue = asyncio.Queue()
        
        # Create DecodeEngine
        log_info(f"[Simulator] Creating DecodeEngine (TP={self.tensor_parallel_size})...")
        
        # Calculate reasonable max_num_gpu_blocks based on max_batch_size and max_model_len
        block_size = self.block_size
        max_blocks_per_req = (self.max_model_len + block_size - 1) // block_size
        max_num_gpu_blocks = int(self.max_batch_size * max_blocks_per_req * 1.1)
        log_info(f"[Simulator] Calculated max_num_gpu_blocks={max_num_gpu_blocks} "
                 f"(batch={self.max_batch_size}, max_len={self.max_model_len}, blocks_per_req={max_blocks_per_req})")
        
        self.decode_engine = DecodeEngine(
            prefill_decode_bridge_queue=bridge_queue,
            model_path=self.model_path,
            num_workers=1,
            tensor_parallel_size=self.tensor_parallel_size,
            block_size=block_size,
            max_num_gpu_blocks=max_num_gpu_blocks,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            kv_transfer_manager=kv_transfer_manager,
            nccl_init_method=f"tcp://localhost:{pick_free_port(29600)}",  # Different port from prefill
            nccl_world_size=self.tensor_parallel_size,  # Only decode workers
            max_batch_size=self.max_batch_size,
            compression_config=self.compression_config,
            service_config=self.service_config,
        )
        
        await self.decode_engine.initialize()
        
        # Set up output callback to collect results (async function)
        decode_outputs = []
        async def on_output(output):
            decode_outputs.append(output)
        self.decode_engine.set_output_callback(on_output)
        
        # Create tokenizer for encoding prompts
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Create migrating requests based on arrival time
        manifest = kv_transfer_manager.get_simulation_manifest()
        sorted_requests = sorted(events.items(), key=lambda x: x[1].t_net_arrive)
        
        log_info(f"[Simulator] Submitting requests to decode based on arrival time...")
        
        # Save request objects to access compute time later (even after they're removed from scheduler)
        request_objects = {}
        submit_done = asyncio.Event()
        
        # Task to submit requests based on arrival time
        async def submit_requests():
            current_time_ms = 0.0
            for req_id, event in sorted_requests:
                # Wait until arrival time
                wait_time_ms = event.t_net_arrive - current_time_ms
                if wait_time_ms > 0:
                    await asyncio.sleep(wait_time_ms / 1000.0)
                    current_time_ms = event.t_net_arrive
                
                # Create Request object for decode
                request = Request(
                    request_id=req_id,
                    prompt=prompts[int(req_id.split('_')[1])],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    arrival_time=time.time(),
                )
                request.prompt_token_ids = event.prompt_token_ids
                # Use a dummy first token as output from prefill (since we're simulating)
                # In real PD separation, prefill generates the first token  
                # Use prompt's last token as a reasonable dummy (simulates continuation)
                if request.prompt_token_ids and len(request.prompt_token_ids) > 0:
                    request.output_token_ids = [request.prompt_token_ids[-1]]  # Last token of prompt as dummy
                else:
                    request.output_token_ids = [128001]  # BOS token as fallback
                
                # Create MigratingRequest
                # For TP>1, check for shard-specific keys (req_0_tp0, req_0_tp1, ...)
                # All TP workers share the same block table, so use the first shard's info
                shard_req_id = f"{req_id}_tp0" if f"{req_id}_tp0" in manifest else req_id
                
                if shard_req_id in manifest:
                    info = manifest[shard_req_id]
                    migrating_req = MigratingRequest(
                        req=request,
                        kv_block_indexes=info.get('src_blocks', [0]),
                        output_token_ids=request.output_token_ids,
                        expanded_prompt_token_ids=event.prompt_token_ids,
                        source_stage=EngineStage.PREFILL,
                        target_stage=EngineStage.DECODING,
                    )
                    
                    # Save request object to access compute time later
                    request_objects[req_id] = request
                    
                    # Put into bridge queue
                    await bridge_queue.put(migrating_req)
                    
                    # Record decode submission time
                    event.t_d_submit = current_time_ms
                    event.io_unpack_ms = self._estimate_io_time(event.kv_size_bytes)
                else:
                    log_error(f"[Simulator] Manifest entry not found for {req_id} (tried {shard_req_id})")
                    log_debug(f"[Simulator] Available manifest keys: {list(manifest.keys())[:10]}...")  # Show first 10 keys
            submit_done.set()
        
        # Task to run decode engine
        async def run_decode_loop():
            max_iterations = 10000
            iteration = 0
            while (not submit_done.is_set() or
                   bridge_queue.qsize() > 0 or
                   self.decode_engine.scheduler.waiting_queue or
                   self.decode_engine.scheduler.running_requests):
                await self.decode_engine.step()
                await asyncio.sleep(0.001)
                iteration += 1
                if iteration >= max_iterations:
                    log_warning("[Simulator] Decode loop reached max iterations, exiting.")
                    break
                # If submitter still running but no work yet, avoid busy wait
                if (not submit_done.is_set() and
                        bridge_queue.qsize() == 0 and
                        not self.decode_engine.scheduler.waiting_queue and
                        not self.decode_engine.scheduler.running_requests):
                    await asyncio.sleep(0.005)
        
        log_info(f"[Simulator] Running decode...")
        t_decode_start = time.time()
        
        # Run both tasks concurrently
        await asyncio.gather(
            submit_requests(),
            run_decode_loop()
        )
        
        t_decode_end = time.time()
        decode_duration_ms = (t_decode_end - t_decode_start) * 1000.0
        
        log_info(f"[Simulator] Decode done: {decode_duration_ms:.2f} ms total")
        
        # Get updated manifest with decompression times
        updated_manifest = kv_transfer_manager.get_simulation_manifest()
        
        # Aggregate decode compute time from engine (pure compute only, no scheduling/IO)
        total_decode_compute_ms = self.decode_engine.get_decode_compute_ms() if self.decode_engine else 0.0
        per_request_decode_ms = total_decode_compute_ms / max(1, len(events))
        
        # Update events with actual decode compute timing (only worker.step_decode.remote time)
        for req_id, event in events.items():
            event.decode_compute_ms = per_request_decode_ms
            event.decode_done = True
            
            # Get decompression time from updated manifest
            # For TP>1, check for shard-specific keys (req_0_tp0, req_0_tp1, ...)
            # All TP workers decompress in parallel, so use max time
            if self.tensor_parallel_size > 1:
                # Find all shard keys for this request
                shard_keys = [key for key in updated_manifest.keys() if key.startswith(f"{req_id}_tp")]
                if shard_keys:
                    # Get max decompression time across all shards (parallel decompression)
                    decompression_times = [updated_manifest[key].get('decompression_time_ms', 0) for key in shard_keys]
                    write_times = [updated_manifest[key].get('write_time_ms', 0) for key in shard_keys]
                    event.decompression_time_ms = max(decompression_times) if decompression_times else 0
                    event.write_time_ms = max(write_times) if write_times else 0
                else:
                    event.decompression_time_ms = 0
                    event.write_time_ms = 0
            else:
                # TP=1: Direct lookup
                if req_id in updated_manifest:
                    info = updated_manifest[req_id]
                    event.decompression_time_ms = info.get('decompression_time_ms', 0)
                    event.write_time_ms = info.get('write_time_ms', 0)
            
            # CRITICAL FIX: t_d_end must include decompression time
            # Timeline: ... → [network arrive] → [decompress] → [decode] → end
            # Note: io_unpack_ms is simulator-internal and doesn't affect real latency
            event.t_d_end = event.t_net_arrive + event.decompression_time_ms + event.decode_compute_ms
        
        # Update controller bandit with simulated observations (batch window)
        if (
            self.service_config
            and hasattr(self.compression_config, "select_profile")
            and hasattr(self.compression_config, "update")
        ):
            bandwidth_gbps = self.service_config.bandwidth_gbps
            B_mbps = (bandwidth_gbps * 1000.0) / 8.0
            acc_req = self.service_config.accuracy_requirement
            update_window = 5
            pending_updates = []

            for event in events.values():
                input_length = len(event.prompt_token_ids)
                V_bytes = event.original_kv_size_bytes or event.kv_size_bytes
                T_model_ms = event.compute_only_ms
                profile, context = self.compression_config.select_profile(
                    input_length=input_length,
                    V_bytes=V_bytes,
                    B_mbps=B_mbps,
                    T_model_ms=T_model_ms,
                    acc_req=acc_req
                )
                if not profile or not context:
                    continue

                V_mb = V_bytes / (1024 * 1024)
                if profile.harmonic_speed > 0:
                    T_codec_ms = (V_mb / profile.harmonic_speed) * 1000.0
                    offline_decompress_ms = max(0.0, T_codec_ms - event.compression_time_ms)
                else:
                    offline_decompress_ms = 0.0

                # Prepare metrics for controller update
                metrics = {
                    'compression_time_ms': event.compression_time_ms,
                    'decompression_time_ms': offline_decompress_ms,
                    'kv_size_bytes': V_bytes
                }
                pending_updates.append((context, metrics))

                if len(pending_updates) >= update_window:
                    for ctx, mtx in pending_updates:
                        self.compression_config.update(ctx, mtx)
                    pending_updates.clear()

            if pending_updates:
                for ctx, mtx in pending_updates:
                    self.compression_config.update(ctx, mtx)
        
        # Save final results
        with open(output_file, 'wb') as f:
            pickle.dump({'events': events}, f)
        
        # Shutdown
        log_info(f"[Simulator] Cleaning up Decode engine...")
        self.decode_engine = None
        ray.shutdown()
        
        log_info(f"[Simulator] ✓ Decode complete, results saved to {output_file}")
        return list(events.values())
    
    def print_stats(self, results: List[RequestEvent]):
        """Print detailed statistics"""
        print("\n" + "="*80)
        print("SIMULATION RESULTS")
        print("="*80)
        
        print(f"\n📊 Configuration:")
        print(f"  Model: {self.model_path}")
        print(f"  TP: {self.tensor_parallel_size}")
        print(f"  Network: {self.network_sim.throughput_gbps} Gbps (efficiency={self.network_sim.efficiency})")
        print(f"  Max Concurrent: {self.network_sim.max_concurrent}")
        print(f"  PCIe: {self.pcie_gbps} GB/s")
        print(f"  Compression: {'ENABLED' if self._compression_enabled() else 'DISABLED'}")
        
        print(f"\n📈 Per-Request Timeline (ms) and KV Size:")
        header = (
            f"{'ID':<8}"
            f"{'t_p_end':>12}"
            f"{'t_net':>12}"
            f"{'t_d_end':>12}"
            f"{'compress':>11}"
            f"{'decompress':>11}"
            f"{'net_q':>7}"
            f"{'net_tx':>7}"
            f"{'KV_MB':>10}"
        )
        print(header)
        print("-" * len(header))
        
        for r in results:
            kv_mb = r.kv_size_bytes / (1024**2)
            print(
                f"{r.request_id:<8}"
                f"{r.t_p_end:>12.2f}"
                f"{r.t_net_arrive:>12.2f}"
                f"{r.t_d_end:>12.2f}"
                f"{r.compression_time_ms:>11.2f}"
                f"{r.decompression_time_ms:>11.2f}"
                f"{r.network_queue_ms:>7.2f}"
                f"{r.network_transfer_ms:>7.2f}"
                f"{kv_mb:>10.2f}"
            )
        
        # Aggregate stats
        avg_latency = sum(r.total_latency_ms for r in results) / len(results)
        avg_compute = sum(r.compute_only_ms for r in results) / len(results)
        avg_prefill = sum(r.prefill_compute_ms for r in results) / len(results)
        avg_decode = sum(r.decode_compute_ms for r in results) / len(results)
        avg_network = sum(r.network_queue_ms + r.network_transfer_ms for r in results) / len(results)
        avg_io = sum(r.io_pack_ms + r.io_unpack_ms for r in results) / len(results)
        avg_compression = sum(r.compression_time_ms for r in results) / len(results)
        avg_decompression = sum(r.decompression_time_ms for r in results) / len(results)
        
        # KV size statistics
        avg_kv_bytes = sum(r.kv_size_bytes for r in results) / len(results)
        avg_kv_mb = avg_kv_bytes / (1024**2)
        
        # Manual calculation of expected transfer time
        # Note: size_gb is in GB (bytes), throughput_gbps is in Gbps (bits)
        # Convert GB to Gb: 1 GB = 8 Gb
        size_gb = avg_kv_bytes / (1024**3)
        size_gb_in_bits = size_gb * 8  # Convert GB to Gb
        expected_transfer_ms = size_gb_in_bits / (self.network_sim.throughput_gbps * self.network_sim.efficiency) * 1000.0
        
        print(f"\n📊 Aggregate Statistics:")
        print(f"  Total Requests: {len(results)}")
        print(f"  Avg Total Latency: {avg_latency:.2f} ms")
        print(f"  Avg Compute Time: {avg_compute:.2f} ms (Prefill: {avg_prefill:.2f}, Decode: {avg_decode:.2f})")
        print(f"  Avg Network Time: {avg_network:.2f} ms")
        print(f"  Avg IO Time: {avg_io:.2f} ms")
        
        if self._compression_enabled():
            print(f"  Avg Compression Time: {avg_compression:.2f} ms")
            print(f"  Avg Decompression Time: {avg_decompression:.2f} ms")
            avg_ratio = sum(r.compression_ratio for r in results if r.compression_ratio > 0) / len([r for r in results if r.compression_ratio > 0])
            print(f"  Avg Compression Ratio: {avg_ratio:.2f}x")
        
        print(f"\n📦 KV Cache Size & Network Calculation:")
        print(f"  Avg KV Size: {avg_kv_mb:.2f} MB ({avg_kv_bytes:,} bytes)")
        print(f"  Network Config: {self.network_sim.throughput_gbps} Gbps × {self.network_sim.efficiency} efficiency = {self.network_sim.throughput_gbps * self.network_sim.efficiency:.1f} Gbps effective")
        print(f"  Expected Transfer Time: {size_gb:.6f} GB × 8 / {self.network_sim.throughput_gbps * self.network_sim.efficiency:.1f} Gbps × 1000 = {expected_transfer_ms:.4f} ms (transfer only, excludes queue)")
        print(f"  Actual Avg Network Time: {avg_network:.2f} ms (includes queue + transfer time)")
        print("=" * 80 + "\n")
