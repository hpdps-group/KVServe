"""
Stage Engines for PD separation
Implements Prefill and Decoding stages
"""

import asyncio
import time
from typing import List, Dict, Optional, Any
from collections import deque

import ray

from kvserve.engine.utils import (
    EngineStage,
    EngineStatus,
    Request,
    BatchedRequests,
    MigratingRequest,
    StepOutput,
    KVTransferStatus,
    pick_free_port,
)
from kvserve.engine.service_config import ServiceConfig
from kvserve.engine.block_manager import BlockManager, BlockLocation
from kvserve.engine.worker import Worker
from kvserve.engine.kv_transfer import KVTransferManager
from kvserve.engine.logger import log_info, log_debug, log_warning, log_error


class StageScheduler:
    """Scheduler for stage engines with memory-aware scheduling"""
    
    def __init__(self, stage: EngineStage, block_manager=None, max_tokens_per_batch: int = 32768):
        self.stage = stage
        self.waiting_queue: deque[Request] = deque()
        self.running_requests: Dict[str, Request] = {}
        self.block_manager = block_manager
        self.max_tokens_per_batch = max_tokens_per_batch
    
    def add_request(self, request: Request):
        """Add request to waiting queue"""
        self.waiting_queue.append(request)
    
    def can_allocate_request(self, request: Request) -> bool:
        """Check if we have enough GPU blocks for this request"""
        if not self.block_manager:
            return True
        return self.block_manager.can_allocate(request)
    
    def schedule(self, max_batch_size: int = 32) -> Optional[BatchedRequests]:
        """
        Schedule a batch of requests
        
        For DECODING stage: Schedule from both running_requests (for continuous generation)
        and waiting_queue (for new requests from prefill)
        """
        batch_requests = []
        total_tokens = 0
        
        # For DECODING stage, prioritize running requests (continuous generation)
        # This ensures already-started requests continue generating tokens
        if self.stage == EngineStage.DECODING:
            # First, add running requests that need more tokens
            running_list = list(self.running_requests.values())
            for request in running_list:
                if len(batch_requests) >= max_batch_size:
                    break
                
                # Check token budget
                tokens = request.get_total_len()
                if total_tokens + tokens > self.max_tokens_per_batch:
                    break
                
                batch_requests.append(request)
                total_tokens += tokens
        
        # Then add new requests from waiting queue (if batch not full)
        while self.waiting_queue and len(batch_requests) < max_batch_size:
            request = self.waiting_queue[0]
            
            # Check token budget
            tokens = request.get_total_len()
            # ✅ FIX: Allow at least one request even if it exceeds max_tokens_per_batch
            # This prevents deadlock when a single request is larger than the batch limit
            if batch_requests and total_tokens + tokens > self.max_tokens_per_batch:
                break
            
            # Check block allocation (MEMORY-AWARE)
            if not self.can_allocate_request(request):
                # Log memory pressure
                if self.block_manager:
                    avail_blocks = self.block_manager.get_num_avail_gpu_blocks()
                    blocks_needed = self.block_manager.get_num_blocks_needed(request)
                    # Use debug level to avoid log spam during normal memory pressure
                    log_debug(f"[{self.stage.value}Scheduler] Memory pressure: Cannot allocate {blocks_needed} blocks for {request.request_id}, "
                              f"available={avail_blocks}, keeping in waiting queue ({len(self.waiting_queue)} waiting)")
                break  # Stop adding new requests
            
            # Add to batch
            batch_requests.append(self.waiting_queue.popleft())
            self.running_requests[request.request_id] = request
            total_tokens += tokens
        
        if batch_requests:
            # ✅ Diagnostic: print scheduler state periodically
            if not hasattr(self, '_schedule_call_count'):
                self._schedule_call_count = 0
                self._last_schedule_print = 0
            
            self._schedule_call_count += 1
            
            return BatchedRequests(requests=batch_requests)
        return None
    
    def finish_request(self, request_id: str):
        """Mark request as finished"""
        if request_id in self.running_requests:
            del self.running_requests[request_id]
    
    def num_waiting_requests(self) -> int:
        """Get number of waiting requests"""
        return len(self.waiting_queue)
    
    def num_running_requests(self) -> int:
        """Get number of running requests"""
        return len(self.running_requests)


class BaseStageEngine:
    """Base class for stage engines"""
    
    def __init__(
        self,
        stage: EngineStage,
        model_path: str,
        num_workers: int = 1,
        block_size: int = 16,
        max_num_gpu_blocks: int = 5000,
        max_num_cpu_blocks: int = 1000,
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        activation_memory_gb: Optional[float] = None,
        kv_transfer_manager: Optional[KVTransferManager] = None,
        nccl_init_method: Optional[str] = None,
        nccl_world_size: Optional[int] = None,
        max_model_len: int = 32768,
        compression_config: Optional[Dict[str, Any]] = None,
        service_config: Optional[ServiceConfig] = None,
    ):
        self.stage = stage
        self.model_path = model_path
        self.num_workers = num_workers
        self.block_size = block_size
        self.max_num_gpu_blocks = max_num_gpu_blocks
        self.max_num_cpu_blocks = max_num_cpu_blocks
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.activation_memory_gb = activation_memory_gb
        self.kv_transfer_manager = kv_transfer_manager
        self.nccl_init_method = nccl_init_method
        self.nccl_world_size = nccl_world_size
        self.tp_init_method = None
        self.max_model_len = max_model_len
        self.compression_config = compression_config
        self.service_config = service_config or ServiceConfig()
        
        # Workers
        self.workers: List[ray.ObjectRef] = []
        
        # Block manager
        self.block_manager = BlockManager(
            stage=stage.value,
            max_num_gpu_blocks=max_num_gpu_blocks,
            max_num_cpu_blocks=max_num_cpu_blocks,
            block_size=block_size,
        )
        
        # Scheduler
        # Limit max_tokens_per_batch to prevent activation memory OOM.
        # Derive from available decode blocks instead of a fixed constant.
        # We do NOT cap by max_model_len here; batching can include multiple shorter requests.
        budget_tokens = int(self.block_size * self.block_manager.max_num_gpu_blocks * 0.8)
        max_tokens_per_batch = budget_tokens
        self.scheduler = StageScheduler(stage, block_manager=self.block_manager, max_tokens_per_batch=max_tokens_per_batch)
        
        # Status
        self.status = EngineStatus.INACTIVE
        
        # Event loop control
        self.pls_stop_loop = asyncio.Event()
        self.is_loop_stopped = asyncio.Event()
        
        # Load balancing
        self.current_worker_index = 0
    
    async def update_service_config(self, **kwargs):
        """
        Update service configuration for all workers
        
        Args:
            **kwargs: Service config parameters to update
        """
        if not self.workers:
            self.service_config.update(**kwargs)
            return
        
        # Update local config
        self.service_config.update(**kwargs)
        
        # Update all workers
        await asyncio.gather(*[
            worker.update_service_config.remote(**kwargs)
            for worker in self.workers
        ])
        log_info(f"[{self.stage.value}Engine] Service config updated for all workers: {kwargs}")
    
    def get_next_worker(self):
        """Get next worker using round-robin"""
        if not self.workers:
            return None
        worker = self.workers[self.current_worker_index]
        self.current_worker_index = (self.current_worker_index + 1) % len(self.workers)
        return worker
    
    async def initialize(self):
        """Initialize workers"""
        log_info(f"[{self.stage.value}Engine] Initializing {self.num_workers} workers...")
        
        # ========================================================================
        # Create workers for TP group
        # ========================================================================
        # For TP>1: Create tensor_parallel_size workers, each with 1 GPU
        # For TP=1: Create num_workers workers (for load balancing)
        # ========================================================================
        num_workers_to_create = self.tensor_parallel_size if self.tensor_parallel_size > 1 else self.num_workers
        
        log_info(f"[{self.stage.value}Engine] Creating {num_workers_to_create} workers (TP={self.tensor_parallel_size})")
        
        # ========================================================================
        # CRITICAL: Create all workers first, THEN initialize in parallel
        # This avoids deadlock when TP>1 (workers wait for each other to join TP group)
        # ========================================================================
        if self.tensor_parallel_size > 1:
            stage_offset = 0 if self.stage == EngineStage.PREFILL else 1
            base_port = int(self.nccl_init_method.split(':')[-1]) if self.nccl_init_method else 29500
            tp_port = base_port + 100 + stage_offset
            chosen_port = pick_free_port(tp_port)
            if chosen_port != tp_port:
                log_warning(
                    f"[{self.stage.value}Engine] TP port {tp_port} in use, switching to {chosen_port}"
                )
            self.tp_init_method = f"tcp://localhost:{chosen_port}"
            log_info(f"[{self.stage.value}Engine] TP init method: {self.tp_init_method}")
        
        # Step 1: Create all worker actors
        for tp_rank in range(num_workers_to_create):
            worker_id = tp_rank
            
            # Calculate global rank for P2P group
            if self.stage == EngineStage.PREFILL:
                global_rank = tp_rank
            else:
                # Decode workers come after prefill workers in the SAME process
                # But in SIMULATION mode (nccl_world_size == tensor_parallel_size),
                # prefill and decode run separately, so decode starts from rank 0
                if self.nccl_world_size == self.tensor_parallel_size:
                    # SIMULATION mode: decode in separate process
                    global_rank = tp_rank
                else:
                    # Normal PD separation: decode after prefill
                    global_rank = num_workers_to_create + tp_rank
            
            # Allocate GPUs: 1 GPU per worker (workers form TP group)
            worker = Worker.options(num_gpus=1).remote(
                worker_id=worker_id,
                stage=self.stage,
                model_path=self.model_path,
                block_size=self.block_size,
                dtype=self.dtype,
                tensor_parallel_size=self.tensor_parallel_size,
                tp_rank=tp_rank,  # NEW: Rank within TP group
                gpu_memory_utilization=self.gpu_memory_utilization,
                activation_memory_gb=self.activation_memory_gb,
                global_rank=global_rank,
                world_size=self.nccl_world_size,
                nccl_init_method=self.nccl_init_method,
                tp_init_method=self.tp_init_method,
                max_model_len=self.max_model_len,
                compression_config=self.compression_config,
                service_config=self.service_config,
            )
            
            self.workers.append(worker)
            log_info(f"[{self.stage.value}Engine] Created Worker {worker_id}: tp_rank={tp_rank}, global_rank={global_rank}")
        
        # Step 2: Initialize all workers in parallel (CRITICAL for TP>1)
        log_info(f"[{self.stage.value}Engine] Initializing {len(self.workers)} workers in parallel...")
        await asyncio.gather(*[worker.ready.remote() for worker in self.workers])
        log_info(f"[{self.stage.value}Engine] All workers ready, loading models...")
        
        await asyncio.gather(*[worker.init_model.remote() for worker in self.workers])
        log_info(f"[{self.stage.value}Engine] All workers initialized")
        
        # ========================================================================
        # ✅ DYNAMIC KV CACHE SIZING (vLLM standard approach)
        # ========================================================================
        # Profile GPU memory and calculate optimal block count
        log_info(f"[{self.stage.value}Engine] Profiling GPU memory for KV cache sizing...")
        block_profiles = await asyncio.gather(*[worker.profile_num_available_blocks.remote() for worker in self.workers])
        
        # Use the minimum across all workers for safety
        min_gpu_blocks = min(profile['num_gpu_blocks'] for profile in block_profiles)
        min_cpu_blocks = min(profile['num_cpu_blocks'] for profile in block_profiles)
        
        log_info(f"[{self.stage.value}Engine] Dynamic KV cache sizing:")
        log_info(f"  ✓ num_gpu_blocks: {min_gpu_blocks} (was {self.max_num_gpu_blocks})")
        log_info(f"  ✓ num_cpu_blocks: {min_cpu_blocks} (was {self.max_num_cpu_blocks})")
        
        # Update block manager with profiled values
        self.block_manager.max_num_gpu_blocks = min_gpu_blocks
        self.block_manager.max_num_cpu_blocks = min_cpu_blocks
        self.block_manager._reset_free_blocks()  # Reinitialize free block lists
        log_info(f"[{self.stage.value}Engine] ✓ Block manager updated with dynamic sizing")
        
        # Update config for consistency
        self.max_num_gpu_blocks = min_gpu_blocks
        self.max_num_cpu_blocks = min_cpu_blocks
        
        # Now initialize KV cache with profiled values (use min for consistency)
        for worker_id, worker in enumerate(self.workers):
            # Calculate global rank consistently with worker creation logic
            if self.stage == EngineStage.PREFILL:
                global_rank = worker_id
            else:
                # Decode workers come after prefill workers in the SAME process
                # But in SIMULATION mode (nccl_world_size == tensor_parallel_size),
                # prefill and decode run separately, so decode starts from rank 0
                if self.nccl_world_size == self.tensor_parallel_size:
                    # SIMULATION mode: decode in separate process
                    global_rank = worker_id
                else:
                    # Normal PD separation: decode after prefill
                    global_rank = self.num_workers + worker_id
            
            # Use min values to ensure consistency across workers
            await worker.init_kvcache.remote(min_gpu_blocks, min_cpu_blocks)
            
            # Register with KV transfer manager
            if self.kv_transfer_manager:
                self.kv_transfer_manager.register_worker(global_rank, worker)
        
        self.status = EngineStatus.ACTIVE
        log_info(f"[{self.stage.value}Engine] Initialized {len(self.workers)} workers")
    
    async def start_event_loop(self):
        """Start the event loop for this stage"""
        self.pls_stop_loop.clear()
        self.is_loop_stopped.clear()
        
        log_info(f"[{self.stage.value}Engine] Starting event loop...")
        
        # Track iterations for periodic logging
        iteration = 0
        last_block_usage_print = 0
        
        while not self.pls_stop_loop.is_set():
            try:
                await self.step()
                await asyncio.sleep(0.001)  # Small delay to prevent busy waiting
                
                # Periodically print block usage (every 1000 iterations ≈ 1 second)
                iteration += 1
                if iteration - last_block_usage_print >= 1000:
                    if self.block_manager and (self.scheduler.num_waiting_requests() > 0 or self.scheduler.num_running_requests() > 0):
                        self.block_manager.print_block_usage()
                    last_block_usage_print = iteration
                    
            except Exception as e:
                log_error(f"[{self.stage.value}Engine] Error in event loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(0.1)
        
        self.is_loop_stopped.set()
        log_info(f"[{self.stage.value}Engine] Event loop stopped")
    
    async def stop_event_loop(self):
        """Stop the event loop"""
        log_info(f"[{self.stage.value}Engine] Stopping event loop...")
        self.pls_stop_loop.set()
        
        try:
            await asyncio.wait_for(self.is_loop_stopped.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            # Event loop didn't stop gracefully, but that's okay
            # This can happen if the loop never started or crashed early
            log_warning(f"[{self.stage.value}Engine] Event loop stop timeout (may not have been running)")
    
    async def step(self):
        """Execute one step - to be implemented by subclasses"""
        raise NotImplementedError


class PrefillEngine(BaseStageEngine):
    """Prefill stage engine"""
    
    def __init__(self, prefill_decode_bridge_queue: asyncio.Queue, max_batch_size: int = 16, **kwargs):
        super().__init__(stage=EngineStage.PREFILL, **kwargs)
        self.prefill_decode_bridge_queue = prefill_decode_bridge_queue
        self.max_batch_size = max_batch_size
    
    async def step(self):
        """Execute one prefill step"""
        # Schedule a batch
        max_batch_size = getattr(self, 'max_batch_size', 16)
        
        # Log scheduling attempt if there are waiting requests
        if self.scheduler.waiting_queue:
            waiting_req = self.scheduler.waiting_queue[0]
            blocks_needed = self.block_manager.get_num_blocks_needed(waiting_req)
            blocks_avail = self.block_manager.get_num_avail_gpu_blocks()
            
            if blocks_avail < blocks_needed:
                # Memory pressure detected - use debug to avoid spam
                log_debug(f"[PrefillEngine] Memory pressure: {len(self.scheduler.waiting_queue)} waiting, "
                          f"next={waiting_req.request_id} needs {blocks_needed} blocks but only {blocks_avail} available")
                # Only print block usage once every 100 steps to avoid spam
                if not hasattr(self, '_memory_pressure_log_count'):
                    self._memory_pressure_log_count = 0
                self._memory_pressure_log_count += 1
                if self._memory_pressure_log_count % 100 == 1:
                    log_warning(f"[PrefillEngine] ⚠️  Prolonged memory pressure: {len(self.scheduler.waiting_queue)} requests waiting")
                    self.block_manager.print_block_usage()
            else:
                log_debug(f"[PrefillEngine] Scheduling attempt: {len(self.scheduler.waiting_queue)} waiting, "
                         f"next={waiting_req.request_id} needs {blocks_needed} blocks, available={blocks_avail}")
        
        batched_requests = self.scheduler.schedule(max_batch_size=max_batch_size)
        
        if not batched_requests:
            # Log why scheduling failed
            if self.scheduler.waiting_queue:
                log_warning(f"[PrefillEngine] Failed to schedule, {len(self.scheduler.waiting_queue)} requests waiting, "
                           f"{self.block_manager.get_num_avail_gpu_blocks()} blocks available")
            await asyncio.sleep(0.01)
            return
        
        # ✅ OPTIMIZATION: Use minimal allocation for prefill
        # Only allocate prompt + small margin (16 tokens)
        # Will expand dynamically in decode stage as needed
        self.block_manager.allocate_blocks_batched(batched_requests, minimal=True)
        
        # Get block tables
        kv_block_tables = {
            req.request_id: self.block_manager.get_block_table(req.request_id)
            for req in batched_requests.requests
        }
        
        # Record prefill start time
        start_time = time.time()
        for req in batched_requests.requests:
            if req.prefill_start_time is None:
                req.prefill_start_time = start_time
        
        # Execute prefill
        log_debug(f"[PrefillEngine] Executing prefill for {len(batched_requests.requests)} requests")
        
        # ⚠️ CRITICAL FOR TP>1: ALL workers must be called simultaneously
        # Ray remote actors require explicit calls on each worker
        # vLLM's NCCL will then synchronize them internally
        if self.tensor_parallel_size > 1:
            # TP>1: Call ALL workers in parallel
            results = await asyncio.gather(*[
                worker.step_prefill.remote(batched_requests, kv_block_tables)
                for worker in self.workers
            ])
            # All workers return same outputs (synchronized via NCCL)
            outputs, expanded_tokens_map = results[0]
        else:
            # TP=1: Single worker
            worker = self.get_next_worker()
            if not worker:
                return
            outputs, expanded_tokens_map = await worker.step_prefill.remote(
                batched_requests,
                kv_block_tables
            )
        
        log_debug(f"[PrefillEngine] Prefill completed: {len(outputs)} outputs, expanded_tokens_map keys: {list(expanded_tokens_map.keys())}")
        
        # Record prefill end time
        end_time = time.time()
        for req in batched_requests.requests:
            req.prefill_end_time = end_time
        
        # Send to decode stage
        for request in batched_requests.requests:
            # Update request with expanded tokens and output
            if request.request_id in expanded_tokens_map:
                request.prompt_token_ids = expanded_tokens_map[request.request_id]
            
            output = next((o for o in outputs if o.request_id == request.request_id), None)
            if output:
                request.output_token_ids = output.output_token_ids
                log_debug(f"[PrefillEngine] Request {request.request_id}: prefill output tokens={len(output.output_token_ids)}")
            else:
                log_warning(f"[PrefillEngine] No output found for request {request.request_id}")
            
            # Create migrating request
            migrating_req = MigratingRequest(
                req=request,
                kv_block_indexes=kv_block_tables[request.request_id],
                output_token_ids=request.output_token_ids,
                expanded_prompt_token_ids=request.prompt_token_ids,
                source_stage=EngineStage.PREFILL,
                target_stage=EngineStage.DECODING,
            )
            
            # In SIMULATION mode, save KV to file immediately
            if self.kv_transfer_manager and hasattr(self.kv_transfer_manager, 'transfer_method'):
                from kvserve.engine.kv_transfer import TransferMethod
                if self.kv_transfer_manager.transfer_method == TransferMethod.SIMULATION:
                    # For TP>1: Save KV from ALL workers (each worker has a shard)
                    # For TP=1: Save from single worker
                    workers_to_save = self.workers if self.tensor_parallel_size > 1 else [self.get_next_worker()]
                    
                    # Parallel save KV from all workers
                    async def save_kv_shard(worker_idx, worker):
                        if not worker:
                            return False
                        src_rank = await worker.get_global_rank.remote()
                        # For TP>1, append tp_rank to request_id to distinguish shards
                        req_id_for_file = f"{request.request_id}_tp{worker_idx}" if self.tensor_parallel_size > 1 else request.request_id
                        
                        # Call transfer to save KV to file
                        success = await self.kv_transfer_manager.transfer_kv_cache(
                            request_id=req_id_for_file,
                            src_rank=src_rank,
                            dst_rank=src_rank,  # Placeholder
                            src_blocks=migrating_req.kv_block_indexes,
                            dst_blocks=migrating_req.kv_block_indexes,
                        )
                        if success:
                            log_debug(f"[PrefillEngine] Saved KV shard {worker_idx} to file for {request.request_id}")
                        else:
                            log_warning(f"[PrefillEngine] Failed to save KV shard {worker_idx} for {request.request_id}")
                        return success
                    
                    # Save all shards in parallel
                    results = await asyncio.gather(*[
                        save_kv_shard(worker_idx, worker) 
                        for worker_idx, worker in enumerate(workers_to_save)
                    ])
                    # Check if all saves succeeded
                    if not all(results):
                        log_warning(f"[PrefillEngine] Some KV shards failed to save for {request.request_id}")
            
            log_debug(f"[PrefillEngine] Sending request {request.request_id} to decode (output_tokens={len(migrating_req.output_token_ids or [])})")
            await self.prefill_decode_bridge_queue.put(migrating_req)
            
            # ✅ Free prefill blocks immediately after sending to decode
            # KV cache is transferred to decode stage, prefill no longer needs these blocks
            if request.request_id in self.block_manager.block_table:
                log_debug(f"[PrefillEngine] Freeing {len(self.block_manager.block_table[request.request_id])} blocks for {request.request_id}")
                self.block_manager.free_blocks(request.request_id)
            
            # Finish request in prefill scheduler
            self.scheduler.finish_request(request.request_id)


class DecodeEngine(BaseStageEngine):
    """Decode stage engine"""
    
    def __init__(self, prefill_decode_bridge_queue: asyncio.Queue, max_batch_size: int = 32, enable_multi_stream: bool = False, **kwargs):
        super().__init__(stage=EngineStage.DECODING, **kwargs)
        self.prefill_decode_bridge_queue = prefill_decode_bridge_queue
        self.max_batch_size = max_batch_size
        self.output_callback = None
        # Accumulate pure decode compute time (seconds) across the whole run
        self.total_decode_compute_time = 0.0
        # Requests waiting for GPU blocks (do not count toward active capacity)
        self.mem_wait_queue = deque()  # type: ignore[var-annotated]
        # For requests that haven't allocated blocks yet (need migrating metadata)
        self.mem_wait_meta: Dict[str, MigratingRequest] = {}
        
        # Multi-stream optimization
        self.enable_multi_stream = enable_multi_stream
        if self.enable_multi_stream:
            import ray
            
            # ✅ TRUE ASYNC: Manage transfer ObjectRefs (not tasks)
            self.transfer_refs = {}        # {request_id: ray.ObjectRef}
            self.transfer_workers = {}     # {request_id: worker_ref}
            self.transfer_start_times = {} # {request_id: start_time}
            self.transfer_metadata = {}    # {request_id: metadata dict}
            
            self.transfer_queue = deque()  # Requests with transfer started
            self.ready_queue = deque()     # Requests ready for decode
            
            # Transfer statistics
            self.transfer_stats = {
                "total_transfers": 0,
                "successful_transfers": 0,
                "failed_transfers": 0,
                "total_transfer_time": 0.0,
                "total_wait_time": 0.0,      # Time spent waiting at sync point
            }
            log_info(f"[DecodeEngine] 🚀 Multi-stream optimization ENABLED (TRUE ASYNC)")
    
    def set_output_callback(self, callback):
        """Set callback for outputs"""
        self.output_callback = callback

    # ---------- Capacity helpers ----------
    def _get_active_count(self) -> int:
        """
        Active = running + ready + transferring.
        
        ✅ CRITICAL FIX: Must include transfer_queue!
        Requests in transfer_queue have blocks allocated and are actively processing,
        so they must count toward active capacity to prevent over-admission.
        
        Waiting/mem_wait do NOT consume active capacity.
        """
        active = self.scheduler.num_running_requests()
        if hasattr(self, "ready_queue"):
            active += len(self.ready_queue)
        if hasattr(self, "transfer_queue"):
            active += len(self.transfer_queue)  # ✅ Include transferring requests!
        return active

    def _get_active_count(self) -> int:
        """
        Active = running + ready + transferring.
        
        ✅ CRITICAL FIX: Must include transfer_queue!
        Requests in transfer_queue have blocks allocated and are actively processing,
        so they must count toward active capacity to prevent over-admission.
        
        Waiting/mem_wait do NOT consume active capacity.
        """
        active = self.scheduler.num_running_requests()
        if hasattr(self, "ready_queue"):
            active += len(self.ready_queue)
        if hasattr(self, "transfer_queue"):
            active += len(self.transfer_queue)  # ✅ Include transferring requests!
        return active

    def _mark_mem_wait(self, request: Request, migrating_req: Optional[MigratingRequest] = None):
        """Place request into mem-wait queue without counting toward active."""
        # Avoid duplicates
        if request.request_id not in {r.request_id for r in self.mem_wait_queue}:
            self.mem_wait_queue.append(request)
        if migrating_req:
            self.mem_wait_meta[request.request_id] = migrating_req
        log_debug(f"[DecodeEngine] {request.request_id} moved to mem-wait queue")

    async def _cleanup_request(self, request: Request, reason: str = ""):
        """Unified cleanup: release blocks and remove from all queues/state."""
        req_id = request.request_id
        
        # Remove from scheduler queues
        self.scheduler.waiting_queue = deque([r for r in self.scheduler.waiting_queue if r.request_id != req_id])
        self.scheduler.running_requests.pop(req_id, None)
        
        # Remove from ready/transfer/mem_wait queues
        if hasattr(self, "ready_queue"):
            self.ready_queue = deque([r for r in self.ready_queue if r.request_id != req_id])
        if hasattr(self, "transfer_queue"):
            self.transfer_queue = deque([r for r in self.transfer_queue if r.request_id != req_id])
        self.mem_wait_queue = deque([r for r in self.mem_wait_queue if r.request_id != req_id])
        self.mem_wait_meta.pop(req_id, None)
        
        # Clear async transfer bookkeeping if any
        for attr in ["transfer_refs", "transfer_workers", "transfer_start_times", "transfer_metadata"]:
            if hasattr(self, attr):
                getattr(self, attr).pop(req_id, None)
        
        # Free blocks if allocated
        if req_id in self.block_manager.block_table:
            self.block_manager.free_blocks(req_id)
        
        # Mark state
        request.is_finished = True
        if reason:
            request.finish_reason = reason
        
        # Try to promote mem-wait requests now that blocks may have freed
        await self._drain_mem_wait_queue(multistream=self.enable_multi_stream)

    async def _drain_mem_wait_queue(self, multistream: bool = False):
        """Try to re-activate requests that were waiting for memory."""
        if not self.mem_wait_queue:
            return
        
        # Iterate over a copy to allow removal
        for req in list(self.mem_wait_queue):
            req_id = req.request_id
            migrating_req = self.mem_wait_meta.get(req_id)
            
            try:
                if migrating_req:
                    # Request still needs initial allocation + transfer
                    blocks_needed = len(migrating_req.kv_block_indexes or [])
                    avail_blocks = self.block_manager.get_num_avail_gpu_blocks()
                    if avail_blocks < blocks_needed:
                        continue
                    
                    # Allocate blocks
                    self.block_manager.allocate_blocks(req, num_blocks=blocks_needed)
                    dst_blocks = self.block_manager.get_block_table(req_id)
                    
                    if multistream:
                        # Start async transfer
                        await self._start_async_transfer(req, migrating_req)
                    else:
                        dst_worker = self.get_next_worker()
                        if not dst_worker:
                            continue
                        src_rank = 0
                        dst_rank = await dst_worker.get_global_rank.remote()
                        try:
                            success = await self.kv_transfer_manager.transfer_kv_cache(
                                request_id=req_id,
                                src_rank=src_rank,
                                dst_rank=dst_rank,
                                src_blocks=migrating_req.kv_block_indexes,
                                dst_blocks=dst_blocks,
                            )
                            if not success:
                                await self._cleanup_request(req, reason="transfer_failed")
                                continue
                        except Exception as e:
                            log_error(f"[DecodeEngine] KV transfer error during mem-wait drain: {e}")
                            await self._cleanup_request(req, reason="transfer_exception")
                            continue
                    
                    # Promotion successful
                    self.mem_wait_meta.pop(req_id, None)
                    self.mem_wait_queue.remove(req)
                    self.scheduler.add_request(req)
                    req.decoding_start_time = time.time()
                    log_debug(f"[DecodeEngine] {req_id} re-activated from mem-wait (initial allocation)")
                    continue
                
                # Expansion path: request already has some blocks
                prompt_tokens = len(req.prompt_token_ids) if req.prompt_token_ids else 0
                output_tokens = len(req.output_token_ids) if req.output_token_ids else 0
                total_seq_len = prompt_tokens + output_tokens
                blocks_needed = (total_seq_len + self.block_manager.block_size - 1) // self.block_manager.block_size
                current_blocks = len(self.block_manager.block_table.get(req_id, []))
                additional_blocks_needed = max(0, blocks_needed - current_blocks)
                
                if additional_blocks_needed > 0:
                    avail_blocks = self.block_manager.get_num_avail_gpu_blocks()
                    if avail_blocks < additional_blocks_needed:
                        continue
                    new_blocks = self.block_manager._get_free_blocks(additional_blocks_needed, BlockLocation.GPU)
                    self.block_manager.block_table.setdefault(req_id, []).extend(new_blocks)
                
                # Ready to run again
                self.mem_wait_queue.remove(req)
                self.scheduler.add_request(req)
                log_debug(f"[DecodeEngine] {req_id} re-activated from mem-wait (expansion)")
            
            except Exception as e:
                log_error(f"[DecodeEngine] Error draining mem-wait for {req_id}: {e}")
                await self._cleanup_request(req, reason="mem_wait_drain_error")
    
    async def _start_async_transfer(self, request: Request, migrating_req):
        """
        🚀 TRUE ASYNC: Start transfer and return immediately (non-blocking)
        
        TP Support: For TP>1, transfer all shards in parallel
        """
        if not self.kv_transfer_manager or not migrating_req.kv_block_indexes:
            # No transfer needed
            request.kv_transfer_status = KVTransferStatus.READY
            self.ready_queue.append(request)
            return
        
        try:
            # Allocate blocks (caller has already checked availability)
            blocks_needed = len(migrating_req.kv_block_indexes)
            self.block_manager.allocate_blocks(request, num_blocks=blocks_needed)
            dst_blocks = self.block_manager.get_block_table(request.request_id)
            
            # ========================================================================
            # TP Support: Transfer all shards
            # ========================================================================
            # For TP>1: All workers in a stage share the same request
            # Transfer: Prefill worker i → Decode worker i (for each TP rank)
            # ========================================================================
            if self.tensor_parallel_size > 1:
                # Transfer all shards in parallel
                transfer_tasks = []
                for tp_rank in range(len(self.workers)):
                    src_rank = tp_rank  # Prefill worker with this TP rank
                    dst_rank = len(self.workers) + tp_rank  # Decode worker with same TP rank
                    
                    result = await self.kv_transfer_manager.transfer_kv_cache_async(
                        request_id=f"{request.request_id}_shard{tp_rank}",
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                        src_blocks=migrating_req.kv_block_indexes,
                        dst_blocks=dst_blocks,
                    )
                    transfer_tasks.append(result)
                
                # Use the first worker as the assigned worker (for compatibility)
                request.assigned_worker_rank = len(self.workers)
                # Store all transfer refs (need to wait for all)
                self.transfer_refs[request.request_id] = [t.get("transfer_ref") for t in transfer_tasks if t.get("success")]
                
                if all(t.get("success") for t in transfer_tasks):
                    request.kv_transfer_status = KVTransferStatus.TRANSFERRING
                    self.transfer_queue.append(request)
                    log_debug(f"🚀 [DecodeEngine] Started {len(transfer_tasks)} shard transfers for {request.request_id}")
                else:
                    request.kv_transfer_status = KVTransferStatus.FAILED
                    log_error(f"[DecodeEngine] Some shard transfers failed for {request.request_id}")
                
                return
            
            # TP=1: Original single-worker logic
            dst_worker = self.get_next_worker()
            if not dst_worker:
                log_warning(f"[DecodeEngine] No worker available for {request.request_id}")
                return
            
            src_rank = 0  # Prefill worker
            dst_rank = await dst_worker.get_global_rank.remote()
            request.assigned_worker_rank = dst_rank
            
            # ✅ Start async transfer - returns ObjectRef immediately
            result = await self.kv_transfer_manager.transfer_kv_cache_async(
                request_id=request.request_id,
                src_rank=src_rank,
                dst_rank=dst_rank,
                src_blocks=migrating_req.kv_block_indexes,
                dst_blocks=dst_blocks,
            )
            
            if result.get("success"):
                # ✅ Store ObjectRef for later waiting
                self.transfer_refs[request.request_id] = result["transfer_ref"]
                self.transfer_workers[request.request_id] = result["dst_worker"]
                self.transfer_start_times[request.request_id] = result["start_time"]
                self.transfer_metadata[request.request_id] = result
                
                request.kv_transfer_status = KVTransferStatus.TRANSFERRING
                self.transfer_queue.append(request)
                
                log_info(f"[DECODE] start_transfer: added {request.request_id} to transfer_queue, len={len(self.transfer_queue)}")
                log_debug(f"🚀 [DecodeEngine] Started NON-BLOCKING transfer for {request.request_id}")
            else:
                log_error(f"[DecodeEngine] Failed to start transfer for {request.request_id}: "
                         f"{result.get('error', 'Unknown')}")
                request.kv_transfer_status = KVTransferStatus.FAILED
        
        except Exception as e:
            log_error(f"[DecodeEngine] Exception starting transfer for {request.request_id}: {e}")
            import traceback
            traceback.print_exc()
            request.kv_transfer_status = KVTransferStatus.FAILED
    
    async def _wait_and_sync_transfer(self, request: Request):
        """
        ✅ TRUE SYNC POINT: Wait for transfer completion and sync CUDA stream
        
        This is where we actually wait for the transfer to complete.
        Called just before decode, maximizing overlap opportunity.
        
        TP Support: Waits for all shard transfers to complete
        """
        request_id = request.request_id
        
        # Check if transfer is pending
        if request_id not in self.transfer_refs:
            return  # Already completed or no transfer needed
        
        import ray
        import time
        
        transfer_ref = self.transfer_refs[request_id]
        
        # TP Support: transfer_ref may be a list (multiple shards)
        if isinstance(transfer_ref, list):
            # Wait for all shard transfers
            try:
                results = await asyncio.gather(*[ray.get(ref) for ref in transfer_ref])
                # Check if all succeeded
                if all(r.get("success", False) for r in results):
                    request.kv_transfer_status = KVTransferStatus.READY
                    log_debug(f"✅ [DecodeEngine] All {len(results)} shard transfers completed for {request_id}")
                else:
                    request.kv_transfer_status = KVTransferStatus.FAILED
                    log_error(f"[DecodeEngine] Some shard transfers failed for {request_id}")
            except Exception as e:
                log_error(f"[DecodeEngine] Error waiting for shard transfers: {e}")
                request.kv_transfer_status = KVTransferStatus.FAILED
            
            # Cleanup
            del self.transfer_refs[request_id]
            return
        
        # TP=1: Original single transfer logic
        
        transfer_ref = self.transfer_refs.pop(request_id)
        worker_ref = self.transfer_workers.pop(request_id)
        start_time = self.transfer_start_times.pop(request_id)
        metadata = self.transfer_metadata.pop(request_id)
        
        wait_start = time.perf_counter()
        log_info(f"[DecodeEngine] Wait transfer {request_id} pending={len(self.transfer_refs)}")
        
        try:
            # ✅ Wait for transfer to complete
            log_debug(f"⏳ [DecodeEngine] Waiting for transfer completion: {request_id}")
            transfer_result = await transfer_ref
            
            # ✅ Sync CUDA stream (critical!)
            sync_result = await worker_ref.sync_comm_stream.remote()
            
            wait_elapsed = time.perf_counter() - wait_start
            total_elapsed = time.perf_counter() - start_time
            sync_time = sync_result.get("elapsed", 0)
            
            if "error" not in transfer_result:
                # Success
                bytes_transferred = transfer_result.get("bytes", 0)
                
                # ✅ FIX: Use actual transfer time from worker, not total elapsed time
                # total_elapsed includes queuing time, we want actual transfer time
                actual_transfer_time = transfer_result.get("elapsed", 0)  # From worker
                if actual_transfer_time == 0:
                    actual_transfer_time = wait_elapsed  # Fallback to wait time
                
                bandwidth_gbps = (bytes_transferred / 1e9) / actual_transfer_time if actual_transfer_time > 0 else 0
                
                request.kv_transfer_time = actual_transfer_time  # Use actual transfer time
                request.kv_transfer_complete = True
                request.kv_transfer_status = KVTransferStatus.READY
                
                # Update stats
                self.transfer_stats["total_transfers"] += 1
                self.transfer_stats["successful_transfers"] += 1
                self.transfer_stats["total_transfer_time"] += actual_transfer_time  # Use actual time
                self.transfer_stats["total_wait_time"] += wait_elapsed
                
                log_info(f"✅ [DecodeEngine] Transfer complete for {request_id}: "
                        f"total={total_elapsed*1000:.1f}ms, "
                        f"wait={wait_elapsed*1000:.1f}ms, "
                        f"sync={sync_time*1000:.1f}ms, "
                        f"{bandwidth_gbps:.2f} GB/s")
            else:
                # Failed
                error = transfer_result.get("error", "Unknown error")
                log_error(f"❌ [DecodeEngine] Transfer failed for {request_id}: {error}")
                request.kv_transfer_status = KVTransferStatus.FAILED
                self.transfer_stats["failed_transfers"] += 1
        
        except Exception as e:
            log_error(f"❌ [DecodeEngine] Exception waiting for transfer {request_id}: {e}")
            import traceback
            traceback.print_exc()
            request.kv_transfer_status = KVTransferStatus.FAILED
            self.transfer_stats["failed_transfers"] += 1
    
    async def _poll_transferring_requests(self):
        """
        🔄 Poll transferring requests using non-blocking ray.wait()
        
        Check which transfers have completed without blocking.
        """
        if not self.transfer_refs:
            return
        
        import ray
        
        # Get all pending transfers
        pending_refs = list(self.transfer_refs.values())
        request_ids = list(self.transfer_refs.keys())
        
        log_debug(f"[DECODE] poll_transfer: checking {len(pending_refs)} pending transfers")
        
        # ✅ Non-blocking check: which transfers are ready?
        ready_refs, remaining_refs = ray.wait(
            pending_refs,
            num_returns=len(pending_refs),
            timeout=0  # Non-blocking!
        )
        
        log_info(f"[DECODE] poll_transfer: ready={len(ready_refs)} remaining={len(remaining_refs)} transfer_q={len(self.transfer_queue)}")
        
        if ready_refs:
            log_info(f"[DecodeEngine] Transfers ready={len(ready_refs)} pending={len(remaining_refs)}")
        
        # Mark completed transfers (but don't sync yet - that happens in _wait_and_sync_transfer)
        for ref in ready_refs:
            # Find corresponding request_id
            for i, req_id in enumerate(request_ids):
                if self.transfer_refs.get(req_id) == ref:
                    # Find the request object
                    for req in list(self.transfer_queue):
                        if req.request_id == req_id:
                            req.kv_transfer_status = KVTransferStatus.READY
                            self.ready_queue.append(req)
                            self.transfer_queue.remove(req)
                            log_info(f"[DecodeEngine] Transfer ready {req_id} tq={len(self.transfer_queue)} rq={len(self.ready_queue)}")
                            break
                    break
    
    async def _receive_from_prefill(self):
        """Receive requests from prefill stage"""
        received_count = 0
        while not self.prefill_decode_bridge_queue.empty():
            try:
                migrating_req = await asyncio.wait_for(
                    self.prefill_decode_bridge_queue.get(),
                    timeout=0.001
                )
                received_count += 1
                
                request = migrating_req.req
                
                # Backpressure: limit waiting queue size
                if len(self.scheduler.waiting_queue) >= self.max_batch_size:
                    await self.prefill_decode_bridge_queue.put(migrating_req)
                    break
                
                # Restore expanded prompt tokens
                if migrating_req.expanded_prompt_token_ids:
                    request.prompt_token_ids = migrating_req.expanded_prompt_token_ids
                
                # Restore output tokens
                if migrating_req.output_token_ids:
                    request.output_token_ids = migrating_req.output_token_ids
                    log_debug(f"[DecodeEngine] Restored output tokens for {request.request_id}: {len(request.output_token_ids)} tokens")
                else:
                    log_warning(f"[DecodeEngine] No output_token_ids in migrating_req for {request.request_id}")
                
                # Admission: only count active (running + ready), waiting should not block
                active = self._get_active_count()
                max_active = self.max_batch_size
                if active >= max_active:
                    log_debug(f"[DecodeEngine] Active cap reached ({active}/{max_active}), deferring {request.request_id}")
                    await self.prefill_decode_bridge_queue.put(migrating_req)
                    break
                
                # Transfer KV cache from prefill to decode
                if self.kv_transfer_manager and migrating_req.kv_block_indexes:
                    # Check blocks BEFORE allocating
                    blocks_needed = len(migrating_req.kv_block_indexes)
                    avail_blocks = self.block_manager.get_num_avail_gpu_blocks()
                    
                    if avail_blocks < blocks_needed:
                        # Not enough blocks, park into mem-wait (non-active)
                        log_debug(f"[DecodeEngine] Blocks insufficient for {request.request_id}: need {blocks_needed}, avail {avail_blocks} -> mem-wait")
                        self._mark_mem_wait(request, migrating_req)
                        continue
                    
                    # Safe to allocate now
                    self.block_manager.allocate_blocks(request, num_blocks=blocks_needed)
                    dst_blocks = self.block_manager.get_block_table(request.request_id)
                    
                    # Transfer KV cache (handle TP>1 in SIMULATION mode)
                    from kvserve.engine.kv_transfer import TransferMethod
                    is_simulation = (self.kv_transfer_manager and 
                                   self.kv_transfer_manager.transfer_method == TransferMethod.SIMULATION)
                    
                    # For TP>1 + SIMULATION: Load KV to ALL workers (each gets its shard)
                    # For TP=1 or non-SIMULATION: Load to single worker
                    workers_to_load = self.workers if (is_simulation and self.tensor_parallel_size > 1) else [self.get_next_worker()]
                    
                    transfer_start = time.time()
                    try:
                        all_success = True
                        for worker_idx, dst_worker in enumerate(workers_to_load):
                            if not dst_worker:
                                all_success = False
                                continue
                            
                            # Get ranks
                            src_rank = worker_idx if is_simulation else 0  # Prefill worker with same TP rank
                            dst_rank = await dst_worker.get_global_rank.remote()
                            
                            # For TP>1 + SIMULATION, use shard-specific request_id
                            req_id_for_file = f"{request.request_id}_tp{worker_idx}" if (is_simulation and self.tensor_parallel_size > 1) else request.request_id
                            
                            # Transfer KV cache
                            success = await self.kv_transfer_manager.transfer_kv_cache(
                                request_id=req_id_for_file,
                                src_rank=src_rank,
                                dst_rank=dst_rank,
                                src_blocks=migrating_req.kv_block_indexes,
                                dst_blocks=dst_blocks,
                            )
                            
                            if not success:
                                log_warning(f"[Decode] KV shard {worker_idx} transfer failed for {request.request_id}")
                                all_success = False
                        
                        transfer_time = time.time() - transfer_start
                        
                        if all_success:
                            request.kv_transfer_time = transfer_time
                            log_debug(f"[Decode] KV cache transferred for {request.request_id} in {transfer_time*1000:.2f}ms")
                        else:
                            log_warning(f"[Decode] KV cache transfer failed for {request.request_id}")
                            await self._cleanup_request(request, reason="transfer_failed")
                            continue
                    except Exception as e:
                        log_error(f"[Decode] KV cache transfer error: {e}")
                        import traceback
                        traceback.print_exc()
                        await self._cleanup_request(request, reason="transfer_exception")
                        continue
                
                # Add to scheduler
                self.scheduler.add_request(request)
                request.decoding_start_time = time.time()
                log_debug(f"[DecodeEngine] Added request {request.request_id} to scheduler (waiting={self.scheduler.num_waiting_requests()}, running={self.scheduler.num_running_requests()})")
                
            except asyncio.TimeoutError:
                break
        
        # ✅ Periodic diagnostic (every 5 seconds)
        if not hasattr(self, '_last_diagnostic_time'):
            self._last_diagnostic_time = 0
        
        if time.time() - self._last_diagnostic_time > 5:
            queue_size = self.prefill_decode_bridge_queue.qsize()
            num_waiting = self.scheduler.num_waiting_requests()
            num_running = self.scheduler.num_running_requests()
            avail_blocks = self.block_manager.get_num_avail_gpu_blocks()
            
            if queue_size > 0 or num_waiting > 0 or num_running > 0:
                self._last_diagnostic_time = time.time()
    
    async def step(self):
        """Execute one decode step"""
        if self.enable_multi_stream:
            # Multi-stream optimized path
            await self._step_multistream()
        else:
            # Original single-stream path
            await self._step_singlestream()
    
    async def _step_singlestream(self):
        """Original single-stream decode step"""
        await self._drain_mem_wait_queue(multistream=False)
        # Receive requests from prefill
        await self._receive_from_prefill()
        
        # Schedule a batch
        batched_requests = self.scheduler.schedule(max_batch_size=self.max_batch_size)
        
        if not batched_requests:
            await asyncio.sleep(0.01)
            return
        
        # ✅ Force print to diagnose decode execution
        if not hasattr(self, '_decode_step_count'):
            self._decode_step_count = 0
        self._decode_step_count += 1
        
        log_debug(f"[DecodeEngine] Scheduling batch of {len(batched_requests)} requests")
        
        # ========================================================================
        # DYNAMIC BLOCK EXPANSION (from ElasticMM)
        # Expand blocks based on actual sequence length (prompt + output tokens)
        # ========================================================================
        requests_that_can_run = []
        requests_delayed = []
        
        for request in batched_requests.requests:
            if request.request_id not in self.block_manager.block_table:
                log_warning(f"[DecodeEngine] {request.request_id} has no blocks allocated!")
                continue
            
            # Calculate blocks needed
            prompt_tokens = len(request.prompt_token_ids) if request.prompt_token_ids else 0
            output_tokens = len(request.output_token_ids) if request.output_token_ids else 0
            total_seq_len = prompt_tokens + output_tokens
            
            # Only allocate what is needed to avoid premature block exhaustion
            blocks_needed = (total_seq_len + self.block_manager.block_size - 1) // self.block_manager.block_size
            current_blocks = len(self.block_manager.block_table[request.request_id])
            
            # Check if expansion is needed
            if current_blocks < blocks_needed:
                additional_blocks_needed = blocks_needed - current_blocks
                avail_blocks = self.block_manager.get_num_avail_gpu_blocks()
                
                # ✅ Diagnostic: print first few expansion attempts
                if not hasattr(self, '_expansion_checks'):
                    self._expansion_checks = 0
                self._expansion_checks += 1
                
                if avail_blocks < additional_blocks_needed:
                    # Move to mem-wait and free compute capacity
                    log_debug(f"[DecodeEngine] {request.request_id} waiting for {additional_blocks_needed} blocks (avail={avail_blocks}) -> mem-wait")
                    self.scheduler.running_requests.pop(request.request_id, None)
                    self._mark_mem_wait(request)
                    requests_delayed.append(request.request_id)
                    continue
                
                # Allocate additional blocks
                new_blocks = self.block_manager._get_free_blocks(additional_blocks_needed, BlockLocation.GPU)
                self.block_manager.block_table[request.request_id].extend(new_blocks)
                log_debug(f"[DecodeEngine] Expanded blocks for {request.request_id}: "
                      f"{current_blocks} -> {blocks_needed} blocks")
            
            # This request can run
            requests_that_can_run.append(request)
        
        # Update batch to only include requests that can run
        if requests_delayed:
            log_debug(f"[DecodeEngine] Memory pressure: Delayed {len(requests_delayed)} requests due to insufficient blocks")
            # Only print detailed info periodically to avoid spam
            if not hasattr(self, '_decode_memory_pressure_count'):
                self._decode_memory_pressure_count = 0
            self._decode_memory_pressure_count += 1
            if self._decode_memory_pressure_count % 50 == 1:
                log_warning(f"[DecodeEngine] ⚠️  Prolonged memory pressure: {len(requests_delayed)} requests delayed")
                self.block_manager.print_block_usage()
            
            batched_requests.requests = requests_that_can_run
        
        # If all requests were delayed, skip this step
        if not batched_requests.requests:
            log_warning("[DecodeEngine] ⚠️  All requests delayed due to memory pressure, waiting for memory to free up...")
            self.block_manager.print_block_usage()
            await asyncio.sleep(0.05)
            return
        
        # Get block tables (after expansion)
        kv_block_tables = {
            req.request_id: self.block_manager.get_block_table(req.request_id)
            for req in batched_requests.requests
        }
        
        # Execute decode
        step_start = time.time()
        try:
            # ⚠️ CRITICAL FOR TP>1: ALL workers must be called simultaneously
            if self.tensor_parallel_size > 1:
                # TP>1: Call ALL workers in parallel
                results = await asyncio.gather(*[
                    worker.step_decode.remote(batched_requests, kv_block_tables)
                    for worker in self.workers
                ])
                # All workers return same outputs (synchronized via NCCL)
                outputs = results[0]
            else:
                # TP=1: Single worker
                worker = self.get_next_worker()
                if not worker:
                    return
                outputs = await worker.step_decode.remote(
                    batched_requests,
                    kv_block_tables
                )
        except Exception as e:
            log_error(f"[DecodeEngine] Error in step_decode: {e}")
            import traceback
            traceback.print_exc()
            # Cleanup affected requests to avoid leaks
            for req in batched_requests.requests:
                await self._cleanup_request(req, reason="decode_exception")
            await asyncio.sleep(0.1)
            return
        step_end = time.time()
        # Compute-only duration for this batch
        batch_compute_time = None
        if outputs and getattr(outputs[0], "step_start_time", None) is not None and getattr(outputs[0], "step_end_time", None) is not None:
            batch_compute_time = outputs[0].step_end_time - outputs[0].step_start_time
        else:
            batch_compute_time = step_end - step_start
        self.total_decode_compute_time += batch_compute_time
        
        if not outputs:
            log_warning(f"[DecodeEngine] No outputs returned from worker")
            await asyncio.sleep(0.1)
            return
        
        # Update requests and handle outputs
        log_debug(f"[DecodeEngine] Received {len(outputs)} outputs from worker")
        for output in outputs:
            request = next((r for r in batched_requests.requests if r.request_id == output.request_id), None)
            if request:
                request.output_token_ids = output.output_token_ids
                # Per-request view: attribute the batch compute time to each member
                request.total_decode_compute_time += batch_compute_time
                
                log_debug(f"[DecodeEngine] Request {output.request_id}: tokens={len(output.output_token_ids)}, finished={output.finished}")
                
                # Always call output callback (even if not finished) for streaming
                if self.output_callback:
                    log_debug(f"[DecodeEngine] Calling output callback for {output.request_id} (finished={output.finished})")
                    await self.output_callback(output)
                else:
                    log_warning(f"[DecodeEngine] No output callback set!")
                
                if output.finished:
                    request.decoding_end_time = time.time()
                    await self._cleanup_request(request, reason=output.finish_reason or "finished")
    
    async def _step_multistream(self):
        """
        🚀 TRUE ASYNC Multi-stream decode step
        
        Key difference from single-stream:
        1. Transfers start immediately and return (non-blocking)
        2. Poll for completed transfers without blocking
        3. Only wait for transfer when we actually need to decode
        4. Sync CUDA stream just before compute
        """
        log_info(f"[DECODE] step_ms_start: transfer_refs={len(self.transfer_refs)} transfer_q={len(self.transfer_queue)} ready_q={len(self.ready_queue)}")
        
        # STEP 1: Receive new requests and start async transfers (non-blocking!)
        await self._receive_from_prefill_multistream()
        log_info(f"[DECODE] after_receive: transfer_refs={len(self.transfer_refs)} transfer_q={len(self.transfer_queue)} ready_q={len(self.ready_queue)}")
        
        # Try to re-activate mem-wait requests
        await self._drain_mem_wait_queue(multistream=True)
        
        # STEP 2: Poll for completed transfers (non-blocking check)
        await self._poll_transferring_requests()
        log_info(f"[DECODE] after_poll: transfer_refs={len(self.transfer_refs)} transfer_q={len(self.transfer_queue)} ready_q={len(self.ready_queue)}")
        
        # STEP 3: Add ready requests to scheduler
        await self._add_ready_to_scheduler()
        log_info(f"[DECODE] after_add_ready: waiting={len(self.scheduler.waiting_queue)} running={len(self.scheduler.running_requests)}")
        
        # STEP 4: Schedule a batch
        batched_requests = self.scheduler.schedule(max_batch_size=self.max_batch_size)
        
        if not batched_requests:
            await asyncio.sleep(0.01)
            return
        
        # STEP 5: ✅ SYNC POINT: Wait for transfers and sync CUDA streams
        # This is where we actually wait - just before compute!
        for request in batched_requests.requests:
            await self._wait_and_sync_transfer(request)
        
        # STEP 6: Execute compute (now KV cache is ready)
        await self._execute_compute_batch(batched_requests)
    
    async def _receive_from_prefill_multistream(self):
        """Receive requests and start async transfers (non-blocking)"""
        processed = 0
        while not self.prefill_decode_bridge_queue.empty():
            try:
                migrating_req = await asyncio.wait_for(
                    self.prefill_decode_bridge_queue.get(),
                    timeout=0.001
                )
                
                request = migrating_req.req
                
                # Backpressure: limit waiting queue size
                if len(self.scheduler.waiting_queue) >= self.max_batch_size:
                    await self.prefill_decode_bridge_queue.put(migrating_req)
                    break
                
                # Admission based on active (running + ready + transfer)
                active = self._get_active_count()
                max_active = self.max_batch_size
                if active >= max_active:
                    await self.prefill_decode_bridge_queue.put(migrating_req)
                    break
                
                # Check blocks BEFORE processing
                if migrating_req.kv_block_indexes:
                    blocks_needed = len(migrating_req.kv_block_indexes)
                    avail_blocks = self.block_manager.get_num_avail_gpu_blocks()
                    
                    if avail_blocks < blocks_needed:
                        self._mark_mem_wait(request, migrating_req)
                        continue
                
                # Restore tokens
                if migrating_req.expanded_prompt_token_ids:
                    request.prompt_token_ids = migrating_req.expanded_prompt_token_ids
                if migrating_req.output_token_ids:
                    request.output_token_ids = migrating_req.output_token_ids
                
                # ✅ Start async transfer (blocks are guaranteed available)
                await self._start_async_transfer(request, migrating_req)
                
            except asyncio.TimeoutError:
                break
    
    async def _add_ready_to_scheduler(self):
        """Add ready requests to scheduler"""
        if self.ready_queue:
            log_debug(f"[DECODE] add_ready: processing {len(self.ready_queue)} ready requests")
        
        processed = []
        
        for request in list(self.ready_queue):
            if request.kv_transfer_status == KVTransferStatus.READY:
                # Check memory
                prompt_tokens = len(request.prompt_token_ids) if request.prompt_token_ids else 0
                min_blocks_needed = (prompt_tokens + self.block_manager.block_size - 1) // self.block_manager.block_size
                avail_blocks = self.block_manager.get_num_avail_gpu_blocks()
                
                if avail_blocks < min_blocks_needed:
                    log_debug(f"[DecodeEngine] {request.request_id} waiting for memory "
                             f"(need {min_blocks_needed}, avail {avail_blocks}) -> mem-wait")
                    self._mark_mem_wait(request)
                    processed.append(request)  # remove from ready_queue to avoid active pressure
                    continue
                
                # Add to scheduler
                self.scheduler.add_request(request)
                request.decoding_start_time = time.time()
                processed.append(request)
                
                log_debug(f"[DecodeEngine] Added {request.request_id} to scheduler")
        
        # Remove processed
        for req in processed:
            self.ready_queue.remove(req)
    
    async def _execute_compute_batch(self, batched_requests):
        """Execute compute batch (shared by both single and multi-stream)"""
        from kvserve.engine.block_manager import BlockLocation
        
        log_debug(f"[DecodeEngine] Scheduling batch of {len(batched_requests)} requests")
        
        # Dynamic block expansion
        requests_that_can_run = []
        requests_delayed = []
        
        for request in batched_requests.requests:
            if request.request_id not in self.block_manager.block_table:
                log_warning(f"[DecodeEngine] {request.request_id} has no blocks allocated!")
                continue
            
            # Calculate blocks needed
            prompt_tokens = len(request.prompt_token_ids) if request.prompt_token_ids else 0
            output_tokens = len(request.output_token_ids) if request.output_token_ids else 0
            total_seq_len = prompt_tokens + output_tokens
            
            blocks_needed = ((total_seq_len + self.block_manager.block_size - 1) // self.block_manager.block_size) + 1
            current_blocks = len(self.block_manager.block_table[request.request_id])
            
            # Check if expansion needed
            if current_blocks < blocks_needed:
                additional_blocks_needed = blocks_needed - current_blocks
                avail_blocks = self.block_manager.get_num_avail_gpu_blocks()
                
                if avail_blocks < additional_blocks_needed:
                    # Move to mem-wait instead of cycling in scheduler
                    log_debug(f"[DecodeEngine] {request.request_id} needs {additional_blocks_needed} blocks (avail={avail_blocks}) -> mem-wait")
                    self.scheduler.running_requests.pop(request.request_id, None)
                    self._mark_mem_wait(request)
                    requests_delayed.append(request.request_id)
                    continue
                
                # Allocate additional blocks
                new_blocks = self.block_manager._get_free_blocks(additional_blocks_needed, BlockLocation.GPU)
                self.block_manager.block_table[request.request_id].extend(new_blocks)
                
                log_debug(f"[DecodeEngine] Expanded blocks for {request.request_id}: "
                         f"{current_blocks} -> {blocks_needed} blocks (+{additional_blocks_needed})")
            
            requests_that_can_run.append(request)
        
        # Handle delayed requests
        if requests_delayed:
            log_debug(f"[DecodeEngine] Memory pressure: Delayed {len(requests_delayed)} requests")
            # Periodic warning only
            if not hasattr(self, '_multistream_memory_pressure_count'):
                self._multistream_memory_pressure_count = 0
            self._multistream_memory_pressure_count += 1
            if self._multistream_memory_pressure_count % 50 == 1:
                log_warning(f"[DecodeEngine] ⚠️  Prolonged memory pressure (multi-stream): {len(requests_delayed)} requests delayed")
                self.block_manager.print_block_usage()
            
            batched_requests.requests = requests_that_can_run
        
        if not batched_requests.requests:
            log_warning("[DecodeEngine] ⚠️  All requests delayed due to memory pressure")
            await asyncio.sleep(0.05)
            return
        
        # Get block tables
        kv_block_tables = {
            req.request_id: self.block_manager.get_block_table(req.request_id)
            for req in batched_requests.requests
        }
        
        # Execute decode
        step_start = time.time()
        try:
            # ⚠️ CRITICAL FOR TP>1: ALL workers must be called simultaneously
            if self.tensor_parallel_size > 1:
                # TP>1: Call ALL workers in parallel
                results = await asyncio.gather(*[
                    worker.step_decode.remote(batched_requests, kv_block_tables)
                    for worker in self.workers
                ])
                # All workers return same outputs (synchronized via NCCL)
                outputs = results[0]
            else:
                # TP=1: Single worker
                worker = self.get_next_worker()
                if not worker:
                    return
            outputs = await worker.step_decode.remote(batched_requests, kv_block_tables)
        except Exception as e:
            log_error(f"[DecodeEngine] Error in step_decode: {e}")
            for req in batched_requests.requests:
                await self._cleanup_request(req, reason="decode_exception")
            await asyncio.sleep(0.1)
            return
        step_end = time.time()
        # Compute-only duration for this batch
        batch_compute_time = None
        if outputs and getattr(outputs[0], "step_start_time", None) is not None and getattr(outputs[0], "step_end_time", None) is not None:
            batch_compute_time = outputs[0].step_end_time - outputs[0].step_start_time
        else:
            batch_compute_time = step_end - step_start
        self.total_decode_compute_time += batch_compute_time
        
        if not outputs:
            log_warning(f"[DecodeEngine] No outputs returned from worker")
            await asyncio.sleep(0.1)
            return
        
        # Handle outputs
        log_debug(f"[DecodeEngine] Received {len(outputs)} outputs from worker")
        for output in outputs:
            request = next((r for r in batched_requests.requests if r.request_id == output.request_id), None)
            if request:
                request.output_token_ids = output.output_token_ids
                request.total_decode_compute_time += batch_compute_time
                
                log_debug(f"[DecodeEngine] Request {output.request_id}: tokens={len(output.output_token_ids)}, finished={output.finished}")
                
                # Call output callback
                if self.output_callback:
                    log_debug(f"[DecodeEngine] Calling output callback for {output.request_id} (finished={output.finished})")
                    await self.output_callback(output)
                else:
                    log_warning(f"[DecodeEngine] No output callback set!")
                
                # Finish if done
                if output.finished:
                    request.decoding_end_time = time.time()
                    await self._cleanup_request(request, reason=output.finish_reason or "finished")
    
    def get_multistream_stats(self) -> dict:
        """
        Get multi-stream statistics (TRUE ASYNC version)
        """
        if not self.enable_multi_stream:
            return {"enabled": False}
        
        # Calculate average times
        avg_transfer_time = 0.0
        avg_wait_time = 0.0
        if self.transfer_stats["successful_transfers"] > 0:
            avg_transfer_time = (self.transfer_stats["total_transfer_time"] / 
                               self.transfer_stats["successful_transfers"])
            avg_wait_time = (self.transfer_stats["total_wait_time"] / 
                           self.transfer_stats["successful_transfers"])
        
        # Overlap ratio: how much time we saved by overlapping
        overlap_ratio = 0.0
        if avg_transfer_time > 0:
            overlap_ratio = max(0.0, (avg_transfer_time - avg_wait_time) / avg_transfer_time)
        
        return {
            "enabled": True,
            "mode": "TRUE_ASYNC",  # Indicate this is true async mode
            
            # Queue states
            "transfer_queue_size": len(self.transfer_queue),
            "ready_queue_size": len(self.ready_queue),
            "pending_transfers": len(self.transfer_refs),  # Transfers not yet complete
            
            # Transfer statistics
            "total_transfers": self.transfer_stats["total_transfers"],
            "successful_transfers": self.transfer_stats["successful_transfers"],
            "failed_transfers": self.transfer_stats["failed_transfers"],
            
            # Timing analysis
            "avg_transfer_time_ms": avg_transfer_time * 1000,
            "avg_wait_time_ms": avg_wait_time * 1000,  # Time spent waiting at sync point
            "avg_overlap_time_ms": (avg_transfer_time - avg_wait_time) * 1000,
            "overlap_ratio": overlap_ratio,  # How much we overlapped (0=none, 1=perfect)
            
            # Scheduler state
            "scheduler_waiting": self.scheduler.num_waiting_requests(),
            "scheduler_running": self.scheduler.num_running_requests(),
        }

    def get_decode_compute_ms(self) -> float:
        """Return accumulated decode compute time (ms) without scheduling/IO."""
        return self.total_decode_compute_time * 1000.0

    def reset_decode_compute_time(self):
        """Reset accumulated decode compute time."""
        self.total_decode_compute_time = 0.0

