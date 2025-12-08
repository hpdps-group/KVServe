"""
Stage Engines for PD separation
Implements Prefill and Decoding stages
"""

import asyncio
import time
from typing import List, Dict, Optional
from collections import deque

import ray

from kvserve.engine.utils import (
    EngineStage,
    EngineStatus,
    Request,
    BatchedRequests,
    MigratingRequest,
    StepOutput,
)
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
            if total_tokens + tokens > self.max_tokens_per_batch:
                break
            
            # Check block allocation
            if not self.can_allocate_request(request):
                break
            
            # Add to batch
            batch_requests.append(self.waiting_queue.popleft())
            self.running_requests[request.request_id] = request
            total_tokens += tokens
        
        if batch_requests:
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
        kv_transfer_manager: Optional[KVTransferManager] = None,
        nccl_init_method: Optional[str] = None,
        nccl_world_size: Optional[int] = None,
        max_model_len: int = 32768,
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
        self.kv_transfer_manager = kv_transfer_manager
        self.nccl_init_method = nccl_init_method
        self.nccl_world_size = nccl_world_size
        self.max_model_len = max_model_len
        
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
        self.scheduler = StageScheduler(stage, block_manager=self.block_manager)
        
        # Status
        self.status = EngineStatus.INACTIVE
        
        # Event loop control
        self.pls_stop_loop = asyncio.Event()
        self.is_loop_stopped = asyncio.Event()
        
        # Load balancing
        self.current_worker_index = 0
    
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
        
        # First, initialize all workers and load models
        for worker_id in range(self.num_workers):
            # Calculate global rank
            global_rank = worker_id if self.stage == EngineStage.PREFILL else worker_id + self.num_workers
            
            worker = Worker.remote(
                worker_id=worker_id,
                stage=self.stage,
                model_path=self.model_path,
                block_size=self.block_size,
                dtype=self.dtype,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                global_rank=global_rank,
                world_size=self.nccl_world_size,
                nccl_init_method=self.nccl_init_method,
            )
            
            self.workers.append(worker)
            
            # Initialize worker
            await worker.ready.remote()
            await worker.init_model.remote()
        
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
            global_rank = worker_id if self.stage == EngineStage.PREFILL else worker_id + self.num_workers
            
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
        
        while not self.pls_stop_loop.is_set():
            try:
                await self.step()
                await asyncio.sleep(0.001)  # Small delay to prevent busy waiting
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
        await asyncio.wait_for(self.is_loop_stopped.wait(), timeout=5.0)
    
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
        batched_requests = self.scheduler.schedule(max_batch_size=max_batch_size)
        
        if not batched_requests:
            await asyncio.sleep(0.01)
            return
        
        # Allocate blocks
        self.block_manager.allocate_blocks_batched(batched_requests)
        
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
        
        # Get worker
        worker = self.get_next_worker()
        if not worker:
            return
        
        # Execute prefill
        log_debug(f"[PrefillEngine] Executing prefill for {len(batched_requests.requests)} requests")
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
            
            log_debug(f"[PrefillEngine] Sending request {request.request_id} to decode (output_tokens={len(migrating_req.output_token_ids or [])})")
            await self.prefill_decode_bridge_queue.put(migrating_req)
        
        # Finish requests
        for req in batched_requests.requests:
            self.scheduler.finish_request(req.request_id)


class DecodeEngine(BaseStageEngine):
    """Decode stage engine"""
    
    def __init__(self, prefill_decode_bridge_queue: asyncio.Queue, max_batch_size: int = 32, **kwargs):
        super().__init__(stage=EngineStage.DECODING, **kwargs)
        self.prefill_decode_bridge_queue = prefill_decode_bridge_queue
        self.max_batch_size = max_batch_size
        self.output_callback = None
    
    def set_output_callback(self, callback):
        """Set callback for outputs"""
        self.output_callback = callback
    
    async def _receive_from_prefill(self):
        """Receive requests from prefill stage"""
        while not self.prefill_decode_bridge_queue.empty():
            try:
                migrating_req = await asyncio.wait_for(
                    self.prefill_decode_bridge_queue.get(),
                    timeout=0.001
                )
                
                request = migrating_req.req
                
                # Restore expanded prompt tokens
                if migrating_req.expanded_prompt_token_ids:
                    request.prompt_token_ids = migrating_req.expanded_prompt_token_ids
                
                # Restore output tokens
                if migrating_req.output_token_ids:
                    request.output_token_ids = migrating_req.output_token_ids
                    log_debug(f"[DecodeEngine] Restored output tokens for {request.request_id}: {len(request.output_token_ids)} tokens")
                else:
                    log_warning(f"[DecodeEngine] No output_token_ids in migrating_req for {request.request_id}")
                
                # Transfer KV cache from prefill to decode
                if self.kv_transfer_manager and migrating_req.kv_block_indexes:
                    # Allocate blocks in decode stage first
                    self.block_manager.allocate_blocks(request, num_blocks=len(migrating_req.kv_block_indexes))
                    dst_blocks = self.block_manager.get_block_table(request.request_id)
                    
                    # Get destination worker and rank
                    dst_worker = self.get_next_worker()
                    
                    if dst_worker:
                        # Get source and destination ranks
                        # Source rank: prefill worker (assuming worker 0 for simplicity)
                        src_rank = 0  # Prefill workers start from rank 0
                        
                        # Destination rank: decode worker
                        dst_rank = await dst_worker.get_global_rank.remote()
                        
                        # Transfer KV cache
                        transfer_start = time.time()
                        try:
                            success = await self.kv_transfer_manager.transfer_kv_cache(
                                request_id=request.request_id,
                                src_rank=src_rank,
                                dst_rank=dst_rank,
                                src_blocks=migrating_req.kv_block_indexes,
                                dst_blocks=dst_blocks,
                            )
                            transfer_time = time.time() - transfer_start
                            
                            if success:
                                request.kv_transfer_time = transfer_time
                                log_debug(f"[Decode] KV cache transferred for {request.request_id} in {transfer_time*1000:.2f}ms")
                            else:
                                log_warning(f"[Decode] KV cache transfer failed for {request.request_id}")
                        except Exception as e:
                            log_error(f"[Decode] KV cache transfer error: {e}")
                            import traceback
                            traceback.print_exc()
                
                # Add to scheduler
                self.scheduler.add_request(request)
                request.decoding_start_time = time.time()
                log_debug(f"[DecodeEngine] Added request {request.request_id} to scheduler (waiting={self.scheduler.num_waiting_requests()}, running={self.scheduler.num_running_requests()})")
                
            except asyncio.TimeoutError:
                break
    
    async def step(self):
        """Execute one decode step"""
        # Receive requests from prefill
        await self._receive_from_prefill()
        
        # Schedule a batch
        batched_requests = self.scheduler.schedule(max_batch_size=32)
        
        if not batched_requests:
            await asyncio.sleep(0.01)
            return
        
        log_debug(f"[DecodeEngine] Scheduling batch of {len(batched_requests)} requests")
        
        # ========================================================================
        # DYNAMIC BLOCK EXPANSION (from ElasticMM)
        # Expand blocks based on actual sequence length (prompt + output tokens)
        # ========================================================================
        requests_that_can_run = []
        requests_delayed = []
        
        for request in batched_requests.requests:
            if request.request_id not in self.block_manager.block_table:
                # First time seeing this request in decode, should have been allocated in _receive_from_prefill
                log_warning(f"[DecodeEngine] {request.request_id} has no blocks allocated!")
                continue
            
            # Calculate blocks needed based on current sequence length
            prompt_tokens = len(request.prompt_token_ids) if request.prompt_token_ids else 0
            output_tokens = len(request.output_token_ids) if request.output_token_ids else 0
            total_seq_len = prompt_tokens + output_tokens
            
            # Calculate blocks needed based on current sequence length
            # For decode, we need blocks for prompt + all output tokens generated so far
            # Add 1 block margin to reduce frequent re-allocation
            blocks_needed = ((total_seq_len + self.block_manager.block_size - 1) // self.block_manager.block_size) + 1
            
            current_blocks = len(self.block_manager.block_table[request.request_id])
            
            # Check if expansion is needed
            if current_blocks < blocks_needed:
                additional_blocks_needed = blocks_needed - current_blocks
                avail_blocks = self.block_manager.get_num_avail_gpu_blocks()
                
                log_debug(f"[DecodeEngine] Block expansion check for {request.request_id}: "
                      f"seq_len={total_seq_len} (prompt={prompt_tokens}+output={output_tokens}), "
                      f"blocks_needed={blocks_needed}, current={current_blocks}, "
                      f"need_add={additional_blocks_needed}, avail={avail_blocks}")
                
                # Check if we have enough blocks for expansion
                if avail_blocks < additional_blocks_needed:
                    # Not enough blocks, delay this request
                    requests_delayed.append(request.request_id)
                    log_debug(f"[DecodeEngine] Delayed {request.request_id} due to insufficient blocks")
                    continue
                
                # Allocate additional blocks directly (like ElasticMM)
                new_blocks = self.block_manager._get_free_blocks(additional_blocks_needed, BlockLocation.GPU)
                self.block_manager.block_table[request.request_id].extend(new_blocks)
                
                log_debug(f"[DecodeEngine] Expanded blocks for {request.request_id}: "
                      f"{current_blocks} -> {blocks_needed} blocks (+{additional_blocks_needed})")
            
            # This request can run in current iteration
            requests_that_can_run.append(request)
        
        # Update batch to only include requests that can run
        if requests_delayed:
            log_debug(f"[DecodeEngine] Delayed {len(requests_delayed)} requests due to memory, "
                  f"will retry next iteration")
            batched_requests.requests = requests_that_can_run
        
        # If all requests were delayed, skip this step
        if not batched_requests.requests:
            log_debug("[DecodeEngine] All requests delayed due to memory pressure, waiting...")
            await asyncio.sleep(0.05)
            return
        
        # Get block tables (after expansion)
        kv_block_tables = {
            req.request_id: self.block_manager.get_block_table(req.request_id)
            for req in batched_requests.requests
        }
        
        # Get worker
        worker = self.get_next_worker()
        if not worker:
            return
        
        # Execute decode
        step_start = time.time()
        try:
            outputs = await worker.step_decode.remote(
                batched_requests,
                kv_block_tables
            )
        except Exception as e:
            log_error(f"[DecodeEngine] Error in step_decode: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(0.1)
            return
        step_end = time.time()
        
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
                request.total_decode_compute_time += (step_end - step_start)
                
                log_debug(f"[DecodeEngine] Request {output.request_id}: tokens={len(output.output_token_ids)}, finished={output.finished}")
                
                # Always call output callback (even if not finished) for streaming
                if self.output_callback:
                    log_debug(f"[DecodeEngine] Calling output callback for {output.request_id} (finished={output.finished})")
                    await self.output_callback(output)
                else:
                    log_warning(f"[DecodeEngine] No output callback set!")
                
                if output.finished:
                    request.is_finished = True
                    request.finish_reason = output.finish_reason
                    request.decoding_end_time = time.time()
                    
                    # Free blocks (only if allocated)
                    if request.request_id in self.block_manager.block_table:
                        self.block_manager.free_blocks(request.request_id)
                    
                    # Finish request in scheduler
                    self.scheduler.finish_request(request.request_id)

