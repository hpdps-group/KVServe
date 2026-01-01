"""
Multi-stream optimization methods for DecodeEngine
Separated for clarity and maintainability
"""

import asyncio
import time
from typing import Dict, List
from collections import deque

from kvserve.engine.utils import Request, BatchedRequests, KVTransferStatus
from kvserve.engine.logger import log_debug, log_warning, log_error, log_info


async def receive_from_prefill_multistream(
    engine,
    prefill_decode_bridge_queue: asyncio.Queue,
    kv_transfer_manager,
    block_manager,
    scheduler,
    transfer_queue: deque,
    transferring: Dict,
    ready_queue: deque,
    transfer_stats: Dict,
):
    """
    🚀 MULTI-STREAM: Receive requests and start async transfer
    
    This method receives requests from prefill and immediately starts
    async transfer tasks without blocking.
    """
    while not prefill_decode_bridge_queue.empty():
        try:
            migrating_req = await asyncio.wait_for(
                prefill_decode_bridge_queue.get(),
                timeout=0.001
            )
            
            request = migrating_req.req
            
            # Restore tokens
            if migrating_req.expanded_prompt_token_ids:
                request.prompt_token_ids = migrating_req.expanded_prompt_token_ids
            if migrating_req.output_token_ids:
                request.output_token_ids = migrating_req.output_token_ids
            
            # ✅ Start async KV transfer (non-blocking)
            if kv_transfer_manager and migrating_req.kv_block_indexes:
                # Allocate blocks
                block_manager.allocate_blocks(
                    request,
                    num_blocks=len(migrating_req.kv_block_indexes)
                )
                dst_blocks = block_manager.get_block_table(request.request_id)
                
                # Get worker
                dst_worker = engine.get_next_worker()
                if not dst_worker:
                    log_warning(f"[DecodeEngine] No worker available for {request.request_id}")
                    continue
                
                src_rank = 0  # Prefill worker
                dst_rank = await dst_worker.get_global_rank.remote()
                request.assigned_worker_rank = dst_rank
                
                # Create async transfer task
                request.kv_transfer_status = KVTransferStatus.PENDING
                transfer_task = asyncio.create_task(
                    transfer_single_request(
                        request,
                        kv_transfer_manager,
                        src_rank,
                        dst_rank,
                        migrating_req.kv_block_indexes,
                        dst_blocks,
                        ready_queue,
                        transferring,
                        transfer_stats
                    )
                )
                
                request.kv_transfer_task = transfer_task
                transferring[request.request_id] = transfer_task
                
                # Add to transfer queue
                transfer_queue.append(request)
                log_debug(f"[DecodeEngine] Started async transfer for {request.request_id}")
            else:
                # No transfer needed
                request.kv_transfer_status = KVTransferStatus.READY
                ready_queue.append(request)
            
        except asyncio.TimeoutError:
            break


async def transfer_single_request(
    request: Request,
    kv_transfer_manager,
    src_rank: int,
    dst_rank: int,
    src_blocks: List[int],
    dst_blocks: List[int],
    ready_queue: deque,
    transferring: Dict,
    transfer_stats: Dict,
):
    """Async transfer task for single request"""
    try:
        request.kv_transfer_status = KVTransferStatus.TRANSFERRING
        
        transfer_start = time.time()
        result = await kv_transfer_manager.transfer_kv_cache_async(
            request_id=request.request_id,
            src_rank=src_rank,
            dst_rank=dst_rank,
            src_blocks=src_blocks,
            dst_blocks=dst_blocks,
        )
        transfer_time = time.time() - transfer_start
        
        if result.get("success"):
            request.kv_transfer_time = transfer_time
            request.kv_transfer_complete = result.get("transfer_complete", False)
            request.kv_transfer_status = KVTransferStatus.READY
            
            # Move to ready queue
            ready_queue.append(request)
            
            log_debug(f"[DecodeEngine] Transfer completed for {request.request_id} "
                     f"in {transfer_time*1000:.2f}ms")
            
            transfer_stats["successful_transfers"] += 1
        else:
            error = result.get("error", "Unknown error")
            request.kv_transfer_status = KVTransferStatus.FAILED
            log_error(f"[DecodeEngine] Transfer failed for {request.request_id}: {error}")
            
            transfer_stats["failed_transfers"] += 1
        
        transfer_stats["total_transfers"] += 1
        transfer_stats["total_transfer_time"] += transfer_time
        
    except Exception as e:
        request.kv_transfer_status = KVTransferStatus.FAILED
        log_error(f"[DecodeEngine] Transfer exception for {request.request_id}: {e}")
        transfer_stats["failed_transfers"] += 1
    finally:
        # Remove from transferring dict
        transferring.pop(request.request_id, None)


async def process_ready_queue(
    engine,
    ready_queue: deque,
    scheduler,
    block_manager,
):
    """
    Process requests that finished transfer, add to scheduler
    """
    processed = []
    
    for request in list(ready_queue):
        if request.kv_transfer_status == KVTransferStatus.READY:
            # Check memory
            prompt_tokens = len(request.prompt_token_ids) if request.prompt_token_ids else 0
            min_blocks_needed = (prompt_tokens + block_manager.block_size - 1) // block_manager.block_size
            avail_blocks = block_manager.get_num_avail_gpu_blocks()
            
            if avail_blocks < min_blocks_needed:
                log_debug(f"[DecodeEngine] {request.request_id} waiting for memory "
                         f"(need {min_blocks_needed}, avail {avail_blocks})")
                continue
            
            # Add to scheduler
            scheduler.add_request(request)
            request.decoding_start_time = time.time()
            processed.append(request)
            
            log_debug(f"[DecodeEngine] {request.request_id} added to scheduler")
    
    # Remove processed from ready_queue
    for req in processed:
        ready_queue.remove(req)


async def ensure_transfers_complete(batched_requests: BatchedRequests):
    """
    🔄 MULTI-STREAM: Validate transfers are ready
    
    NOTE: Synchronization is now handled automatically in worker.step_decode()
    via worker.sync_comm_stream(). This function just validates status.
    """
    for request in batched_requests.requests:
        if not request.kv_transfer_complete:
            log_debug(f"[DecodeEngine] {request.request_id} waiting for transfer completion "
                     f"(sync happens automatically in worker)")


def get_multistream_stats(
    transfer_queue: deque,
    transferring: Dict,
    ready_queue: deque,
    scheduler,
    transfer_stats: Dict,
) -> Dict:
    """Get multi-stream statistics"""
    return {
        "transfer_queue_size": len(transfer_queue),
        "transferring_count": len(transferring),
        "ready_queue_size": len(ready_queue),
        "scheduler_waiting": scheduler.num_waiting_requests(),
        "scheduler_running": scheduler.num_running_requests(),
        "transfer_stats": transfer_stats,
    }

