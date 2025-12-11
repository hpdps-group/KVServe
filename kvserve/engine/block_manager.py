"""
Block Manager for PD separation engine
Manages KV cache at block level
"""

from typing import List, Dict, Optional, Callable
from enum import Enum
<<<<<<< HEAD

from kvserve.engine.utils import Request, BatchedRequests

=======
import logging

from kvserve.engine.utils import Request, BatchedRequests

logger = logging.getLogger(__name__)

>>>>>>> c7601a7a1e297ef5ea04e70be674e52ba15e3d08

class BlockLocation(Enum):
    """The location of a block"""
    GPU = "gpu"
    CPU = "cpu"


class BlockManager:
    """
    Block Manager for KV cache
    Maintains key-value cache at block level with GPU/CPU swapping support
    """
    
    def __init__(
        self,
        stage: str,  # 'prefill' or 'decoding'
        max_num_gpu_blocks: int,
        max_num_cpu_blocks: int,
        block_size: int,
        engine_remote_call_all_workers_async: Optional[Callable] = None,
    ):
        """
        Args:
            stage: Engine stage name
            max_num_gpu_blocks: Maximum number of GPU blocks
            max_num_cpu_blocks: Maximum number of CPU blocks
            block_size: Number of tokens per block
            engine_remote_call_all_workers_async: Function to call workers asynchronously
        """
        self.stage = stage
        self.max_num_gpu_blocks = max_num_gpu_blocks
        self.max_num_cpu_blocks = max_num_cpu_blocks
        self.block_size = block_size
        self.engine_remote_call_all_workers_async = engine_remote_call_all_workers_async
        
        # Free block pools
        self.free_gpu_blocks_list = list(range(max_num_gpu_blocks))
        self.free_cpu_blocks_list = list(range(max_num_cpu_blocks))
        
        # Blocks currently being swapped
        self.swapping_gpu_blocks_list = []
        self.swapping_cpu_blocks_list = []
        
        # Block table: request_id => [block0_id, block1_id, ...]
        self.block_table: Dict[str, List[int]] = {}
        
        # Request location: request_id => BlockLocation
        self.request_location: Dict[str, BlockLocation] = {}
        
        # Statistics tracking
        self.swap_count = 0
        self.total_swap_time = 0.0
    
    def _reset_free_blocks(self):
        """
        Reset free block lists after dynamic resizing
        Called after profiling GPU memory to update block counts
        """
        # Clear and reinitialize free block lists with new sizes
        self.free_gpu_blocks_list = list(range(self.max_num_gpu_blocks))
        self.free_cpu_blocks_list = list(range(self.max_num_cpu_blocks))
        
        # Clear swapping lists (should be empty at init anyway)
        self.swapping_gpu_blocks_list = []
        self.swapping_cpu_blocks_list = []
        
        # Note: block_table and request_location are NOT cleared
        # as they may contain allocated blocks from before resizing
        # This method should only be called during initialization
    
    def get_num_avail_gpu_blocks(self) -> int:
        """Get the number of available GPU blocks"""
        return len(self.free_gpu_blocks_list) + len(self.swapping_gpu_blocks_list)
    
    def get_num_avail_cpu_blocks(self) -> int:
        """Get the number of available CPU blocks"""
        return len(self.free_cpu_blocks_list) + len(self.swapping_cpu_blocks_list)
    
    def _get_free_blocks(self, num_blocks: int, location: BlockLocation) -> List[int]:
        """
        Get free blocks from the free block pool indicated by `location`
        
        Args:
            num_blocks: Number of blocks to allocate
            location: GPU or CPU
            
        Returns:
            List of block IDs
        """
        if location == BlockLocation.GPU:
            num_avail_blocks = self.get_num_avail_gpu_blocks()
            assert num_avail_blocks >= num_blocks, \
                f"Not enough free blocks on GPU, requested {num_blocks}, available {num_avail_blocks}"
            
            if len(self.free_gpu_blocks_list) < num_blocks:
                # Wait for swap-out operations to complete
                if self.engine_remote_call_all_workers_async:
                    self.engine_remote_call_all_workers_async("wait_for_all_swap_out")
                self.free_gpu_blocks_list += self.swapping_gpu_blocks_list
                self.swapping_gpu_blocks_list = []
            
            blocks = self.free_gpu_blocks_list[:num_blocks]
            self.free_gpu_blocks_list = self.free_gpu_blocks_list[num_blocks:]
        else:
            num_avail_blocks = self.get_num_avail_cpu_blocks()
            assert num_avail_blocks >= num_blocks, \
                f"Not enough free blocks on CPU, requested {num_blocks}, available {num_avail_blocks}"
            
            if len(self.free_cpu_blocks_list) < num_blocks:
                # Wait for swap-in operations to complete
                if self.engine_remote_call_all_workers_async:
                    self.engine_remote_call_all_workers_async("wait_for_all_swap_in")
                self.free_cpu_blocks_list += self.swapping_cpu_blocks_list
                self.swapping_cpu_blocks_list = []
            
            blocks = self.free_cpu_blocks_list[:num_blocks]
            self.free_cpu_blocks_list = self.free_cpu_blocks_list[num_blocks:]
        
        return blocks
    
    def get_allocated_num_blocks(self, request_id: str) -> int:
        """Get the number of allocated blocks for a request"""
        return len(self.block_table.get(request_id, []))
    
    def get_location(self, request_id: str) -> Optional[BlockLocation]:
        """Get the KV cache blocks location of a request"""
        return self.request_location.get(request_id, None)
    
    def get_num_blocks_needed(self, request: Request) -> int:
        """
        Calculate the number of blocks needed for a request
        
        Args:
            request: Request object
            
        Returns:
            Number of blocks needed
        """
        total_len = request.get_total_len()
        return (total_len + self.block_size - 1) // self.block_size
    
    def get_num_append_blocks_needed(self, request: Request) -> int:
        """Get the number of additional blocks needed for a request already on GPU"""
        assert self.request_location.get(request.request_id) == BlockLocation.GPU, \
            f"Request {request.request_id} is not on GPU"
        
        num_blocks_cur = len(self.block_table[request.request_id])
        num_blocks_needed = self.get_num_blocks_needed(request)
        return max(0, num_blocks_needed - num_blocks_cur)
    
    def allocate_blocks(self, request: Request, num_blocks: Optional[int] = None):
        """
        Allocate blocks for a request
        
        Args:
            request: Request to allocate blocks for
            num_blocks: Optional explicit number of blocks to allocate. If None, calculated from request.
        """
        # Ensure request is not on CPU
        assert (request.request_id not in self.block_table
                or self.request_location.get(request.request_id) == BlockLocation.GPU), \
            f"Request {request.request_id} is on CPU. Migrate to GPU first."
        
        # Use explicit num_blocks if provided, otherwise calculate
        if num_blocks is None:
            num_blocks_needed = self.get_num_blocks_needed(request)
        else:
            num_blocks_needed = num_blocks
        
        if request.request_id not in self.block_table:
            # First time allocation
            self.block_table[request.request_id] = self._get_free_blocks(
                num_blocks_needed, BlockLocation.GPU
            )
            self.request_location[request.request_id] = BlockLocation.GPU
        else:
            # Request already has blocks, append if needed
            assert self.request_location[request.request_id] == BlockLocation.GPU
            num_blocks_cur = len(self.block_table[request.request_id])
            if num_blocks_cur < num_blocks_needed:
                additional_blocks = self._get_free_blocks(
                    num_blocks_needed - num_blocks_cur, BlockLocation.GPU
                )
                self.block_table[request.request_id] += additional_blocks
    
    def allocate_blocks_batched(self, batched_requests: BatchedRequests):
        """Allocate blocks for a batch of requests"""
        for request in batched_requests.requests:
            self.allocate_blocks(request)
    
    def free_blocks(self, request_id: str):
        """Free blocks for a request"""
        assert request_id in self.block_table, f"Request {request_id} not allocated"
        
        if self.request_location[request_id] == BlockLocation.GPU:
            self.free_gpu_blocks_list += self.block_table.pop(request_id)
        else:
            self.free_cpu_blocks_list += self.block_table.pop(request_id)
        
        self.request_location.pop(request_id)
    
    def free_blocks_batched(self, requests: List[Request]):
        """Free blocks for a batch of requests"""
        for request in requests:
            if request.request_id in self.block_table:
                self.free_blocks(request.request_id)
    
    def get_block_table(self, request_id: str) -> List[int]:
        """Get the block table for a request"""
        return self.block_table.get(request_id, [])
    
    def can_allocate(self, request: Request) -> bool:
        """Check if we can allocate blocks for a request"""
        num_blocks_needed = self.get_num_blocks_needed(request)
        return self.get_num_avail_gpu_blocks() >= num_blocks_needed
    
    def can_append(self, request: Request) -> bool:
        """Check if we can append blocks for a request"""
        if request.request_id not in self.block_table:
            return self.can_allocate(request)
        
        num_append_blocks = self.get_num_append_blocks_needed(request)
        return self.get_num_avail_gpu_blocks() >= num_append_blocks
    
    def append_blocks(self, request: Request):
        """
        Append additional blocks to an existing request (for autoregressive decode)
        
        Args:
            request: Request to append blocks for
        """
        assert request.request_id in self.block_table, \
            f"Request {request.request_id} not allocated yet"
        assert self.request_location.get(request.request_id) == BlockLocation.GPU, \
            f"Request {request.request_id} is not on GPU"
        
        num_append_blocks = self.get_num_append_blocks_needed(request)
        
        if num_append_blocks <= 0:
            return  # No need to append
        
        # Allocate new blocks
        new_blocks = self._get_free_blocks(num_append_blocks, BlockLocation.GPU)
        self.block_table[request.request_id].extend(new_blocks)
<<<<<<< HEAD
=======
    
    def get_block_usage(self) -> dict:
        """Get block usage statistics (from ElasticMM)"""
        num_cpu_blocks_used = (
            self.max_num_cpu_blocks - len(self.free_cpu_blocks_list) - len(self.swapping_cpu_blocks_list)
        )
        num_gpu_blocks_used = (
            self.max_num_gpu_blocks - len(self.free_gpu_blocks_list) - len(self.swapping_gpu_blocks_list)
        )
        
        safe_div = lambda n, d: n / d if d else 0
        
        return {
            'gpu': f'{round(safe_div(num_gpu_blocks_used, self.max_num_gpu_blocks)*100)}% ({num_gpu_blocks_used}/{self.max_num_gpu_blocks})',
            'cpu': f'{round(safe_div(num_cpu_blocks_used, self.max_num_cpu_blocks)*100)}% ({num_cpu_blocks_used}/{self.max_num_cpu_blocks})',
            'swap': f'{len(self.swapping_gpu_blocks_list)} -> {len(self.swapping_cpu_blocks_list)}',
            '#req': f'{len(self.block_table)}'
        }
    
    def print_block_usage(self):
        """Print block usage statistics (using debug level to avoid cluttering output)"""
        usage = self.get_block_usage()
        logger.debug(f"[{self.stage}] Block usage: GPU={usage['gpu']}, CPU={usage['cpu']}, "
                    f"Swapping={usage['swap']}, Requests={usage['#req']}")
>>>>>>> c7601a7a1e297ef5ea04e70be674e52ba15e3d08


