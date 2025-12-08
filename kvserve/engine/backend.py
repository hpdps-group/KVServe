"""
Backend for PD separation
Coordinates Prefill and Decode stages
"""

import asyncio
from typing import List, Dict, Optional
from collections import deque

from kvserve.engine.utils import Request, StepOutput
from kvserve.engine.stage_engine import PrefillEngine, DecodeEngine
from kvserve.engine.kv_transfer import KVTransferManager, TransferMethod
from kvserve.engine.logger import LogLevel, set_log_level, log_info


class PDBackend:
    """
    Backend for Prefill-Decode separation
    """
    
    def __init__(
        self,
        model_path: str,
        num_prefill_workers: int = 1,
        num_decoding_workers: int = 1,
        block_size: int = 16,
        max_num_gpu_blocks: int = 5000,
        max_num_cpu_blocks: int = 1000,
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        kv_transfer_method: str = "nccl",
        nccl_init_method: str = "tcp://localhost:29500",
        log_level: str = "WARNING",
        max_model_len: int = 32768,
        max_batch_size: int = 32,
    ):
        """
        Initialize PD Backend
        
        Args:
            model_path: Path to model
            num_prefill_workers: Number of prefill workers
            num_decoding_workers: Number of decoding workers
            block_size: KV cache block size
            max_num_gpu_blocks: Maximum GPU blocks
            max_num_cpu_blocks: Maximum CPU blocks
            dtype: Model dtype
            tensor_parallel_size: Tensor parallel size
            gpu_memory_utilization: GPU memory utilization
            kv_transfer_method: KV transfer method ('nccl' or 'p2p_copy')
            nccl_init_method: NCCL initialization method
            log_level: Log level ('ERROR', 'WARNING', 'INFO', 'DEBUG')
        """
        # Set log level
        try:
            level = LogLevel[log_level.upper()]
            set_log_level(level)
        except KeyError:
            # Invalid log level, use WARNING as default
            set_log_level(LogLevel.WARNING)
        
        self.model_path = model_path
        self.num_prefill_workers = num_prefill_workers
        self.num_decoding_workers = num_decoding_workers
        
        # Bridge queue for PD communication
        self.prefill_decode_bridge_queue = asyncio.Queue()
        
        # Output queue
        self.output_queue: deque[StepOutput] = deque()
        self._request_outputs: Dict[str, List[StepOutput]] = {}
        
        # KV transfer manager
        transfer_method = TransferMethod.NCCL if kv_transfer_method == "nccl" else TransferMethod.P2P_COPY
        self.kv_transfer_manager = KVTransferManager(transfer_method=transfer_method)
        
        # NCCL configuration
        self.nccl_init_method = nccl_init_method
        self.nccl_world_size = num_prefill_workers + num_decoding_workers
        
        # Engine configuration
        self.max_model_len = max_model_len
        self.max_batch_size = max_batch_size
        self.engine_config = {
            'model_path': model_path,
            'block_size': block_size,
            'max_num_gpu_blocks': max_num_gpu_blocks,
            'max_num_cpu_blocks': max_num_cpu_blocks,
            'dtype': dtype,
            'tensor_parallel_size': tensor_parallel_size,
            'gpu_memory_utilization': gpu_memory_utilization,
            'kv_transfer_manager': self.kv_transfer_manager,
            'nccl_init_method': nccl_init_method,
            'nccl_world_size': self.nccl_world_size,
            'max_model_len': max_model_len,
        }
        
        # Stage engines
        self.prefill_engine: Optional[PrefillEngine] = None
        self.decode_engine: Optional[DecodeEngine] = None
        
        # Event loop tasks
        self.engine_tasks = []
    
    async def _on_decode_output(self, output: StepOutput):
        """Callback for decode outputs"""
        self.output_queue.append(output)
        if output.request_id not in self._request_outputs:
            self._request_outputs[output.request_id] = []
        self._request_outputs[output.request_id].append(output)
    
    async def initialize(self):
        """Initialize all stage engines"""
        log_info("[PDBackend] Initializing engines...")
        
        # Create prefill engine
        self.prefill_engine = PrefillEngine(
            prefill_decode_bridge_queue=self.prefill_decode_bridge_queue,
            num_workers=self.num_prefill_workers,
            max_batch_size=self.max_batch_size,
            **self.engine_config
        )
        
        # Create decode engine
        self.decode_engine = DecodeEngine(
            prefill_decode_bridge_queue=self.prefill_decode_bridge_queue,
            num_workers=self.num_decoding_workers,
            max_batch_size=self.max_batch_size,
            **self.engine_config
        )
        
        # Set output callback
        self.decode_engine.set_output_callback(self._on_decode_output)
        
        # Initialize engines
        await asyncio.gather(
            self.prefill_engine.initialize(),
            self.decode_engine.initialize(),
        )
        
        log_info("[PDBackend] Engines initialized")
    
    async def start(self):
        """Start all stage engines"""
        if not all([self.prefill_engine, self.decode_engine]):
            raise RuntimeError("Engines not initialized. Call initialize() first.")
        
        log_info("[PDBackend] Starting event loops...")
        
        # Start event loops
        prefill_task = asyncio.create_task(self.prefill_engine.start_event_loop())
        decode_task = asyncio.create_task(self.decode_engine.start_event_loop())
        
        self.engine_tasks = [prefill_task, decode_task]
        
        await asyncio.sleep(0.1)  # Give tasks time to start
        
        log_info("[PDBackend] Event loops started")
    
    async def stop(self):
        """Stop all stage engines"""
        log_info("[PDBackend] Stopping engines...")
        
        # Stop event loops
        if self.prefill_engine:
            await self.prefill_engine.stop_event_loop()
        if self.decode_engine:
            await self.decode_engine.stop_event_loop()
        
        # Cancel tasks
        for task in self.engine_tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*self.engine_tasks, return_exceptions=True)
        
        log_info("[PDBackend] Engines stopped")
    
    async def add_request(self, request: Request):
        """Add a request to the prefill stage"""
        if not self.prefill_engine:
            raise RuntimeError("Backend not initialized")
        
        self.prefill_engine.scheduler.add_request(request)
        log_info(f"[PDBackend] Added request {request.request_id}")
    
    async def get_outputs(self) -> List[StepOutput]:
        """Get outputs from decode stage"""
        outputs = []
        while self.output_queue:
            outputs.append(self.output_queue.popleft())
        return outputs
    
    def get_stats(self) -> Dict:
        """Get backend statistics"""
        stats = {}
        
        if self.prefill_engine:
            stats['prefill'] = {
                'num_waiting': self.prefill_engine.scheduler.num_waiting_requests(),
                'num_running': self.prefill_engine.scheduler.num_running_requests(),
            }
        
        if self.decode_engine:
            stats['decoding'] = {
                'num_waiting': self.decode_engine.scheduler.num_waiting_requests(),
                'num_running': self.decode_engine.scheduler.num_running_requests(),
            }
        
        if self.kv_transfer_manager:
            stats['kv_transfer'] = self.kv_transfer_manager.get_stats()
        
        return stats


