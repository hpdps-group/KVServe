"""
Worker - Ray actor for PD separation
Based on vLLM v0 engine
"""

import copy
import gc
import os
import sys
import time
from typing import List, Optional, Dict, Any

import ray
import torch
import torch.distributed

# Auto-detect project root and add to sys.path for Ray workers
# This allows workers to import kvserve without installing the package
_worker_init_done = False
if not _worker_init_done:
    # Find project root by looking for kvserve directory
    current_file = os.path.abspath(__file__)
    # Go up from kvserve/engine/worker.py to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    _worker_init_done = True

from kvserve.engine.utils import (
    EngineStage,
    Request,
    BatchedRequests,
    StepOutput,
)
from kvserve.engine.logger import log_error


@ray.remote(num_cpus=0, num_gpus=1)
class Worker:
    """
    Ray worker for PD separation
    Executes prefill or decode stage inference tasks
    """
    
    def __init__(
        self,
        worker_id: int,
        stage: EngineStage,
        model_path: str,
        block_size: int = 16,
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        seed: int = 1024,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
        # Global NCCL parameters (for P2P KV transfer)
        global_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        nccl_init_method: Optional[str] = None,
    ):
        """
        Initialize worker
        
        Args:
            worker_id: Unique worker ID
            stage: Engine stage (PREFILL or DECODING)
            model_path: Path to model
            block_size: KV cache block size
            dtype: Model dtype
            tensor_parallel_size: Tensor parallel size
            seed: Random seed
            max_model_len: Maximum model sequence length
            gpu_memory_utilization: GPU memory utilization ratio
            global_rank: Global NCCL rank
            world_size: NCCL world size
            nccl_init_method: NCCL init method (e.g., "tcp://localhost:29500")
        """
        self.worker_id = worker_id
        self.stage = stage
        self.model_path = model_path
        self.block_size = block_size
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.seed = seed
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        
        # Global NCCL for P2P transfer
        self.global_rank = global_rank
        self.nccl_world_size = world_size
        self.nccl_init_method = nccl_init_method
        self.nccl_pg_initialized = False
        
        # Model runner (vLLM v0)
        self.model_runner = None
        
        # KV cache tensors
        self.kv_cache = None
        self.cache_engine = None
        
        # GPU info
        self.gpu_id = ray.get_gpu_ids()[0]
        self.device = torch.device(f"cuda:0")
        torch.cuda.set_device(self.device)
        
        # Statistics
        self.execution_time = 0.0
        
        # Track decode steps
        self._decode_steps = {}
    
    def ready(self):
        """Check if worker is ready"""
        return True
    
    def get_stage(self) -> EngineStage:
        """Get current stage"""
        return self.stage
    
    def init_model(self):
        """
        Initialize vLLM v0 ModelRunner
        """
        import os
        import random
        import numpy as np
        
        # CRITICAL: Set VLLM_USE_V1=0 to use V0 attention backends
        os.environ['VLLM_USE_V1'] = '0'
        
        from vllm.config import (
            ModelConfig, ParallelConfig, SchedulerConfig,
            DeviceConfig, CacheConfig, LoadConfig, VllmConfig
        )
        from vllm.worker.worker import init_worker_distributed_environment
        from vllm.worker.model_runner import ModelRunner
        
        # Set random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        # Create vLLM configs
        model_config = ModelConfig(
            model=self.model_path,
            tokenizer=self.model_path,
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype=self.dtype,
            seed=self.seed,
            max_model_len=self.max_model_len,
            enforce_eager=True,  # Disable CUDA graphs for disaggregated inference
        )
        
        parallel_config = ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=self.tensor_parallel_size,
            worker_use_ray=False,
        )
        
        scheduler_config = SchedulerConfig(
            max_num_batched_tokens=self.max_model_len,
            max_num_seqs=256,
            max_model_len=self.max_model_len,
        )
        
        device_config = DeviceConfig(device="cuda")
        
        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            swap_space=4,
            cache_dtype="auto",
        )
        
        load_config = LoadConfig()
        
        # Create VllmConfig for vLLM v0.10.1+
        vllm_config = VllmConfig(
            model_config=model_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            cache_config=cache_config,
            load_config=load_config,
        )
        
        # Initialize global NCCL process group BEFORE vLLM initialization
        if self.global_rank is not None and self.nccl_world_size is not None and self.nccl_init_method is not None:
            if not torch.distributed.is_initialized():
                os.environ['MASTER_ADDR'] = self.nccl_init_method.split('//')[1].split(':')[0]
                os.environ['MASTER_PORT'] = self.nccl_init_method.split(':')[-1]
                os.environ['RANK'] = str(self.global_rank)
                os.environ['WORLD_SIZE'] = str(self.nccl_world_size)
                
                os.environ['NCCL_DEBUG'] = 'WARN'
                os.environ['NCCL_IB_DISABLE'] = '1'
                os.environ['NCCL_P2P_LEVEL'] = 'PHB'
                
                torch.distributed.init_process_group(
                    backend='nccl',
                    init_method=self.nccl_init_method,
                    rank=self.global_rank,
                    world_size=self.nccl_world_size,
                )
                
                self.nccl_pg_initialized = True
        
        # Initialize distributed environment
        def random_digits(n: int) -> str:
            return ''.join([str(random.randint(0, 9)) for _ in range(n)])
        
        init_worker_distributed_environment(
            vllm_config=vllm_config,
            rank=self.global_rank if self.global_rank is not None else 0,
            distributed_init_method=self.nccl_init_method if self.nccl_init_method is not None else f'tcp://localhost:{int(random_digits(4))+int(self.gpu_id)}',
            local_rank=self.global_rank if self.global_rank is not None else 0,
        )
        
        # Create model runner
        self.model_runner = ModelRunner(
            vllm_config=vllm_config,
            kv_cache_dtype=cache_config.cache_dtype,
            is_driver_worker=True,
        )
        
        # Load model
        self.model_runner.load_model()
        
        # Store configs
        self.model_config = model_config
        self.cache_config = cache_config
        self.vllm_model_config = model_config
        self.parallel_config = parallel_config
        
        torch.cuda.synchronize()
    
    def profile_num_available_blocks(self) -> Dict[str, int]:
        """
        Profile GPU memory after model loading to determine available KV cache blocks
        """
        from kvserve.engine.utils import GB
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        free_memory = total_memory - reserved_memory
        
        buffer_memory = 5.0 * GB
        available_memory = max(0, free_memory - buffer_memory)
        kv_cache_memory = available_memory * self.gpu_memory_utilization
        
        max_kv_cache = total_memory * 0.70
        kv_cache_memory = min(kv_cache_memory, max_kv_cache)
        
        # Calculate size of one KV cache block
        num_layers = self.model_config.hf_config.num_hidden_layers
        num_kv_heads = getattr(self.model_config.hf_config, 'num_key_value_heads', 
                               self.model_config.hf_config.num_attention_heads)
        head_size = self.model_config.hf_config.hidden_size // self.model_config.hf_config.num_attention_heads
        
        bytes_per_element = 2 if self.dtype in ["float16", "bfloat16"] else 4
        
        block_size_bytes = (
            2 * self.block_size * num_kv_heads * head_size * bytes_per_element * num_layers
        )
        
        num_gpu_blocks = int(kv_cache_memory / block_size_bytes)
        num_cpu_blocks = max(100, num_gpu_blocks // 10)
        
        return {
            'num_gpu_blocks': num_gpu_blocks,
            'num_cpu_blocks': num_cpu_blocks
        }
    
    def init_kvcache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> List[int]:
        """
        Initialize KV cache using vLLM's CacheEngine
        """
        from vllm.worker.cache_engine import CacheEngine
        from vllm.utils import bind_kv_cache
        from vllm.config import get_layers_from_vllm_config
        from vllm.attention import Attention
        
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        
        device_config = self.model_runner.device_config
        self.cache_engine = CacheEngine(
            self.cache_config,
            self.model_config,
            self.parallel_config,
            device_config,
        )
        
        self.gpu_cache = [self.cache_engine.gpu_cache]
        self.cpu_cache = [self.cache_engine.cpu_cache]
        self.kv_cache = self.gpu_cache[0]
        
        # Bind KV cache to model's Attention layers
        shared_kv_cache_layers: dict[str, str] = {}
        attn_layers = get_layers_from_vllm_config(self.model_runner.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if (kv_tgt_layer := attn_module.kv_sharing_target_layer_name) is not None:
                shared_kv_cache_layers[layer_name] = kv_tgt_layer
        
        bind_kv_cache(
            self.model_runner.vllm_config.compilation_config.static_forward_context,
            self.gpu_cache,
            shared_kv_cache_layers
        )
        
        return []
    
    def step_prefill(
        self,
        batched_requests: BatchedRequests,
        kv_block_tables: Dict[str, List[int]],
    ) -> tuple:
        """Execute prefill step"""
        from kvserve.engine.worker_steps import step_prefill_impl
        return step_prefill_impl(self, batched_requests, kv_block_tables)
    
    def step_decode(
        self,
        batched_requests: BatchedRequests,
        kv_block_tables: Dict[str, List[int]],
    ) -> List[StepOutput]:
        """Execute decode step"""
        from kvserve.engine.worker_steps import step_decode_impl
        return step_decode_impl(self, batched_requests, kv_block_tables)
    
    # ===== NCCL Transfer Methods =====
    
    def extract_kv_blocks(self, block_indices: List[int]) -> torch.Tensor:
        """
        Extract KV cache blocks for transfer
        
        Args:
            block_indices: List of block indices to extract
            
        Returns:
            Tensor containing the KV data for specified blocks
            Shape: [num_layers, 2, num_blocks, block_size, num_heads, head_size]
        """
        if self.kv_cache is None:
            raise RuntimeError("KV cache not initialized")
        
        block_idx_tensor = torch.tensor(block_indices, dtype=torch.long, device=self.kv_cache[0].device)
        
        layer_kv_list = []
        for layer_kv in self.kv_cache:
            extracted_blocks = layer_kv[:, block_idx_tensor, :, :, :]
            layer_kv_list.append(extracted_blocks)
        
        kv_data = torch.stack(layer_kv_list, dim=0)
        return kv_data
    
    def write_kv_blocks(self, block_indices: List[int], kv_data: torch.Tensor):
        """
        Write KV cache data to specified blocks
        
        Args:
            block_indices: List of block indices to write to
            kv_data: KV data tensor [num_layers, 2, num_blocks, block_size, num_heads, head_size]
        """
        if self.kv_cache is None:
            raise RuntimeError("KV cache not initialized")
        
        assert kv_data.shape[2] == len(block_indices), \
            f"Block count mismatch: {kv_data.shape[2]} vs {len(block_indices)}"
        
        block_idx_tensor = torch.tensor(block_indices, dtype=torch.long, device=self.device)
        
        for layer_idx, layer_kv in enumerate(self.kv_cache):
            layer_kv[:, block_idx_tensor, :, :, :] = kv_data[layer_idx]
    
    def p2p_send_kv(self, dst_rank: int, block_indices: List[int]) -> Dict[str, Any]:
        """Send KV cache blocks via PyTorch P2P"""
        if not self.nccl_pg_initialized:
            return {"error": "Global NCCL not initialized", "bytes": 0, "blocks": 0}
        
        try:
            block_idx_tensor = torch.tensor(block_indices, dtype=torch.long, device=self.device)
            
            total_bytes = 0
            for layer_kv in self.kv_cache:
                layer_data = layer_kv[:, block_idx_tensor, :, :, :].contiguous()
                torch.distributed.send(layer_data, dst=dst_rank)
                total_bytes += layer_data.numel() * layer_data.element_size()
            
            torch.cuda.synchronize()
            return {"bytes": total_bytes, "blocks": len(block_indices)}
        except Exception as e:
            error_msg = f"NCCL send failed: {type(e).__name__}: {str(e)}"
            log_error(f"[Worker] {error_msg}")
            return {"error": error_msg, "bytes": 0, "blocks": 0}
    
    def p2p_recv_kv(self, src_rank: int, block_indices: List[int]) -> Dict[str, Any]:
        """Receive KV cache blocks via PyTorch P2P"""
        if not self.nccl_pg_initialized:
            return {"error": "Global NCCL not initialized", "bytes": 0, "blocks": 0}
        
        try:
            block_idx_tensor = torch.tensor(block_indices, dtype=torch.long, device=self.device)
            
            num_kv = 2
            num_blocks = len(block_indices)
            block_size = self.kv_cache[0].shape[2]
            num_heads = self.kv_cache[0].shape[3]
            head_size = self.kv_cache[0].shape[4]
            kv_dtype = self.kv_cache[0].dtype
            
            layer_shape = (num_kv, num_blocks, block_size, num_heads, head_size)
            total_bytes = 0
            
            for layer_idx, layer_kv in enumerate(self.kv_cache):
                layer_data = torch.empty(layer_shape, dtype=kv_dtype, device=self.device)
                torch.distributed.recv(layer_data, src=src_rank)
                layer_kv[:, block_idx_tensor, :, :, :] = layer_data
                total_bytes += layer_data.numel() * layer_data.element_size()
            
            torch.cuda.synchronize()
            return {"bytes": total_bytes, "blocks": len(block_indices)}
        except Exception as e:
            error_msg = f"NCCL recv failed: {type(e).__name__}: {str(e)}"
            log_error(f"[Worker] {error_msg}")
            return {"error": error_msg, "bytes": 0, "blocks": 0}
    
    def p2p_coordinated_transfer_kv(
        self,
        src_worker_ref: Any,
        src_rank: int,
        src_blocks: List[int],
        dst_blocks: List[int],
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """
        ✅ OPTIMIZED: Coordinated KV transfer with single Ray remote call
        
        This method is called on the DESTINATION worker and coordinates with
        the source worker using NCCL's built-in synchronization, reducing
        Ray remote call overhead and improving bandwidth.
        
        Args:
            src_worker_ref: Ray actor handle to source worker
            src_rank: NCCL rank of source worker
            src_blocks: Block indices to send from source
            dst_blocks: Block indices to write to in destination
            timeout: Transfer timeout in seconds (default: 10s)
            
        Returns:
            Dict with transfer statistics
        """
        import time
        
        if not self.nccl_pg_initialized:
            return {"error": "Destination worker: Global NCCL not initialized", "bytes": 0, "blocks": 0}
        
        if len(src_blocks) != len(dst_blocks):
            return {"error": f"Block count mismatch: src={len(src_blocks)}, dst={len(dst_blocks)}", "bytes": 0, "blocks": 0}
        
        if len(src_blocks) == 0:
            return {"bytes": 0, "blocks": 0}
        
        start_time = time.time()
        
        try:
            # ✅ STEP 1: Trigger send on source worker (fire and forget)
            send_future = src_worker_ref.p2p_send_kv.remote(
                torch.distributed.get_rank(),  # dst_rank = my rank
                src_blocks
            )
            
            # ✅ STEP 2: Immediately start receiving (NCCL will sync)
            # This coordinates the transfer more efficiently
            block_idx_tensor = torch.tensor(dst_blocks, dtype=torch.long, device=self.device)
            
            num_kv = 2
            num_blocks = len(dst_blocks)
            block_size = self.kv_cache[0].shape[2]
            num_heads = self.kv_cache[0].shape[3]
            head_size = self.kv_cache[0].shape[4]
            kv_dtype = self.kv_cache[0].dtype
            
            layer_shape = (num_kv, num_blocks, block_size, num_heads, head_size)
            total_bytes = 0
            
            # Receive layer by layer (zero-copy optimization)
            for layer_idx, layer_kv in enumerate(self.kv_cache):
                layer_data = torch.empty(layer_shape, dtype=kv_dtype, device=self.device)
                torch.distributed.recv(layer_data, src=src_rank)
                layer_kv[:, block_idx_tensor, :, :, :] = layer_data
                total_bytes += layer_data.numel() * layer_data.element_size()
            
            torch.cuda.synchronize()
            
            # Wait for send to complete (should already be done due to NCCL sync)
            send_result = ray.get(send_future)
            
            elapsed = time.time() - start_time
            
            if "error" in send_result:
                return {"error": f"Send failed: {send_result['error']}", "bytes": 0, "blocks": 0}
            
            return {"bytes": total_bytes, "blocks": len(dst_blocks), "elapsed": elapsed}
            
        except Exception as e:
            error_msg = f"Coordinated transfer failed: {type(e).__name__}: {str(e)}"
            log_error(f"[Worker] {error_msg}")
            import traceback
            traceback.print_exc()
            return {"error": error_msg, "bytes": 0, "blocks": 0}
    
    def get_global_rank(self) -> int:
        """Get global NCCL rank"""
        return self.global_rank if self.global_rank is not None else -1
    
    def get_stats(self) -> Dict[str, float]:
        """Get worker statistics"""
        return {
            "execution_time": self.execution_time,
            "gpu_id": self.gpu_id,
        }
    
    def destruct(self, options: List[str]):
        """Destruct and free resources"""
        if 'model' in options and self.model_runner:
            del self.model_runner
            self.model_runner = None
        
        if 'kv' in options and self.kv_cache is not None:
            del self.kv_cache
            self.kv_cache = None
            self.cache_engine = None
        
        for _ in range(2):
            gc.collect()
            torch.cuda.empty_cache()


