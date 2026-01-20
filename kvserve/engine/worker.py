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
from kvserve.engine.service_config import ServiceConfig
from kvserve.manager.compression_manager import (
    CompressedKVData,
    CompressionManager,
    CompressionConfig,
    EasyDist,
    get_default_compression_config,
)
from kvserve.engine.logger import log_error, log_info, log_debug, log_warning


@ray.remote(num_cpus=0, num_gpus=1)  # Default for TP=1; override with .options(num_gpus=N) for TP>1
class Worker:
    """
    Ray worker for PD separation
    Executes prefill or decode stage inference tasks
    
    Note: For tensor_parallel_size > 1, use Worker.options(num_gpus=N).remote(...)
    """
    
    def __init__(
        self,
        worker_id: int,
        stage: EngineStage,
        model_path: str,
        block_size: int = 16,
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        tp_rank: int = 0,  # NEW: Rank within TP group
        seed: int = 1024,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
        activation_memory_gb: Optional[float] = None,
        # Global NCCL parameters (for P2P KV transfer)
        global_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        nccl_init_method: Optional[str] = None,
        # KV compression parameters
        compression_config = None,  # Can be Dict, "default", None, or OnlineController instance
        service_config: Optional[ServiceConfig] = None,  # Service parameters for controller mode
    ):
        """
        Initialize worker
        
        Args:
            worker_id: Unique worker ID
            stage: Engine stage (PREFILL or DECODING)
            model_path: Path to model
            block_size: KV cache block size
            dtype: Model dtype
            tensor_parallel_size: Tensor parallel size (number of workers in TP group)
            tp_rank: Rank within TP group (0 to tensor_parallel_size-1)
            seed: Random seed
            max_model_len: Maximum model sequence length
            gpu_memory_utilization: GPU memory utilization ratio
            global_rank: Global NCCL rank
            world_size: NCCL world size
            nccl_init_method: NCCL init method (e.g., "tcp://localhost:29500")
            compression_config: KV compression configuration, supports three modes:
                - Dict: Custom compression config
                - None or "default": Use default compression config
                - OnlineController instance: Dynamic profile selection
            service_config: Service configuration for controller mode (bandwidth, SLO, etc.)
        """
        self.worker_id = worker_id
        self.stage = stage
        self.model_path = model_path
        self.block_size = block_size
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.tp_rank = tp_rank  # NEW: Store TP rank
        self.seed = seed
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.activation_memory_gb = activation_memory_gb
        
        # Global NCCL for P2P transfer
        self.global_rank = global_rank
        self.nccl_world_size = world_size
        self.nccl_init_method = nccl_init_method
        self.nccl_pg_initialized = False
        self.p2p_process_group = None  # Separate process group for P2P transfers
        
        # Model runner (vLLM v0)
        self.model_runner = None
        
        # KV cache tensors
        self.kv_cache = None
        self.cache_engine = None
        
        # KV compression
        self.compression_manager = None
        self.compression_config_input = compression_config  # Store original input
        self.online_controller = None  # Store controller if provided
        self.service_config = service_config or ServiceConfig()  # Store service config
        
        # Resolve compression config based on input type
        self.compression_config = self._resolve_compression_config(compression_config)
        
        if self.compression_config and self.compression_config.get("enabled", False):
            self._init_compression_manager()
        
        # GPU info
        # Ray manages GPU allocation through CUDA_VISIBLE_DEVICES
        # When Ray assigns a GPU, it sets CUDA_VISIBLE_DEVICES for that worker
        # So we always use cuda:0 in the worker process (the only visible GPU)
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        
        # CUDA Streams for multi-stream optimization
        self.compute_stream = torch.cuda.current_stream()  # Default stream for compute
        self.comm_stream = None  # Communication stream (initialized on demand)
        self.comm_event = None  # Event for stream synchronization (initialized with comm_stream)
        self.transfer_events: Dict[str, torch.cuda.Event] = {}  # {request_id: event}
        
        # Statistics
        self.execution_time = 0.0
        
        # Track decode steps
        self._decode_steps = {}
    
    def _resolve_compression_config(self, compression_config):
        """
        Resolve compression config based on input type
        
        Args:
            compression_config: Can be Dict, "default", None, or OnlineController instance
            
        Returns:
            Resolved compression config dictionary
        """
        # Check if it's an OnlineController instance
        try:
            from kvserve.controller import OnlineController
            if isinstance(compression_config, OnlineController):
                self.online_controller = compression_config
                log_info(f"[Worker-{self.worker_id}] Using OnlineController for dynamic compression")
                # For controller mode, return a placeholder config
                # Actual config will be determined at runtime
                return {"enabled": True, "mode": "controller"}
        except ImportError:
            pass
        
        # Handle Dict (custom config)
        if isinstance(compression_config, dict):
            log_info(f"[Worker-{self.worker_id}] Using custom compression config")
            return compression_config
        
        # Handle "default" string or None
        if compression_config is None or compression_config == "default":
            log_info(f"[Worker-{self.worker_id}] Using default compression config")
            return get_default_compression_config()
        
        # Unknown type, use default
        log_warning(f"[Worker-{self.worker_id}] Unknown compression_config type: {type(compression_config)}, using default")
        return get_default_compression_config()
    
    def _init_compression_manager(self):
        """Initialize compression manager with configured pipeline"""
        try:
            # Skip initialization for controller mode (will be handled at runtime)
            if self.compression_config.get("mode") == "controller":
                log_info(f"[Worker-{self.worker_id}] Controller mode: compression manager will be initialized at runtime")
                return
            
            from kvserve.manager.compression_manager import CompressionManager, CompressionConfig
            from kvserve.transformer import KVServeTransformer
            from kvserve.quantizer import KVServeQuantizer
            from kvserve.codec import KVServeCodec
            
            # Create compression config
            config = CompressionConfig(
                enabled=self.compression_config.get("enabled", True),
                transformer_config=self.compression_config.get("transformer_config"),
                quantizer_config=self.compression_config.get("quantizer_config"),
                codec_config=self.compression_config.get("codec_config"),
                pipeline=self.compression_config.get("pipeline", []),
                min_compress_size=self.compression_config.get("min_compress_size", 0),
            )
            
            # Instantiate components based on pipeline
            transformer = KVServeTransformer if "transformer" in config.pipeline else None
            quantizer = KVServeQuantizer if "quantizer" in config.pipeline else None
            codec = KVServeCodec if "codec" in config.pipeline else None
            
            self.compression_manager = CompressionManager(
                config=config,
                transformer=transformer,
                quantizer=quantizer,
                codec=codec,
            )
            
            log_info(f"[Worker-{self.worker_id}] Compression manager initialized with pipeline: {config.pipeline}")
        except Exception as e:
            log_error(f"[Worker-{self.worker_id}] Failed to initialize compression manager: {e}")
            self.compression_manager = None
    
    def ready(self):
        """Check if worker is ready"""
        return True
    
    def compression_manager_is_enabled(self) -> bool:
        """Check if compression manager is enabled"""
        if self.compression_config and self.compression_config.get("mode") == "controller":
            # Controller mode is always "enabled" but actual decision is made at runtime
            return True
        return self.compression_manager is not None and self.compression_config and self.compression_config.get("enabled", False)
    
    def get_runtime_compression_config(
        self,
        kv_volume_bytes: float,
        bandwidth_mbps: Optional[float] = None,
        model_latency_ms: Optional[float] = None,
        slo_ms: Optional[float] = None,
        accuracy_requirement: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get runtime compression config (only for controller mode)
        
        Args:
            kv_volume_bytes: KV cache volume in bytes
            bandwidth_mbps: Network bandwidth in MB/s (uses service_config if None)
            model_latency_ms: Model computation latency in milliseconds
            slo_ms: SLO constraint in milliseconds (uses service_config if None)
            accuracy_requirement: Required accuracy (uses service_config if None)
            
        Returns:
            Compression config dictionary or None
        """
        if self.online_controller is None:
            # Not in controller mode, return current config
            return self.compression_config
        
        # Use service_config defaults if parameters not provided
        bandwidth_mbps = bandwidth_mbps or self.service_config.bandwidth_mbps
        slo_ms = slo_ms or self.service_config.slo_ms
        accuracy_requirement = accuracy_requirement or self.service_config.accuracy_requirement
        model_latency_ms = model_latency_ms or 100.0  # Default if not specified
        
        # Call controller to select profile
        profile, context = self.online_controller.select_profile(
            V_bytes=kv_volume_bytes,
            B_mbps=bandwidth_mbps,
            T_model_ms=model_latency_ms,
            T_SLO_ms=slo_ms,
            acc_req=accuracy_requirement
        )
        
        # Store context for later update
        self._last_controller_context = context
        
        if profile:
            log_debug(f"[Worker-{self.worker_id}] Controller selected profile: {profile.profile_id}")
            return profile.compression_config
        else:
            reason = context.get('reason', 'unknown')
            log_debug(f"[Worker-{self.worker_id}] Controller: no compression (reason: {reason})")
            return {"enabled": False}
    
    def update_service_config(self, **kwargs):
        """
        Update service configuration parameters
        
        Args:
            **kwargs: Service config parameters to update (bandwidth_mbps, slo_ms, etc.)
        """
        self.service_config.update(**kwargs)
        log_info(f"[Worker-{self.worker_id}] Service config updated: {kwargs}")
    
    def update_controller_stats(self, observed_latency_ms: float):
        """
        Update controller statistics with observed latency (only for controller mode)
        
        Args:
            observed_latency_ms: Observed end-to-end latency in milliseconds
        """
        if self.online_controller is None or not hasattr(self, '_last_controller_context'):
            return
        
        self.online_controller.update(self._last_controller_context, observed_latency_ms)
        log_debug(f"[Worker-{self.worker_id}] Updated controller with T_obs={observed_latency_ms:.2f}ms")
    
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
        
        # CRITICAL: Disable CustomAllreduce for TP in Ray environment
        # CustomAllreduce expects all GPUs visible, but Ray isolates GPUs per worker
        os.environ['VLLM_USE_CUSTOM_ALLREDUCE'] = '0'
        
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
            # CRITICAL: Disable custom allreduce in Ray TP (each worker sees 1 GPU)
            disable_custom_all_reduce=True,
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
        
        # ========================================================================
        # ✅ SOLUTION: Multiple Process Groups for TP + NCCL P2P
        # ========================================================================
        # Strategy:
        # 1. Let vLLM initialize the default distributed group (for TP)
        # 2. After vLLM init, create a separate process group for P2P transfers
        # 3. Use the P2P group explicitly in send/recv operations
        # ========================================================================
        
        # Initialize distributed environment for vLLM (this will be the default group)
        def random_digits(n: int) -> str:
            return ''.join([str(random.randint(0, 9)) for _ in range(n)])
        
        # Determine vLLM's distributed init method
        # For TP>1: All workers in the same TP group use the SAME port
        # Each TP group (Prefill vs Decode) uses different ports
        if self.tensor_parallel_size > 1:
            # Use stage-specific port for TP group
            # Prefill and Decode use different ports
            stage_offset = 0 if self.stage == EngineStage.PREFILL else 1
            base_port = int(self.nccl_init_method.split(':')[-1]) if self.nccl_init_method else 29500
            tp_port = base_port + 100 + stage_offset
            vllm_init_method = f'tcp://localhost:{tp_port}'
            log_info(f"[Worker-{self.worker_id}] TP group init: port={tp_port}, tp_rank={self.tp_rank}/{self.tensor_parallel_size}, stage={self.stage.value}")
        else:
            # TP=1: Use worker-specific port (no TP coordination needed)
            worker_unique_id = self.global_rank if self.global_rank is not None else self.worker_id
            vllm_init_method = f'tcp://localhost:{29600 + worker_unique_id}'
            log_info(f"[Worker-{self.worker_id}] Single-worker init: {vllm_init_method}")
        
        # Initialize global NCCL group for real KV transfer when TP=1.
        # vLLM defaults to world_size=1 for TP=1, which is too small for
        # prefill+decode P2P transfer. Pre-init a larger group if needed.
        if (self.nccl_world_size is not None
                and self.nccl_world_size > self.tensor_parallel_size
                and self.global_rank is not None
                and self.nccl_init_method is not None
                and not torch.distributed.is_initialized()):
            log_info(
                f"[Worker-{self.worker_id}] Pre-initializing NCCL group for P2P: "
                f"rank={self.global_rank}, world={self.nccl_world_size}, init={self.nccl_init_method}"
            )
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=self.nccl_init_method,
                rank=self.global_rank,
                world_size=self.nccl_world_size,
            )

        # Initialize vLLM's distributed environment
        # For TP>1: Each worker has a rank within the TP group
        log_info(f"[Worker-{self.worker_id}] Calling init_worker_distributed_environment...")
        log_info(f"[Worker-{self.worker_id}]   TP: rank={self.tp_rank}, size={vllm_config.parallel_config.tensor_parallel_size}")
        log_info(f"[Worker-{self.worker_id}]   init_method={vllm_init_method}")
        log_info(f"[Worker-{self.worker_id}]   local_rank=0 (Ray gives each worker 1 GPU as cuda:0)")
        
        init_worker_distributed_environment(
            vllm_config=vllm_config,
            rank=self.tp_rank,  # TP rank within the group
            distributed_init_method=vllm_init_method,
            local_rank=0,  # CRITICAL: Always 0 (Ray makes it cuda:0 via CUDA_VISIBLE_DEVICES)
        )
        log_info(f"[Worker-{self.worker_id}] ✓ vLLM distributed environment initialized")
        
        # Create model runner
        log_info(f"[Worker-{self.worker_id}] Creating ModelRunner...")
        self.model_runner = ModelRunner(
            vllm_config=vllm_config,
            kv_cache_dtype=cache_config.cache_dtype,
            is_driver_worker=True,
        )
        log_info(f"[Worker-{self.worker_id}] ✓ ModelRunner created")
        
        # Load model
        log_info(f"[Worker-{self.worker_id}] Loading model...")
        self.model_runner.load_model()
        log_info(f"[Worker-{self.worker_id}] ✓ Model loaded")
        
        # Store configs
        self.model_config = model_config
        self.cache_config = cache_config
        self.vllm_model_config = model_config
        self.parallel_config = parallel_config
        
        torch.cuda.synchronize()
        
        # ========================================================================
        # Initialize P2P Process Group (after vLLM init)
        # ========================================================================
        # Now that vLLM has initialized (and possibly created a TP group),
        # we can create a separate process group for P2P KV transfers
        # ========================================================================
        if self.global_rank is not None and self.nccl_world_size is not None and self.nccl_init_method is not None:
            self._init_p2p_process_group()
    
    def _init_p2p_process_group(self):
        """Initialize a separate process group for P2P KV transfers"""
        try:
            # Check if default group exists (from vLLM TP init)
            if torch.distributed.is_initialized():
                log_info(f"[Worker-{self.worker_id}] Default group already initialized (by vLLM TP)")
                # Create a new group for P2P transfers
                # All workers participate in this group
                all_ranks = list(range(self.nccl_world_size))
                self.p2p_process_group = torch.distributed.new_group(
                    ranks=all_ranks,
                    backend='nccl',
                )
                self.nccl_pg_initialized = True
                log_info(f"[Worker-{self.worker_id}] ✅ Created P2P process group (rank={self.global_rank}, world={self.nccl_world_size})")
            else:
                # No default group, initialize one for P2P
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
                # Default group is the P2P group
                self.p2p_process_group = torch.distributed.group.WORLD
                self.nccl_pg_initialized = True
                log_info(f"[Worker-{self.worker_id}] ✅ Initialized default group for P2P")
        except Exception as e:
            log_error(f"[Worker-{self.worker_id}] Failed to init P2P group: {e}")
            self.nccl_pg_initialized = False
    
    def profile_num_available_blocks(self) -> Dict[str, int]:
        """
        Profile GPU memory after model loading to determine available KV cache blocks
        
        ✅ CRITICAL FIX: Reserve memory for activation peaks
        
        The ROOT CAUSE of OOM:
        - Old strategy allocated KV cache based on *free memory* after model load
        - This didn't account for dynamic activation allocation during inference
        - Result: KV cache took ~12GB, leaving only ~1GB for activations (need ~5-6GB)
        
        New strategy:
        - Estimate peak activation memory needs (based on batch_size, max_seq_len)
        - Reserve this activation memory FIRST
        - Allocate remaining budget to KV cache
        - Formula: KV_cache = (TotalBudget - Model - EstimatedActivation)
        """
        from kvserve.engine.utils import GB
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        free_memory = total_memory - reserved_memory
        
        # Step 1: Determine activation reserve from two knobs
        #   A) gpu_memory_utilization -> activation_from_util = total * (1 - util)
        #   B) activation_memory_gb (explicit) -> activation_from_user
        activation_from_util = total_memory * (1 - self.gpu_memory_utilization)
        activation_from_user = self.activation_memory_gb * GB if self.activation_memory_gb is not None else None
        
        if activation_from_user is not None:
            activation_required = max(activation_from_util, activation_from_user)
            activation_source = "max(user, util)"
        else:
            activation_required = activation_from_util
            activation_source = "util"
        
        # Step 2: Budget left for (Model + KV)
        model_memory = reserved_memory
        budget_for_model_kv = max(0, total_memory - activation_required)
        
        # Step 3: KV cache = budget_for_model_kv - model
        kv_cache_memory = max(0, 0.9*(budget_for_model_kv - model_memory))
        
        # Safety cap: never exceed 50% of total memory for KV cache
        max_kv_cache = total_memory * 0.50
        kv_cache_memory = min(kv_cache_memory, max_kv_cache)
        
        # Calculate size of one KV cache block
        num_layers = self.model_config.hf_config.num_hidden_layers
        num_kv_heads = getattr(self.model_config.hf_config, 'num_key_value_heads', 
                               self.model_config.hf_config.num_attention_heads)
        head_size = self.model_config.hf_config.hidden_size // self.model_config.hf_config.num_attention_heads
        
        # For TP > 1, KV heads are sharded across devices
        num_kv_heads_per_shard = num_kv_heads // self.tensor_parallel_size
        
        bytes_per_element = 2 if self.dtype in ["float16", "bfloat16"] else 4
        
        block_size_bytes = (
            2 * self.block_size * num_kv_heads_per_shard * head_size * bytes_per_element * num_layers
        )
        
        num_gpu_blocks = int(kv_cache_memory / block_size_bytes)
        num_cpu_blocks = max(100, num_gpu_blocks // 10)
        
        # ✅ Detailed logging for debugging
        log_info(f"[Worker {self.worker_id}] GPU Memory Profile (TP={self.tensor_parallel_size}):")
        log_info(f"  Total: {total_memory / GB:.2f} GB")
        log_info(f"  Model + Overhead: {model_memory / GB:.2f} GB")
        log_info(f"  Free after model: {free_memory / GB:.2f} GB")
        log_info(f"  Activation Source: {activation_source}")
        log_info(f"  Activation Reserved (util): {activation_from_util / GB:.2f} GB")
        if activation_from_user is not None:
            log_info(f"  Activation User: {activation_from_user / GB:.2f} GB")
        log_info(f"  Activation Used: {activation_required / GB:.2f} GB")
        log_info(f"  Budget for Model+KV: {budget_for_model_kv / GB:.2f} GB")
        log_info(f"  KV Cache Allocation: {kv_cache_memory / GB:.2f} GB")
        log_info(f"  KV heads per shard: {num_kv_heads_per_shard} (total: {num_kv_heads})")
        log_info(f"  Block size: {block_size_bytes / 1024:.2f} KB")
        log_info(f"  → GPU blocks: {num_gpu_blocks}, CPU blocks: {num_cpu_blocks}")
        
        return {
            'num_gpu_blocks': num_gpu_blocks,
            'num_cpu_blocks': num_cpu_blocks
        }
    
    def init_comm_stream(self):
        """Initialize communication stream for multi-stream optimization"""
        if self.comm_stream is None:
            self.comm_stream = torch.cuda.Stream()
            self.comm_event = torch.cuda.Event()
            log_info(f"[Worker {self.worker_id}] Created communication stream and event for async KV transfer")
        return self.comm_stream
    
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
    
    def compute_logprobs(
        self,
        context_tokens: List[int],
        continuation_tokens: List[int],
        block_ids: List[int],
    ) -> tuple[float, bool]:
        """
        Compute log-likelihood of continuation given context.
        Uses vLLM's ModelRunner with prompt_logprobs parameter.
        
        Args:
            context_tokens: Token IDs for context
            continuation_tokens: Token IDs for continuation
            block_ids: Pre-allocated block IDs for this computation (from block_manager)
            
        Returns:
            (logprob, is_greedy) tuple
            - logprob: Sum of log probabilities of continuation tokens
            - is_greedy: Whether continuation matches greedy decoding
        """
        from vllm.sequence import SequenceData, SequenceGroupMetadata
        from vllm import SamplingParams
        
        # Combine context and continuation
        full_tokens = context_tokens + continuation_tokens
        ctx_len = len(context_tokens)
        cont_len = len(continuation_tokens)
        
        # Truncate if too long (max_length from model config)
        max_length = getattr(self.model_config.hf_config, 'max_position_embeddings', 2048)
        if len(full_tokens) > max_length:
            # Truncate from left, keep last max_length tokens
            full_tokens = full_tokens[-max_length:]
            # After truncation, update lengths
            if len(full_tokens) < cont_len:
                # Severe truncation: adjust continuation
                continuation_tokens = full_tokens[-cont_len:] if len(full_tokens) >= cont_len else full_tokens
                cont_len = len(continuation_tokens)
                ctx_len = len(full_tokens) - cont_len
            else:
                # Context was truncated, continuation remains the same
                ctx_len = len(full_tokens) - cont_len
        
        # Use provided block_ids (allocated by block_manager in evaluator)
        # Ensure we have enough blocks for the sequence
        num_blocks_needed = (len(full_tokens) + self.block_size - 1) // self.block_size
        if len(block_ids) < num_blocks_needed:
            # This shouldn't happen if block_manager allocated correctly
            raise ValueError(f"Not enough blocks provided: need {num_blocks_needed}, got {len(block_ids)}")
        
        # Use only the blocks we need
        temp_block_ids = block_ids[:num_blocks_needed]
        
        # Create SequenceData with full tokens (for prompt_logprobs, we need the full sequence)
        # Use None for output_token_ids (like normal prefill) to avoid sampler assertion errors
        seq_data = SequenceData.from_seqs(
            prompt_token_ids=full_tokens,
            output_token_ids=None,  # None for prefill stage, not empty list
        )
        
        # Create SamplingParams with prompt_logprobs=1 to get logprobs for prompt tokens
        sampling_params = SamplingParams(
            temperature=0.0,  # Greedy for logprobs
            prompt_logprobs=1,  # Request logprobs for prompt tokens
            max_tokens=1,
            detokenize=False,
        )
        
        # Create SequenceGroupMetadata
        seq_id = 0
        seq_group_metadata = SequenceGroupMetadata(
            request_id="logprob_compute",
            is_prompt=True,
            seq_data={seq_id: seq_data},
            sampling_params=sampling_params,
            block_tables={seq_id: temp_block_ids},  # Use pre-allocated blocks
            do_sample=True,  # Must be True for vLLM to generate token and compute logprobs
            pooling_params=None,
            token_chunk_size=len(full_tokens),
            lora_request=None,
            computed_block_nums=[],
            multi_modal_data=None,
            multi_modal_placeholders=None,
        )
        
        # Prepare model input
        finished_requests_ids = []
        model_input = self.model_runner.prepare_model_input(
            [seq_group_metadata],
            virtual_engine=0,
            finished_requests_ids=finished_requests_ids
        )
        
        # Execute model to get logprobs
        seq_outs = self.model_runner.execute_model(model_input, [], None)
        
        if not seq_outs or len(seq_outs) == 0 or len(seq_outs[0]) == 0:
            return 0.0, False
        
        # Extract prompt_logprobs from output
        output = seq_outs[0][0]  # First (and only) sequence group output
        prompt_logprobs = output.prompt_logprobs
        
        if prompt_logprobs is None:
            return 0.0, False
        
        # Helper to extract logprob value (handles vLLM's Logprob object)
        def coerce_logprob_to_num(logprob):
            return getattr(logprob, "logprob", logprob)
        
        # Process prompt_logprobs
        # prompt_logprobs is a list: [None, {token_id: logprob, ...}, ...]
        # The first entry is None (no previous tokens to condition on)
        # We need logprobs for continuation tokens (starting at ctx_len)
        continuation_logprobs_dicts = [
            {
                token: coerce_logprob_to_num(logprob)
                for token, logprob in logprob_dict.items()
            }
            if logprob_dict is not None
            else None
            for logprob_dict in prompt_logprobs
        ]
        
        # Calculate continuation logprobs
        # According to vLLM: prompt_logprobs[i] contains logprobs for predicting token at position i+1
        # So for continuation tokens at positions [ctx_len, ctx_len+1, ..., ctx_len+cont_len-1]
        # We need prompt_logprobs at positions [ctx_len-1, ctx_len, ..., ctx_len+cont_len-2]
        # 
        # Example: full_tokens = [a, b, c, d, e] where ctx_len=2, cont_len=3
        #   full_tokens[0:2] = [a, b] (context)
        #   full_tokens[2:5] = [c, d, e] (continuation)
        #   prompt_logprobs[1] predicts full_tokens[2] (c) - continuation[0]
        #   prompt_logprobs[2] predicts full_tokens[3] (d) - continuation[1]
        #   prompt_logprobs[3] predicts full_tokens[4] (e) - continuation[2]
        # So we use prompt_logprobs[ctx_len-1:ctx_len+cont_len-1] for continuation tokens
        # 
        # Note: If ctx_len == 0, we use prompt_logprobs[0:] (first token has no context)
        
        continuation_logprobs = 0.0
        continuation_tokens_slice = full_tokens[ctx_len:ctx_len+cont_len]
        
        for i, token in enumerate(continuation_tokens_slice):
            # prompt_logprobs[i] predicts token at position i+1
            # So for continuation token at position ctx_len + i, we need prompt_logprobs[ctx_len + i - 1]
            # But if ctx_len == 0, we use prompt_logprobs[i] (since first token is at position 0)
            logprob_idx = max(0, ctx_len - 1 + i) if ctx_len > 0 else i
            if logprob_idx < len(continuation_logprobs_dicts):
                logprob_dict = continuation_logprobs_dicts[logprob_idx]
                if logprob_dict is not None and token in logprob_dict:
                    continuation_logprobs += logprob_dict[token]
        
        # Determine if greedy
        is_greedy = True
        for i, token in enumerate(continuation_tokens_slice):
            logprob_idx = max(0, ctx_len - 1 + i) if ctx_len > 0 else i
            if logprob_idx < len(continuation_logprobs_dicts):
                logprob_dict = continuation_logprobs_dicts[logprob_idx]
                if logprob_dict:
                    top_token = max(logprob_dict, key=logprob_dict.get)
                    if top_token != token:
                        is_greedy = False
                        break
        
        return float(continuation_logprobs), is_greedy
    
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
        
        # Extract first layer to determine output shape and dtype
        first_layer_kv = self.kv_cache[0][:, block_idx_tensor, :, :, :]
        num_layers = len(self.kv_cache)
        output_shape = (num_layers,) + first_layer_kv.shape
        
        # Pre-allocate output tensor (avoids list accumulation + stack overhead)
        kv_data = torch.empty(output_shape, dtype=first_layer_kv.dtype, device=first_layer_kv.device)
        kv_data[0] = first_layer_kv
        
        # Fill remaining layers in-place
        for layer_idx in range(1, num_layers):
            kv_data[layer_idx] = self.kv_cache[layer_idx][:, block_idx_tensor, :, :, :]
        
        return kv_data
    
    def write_kv_blocks(self, block_indices: List[int], kv_data: torch.Tensor, layer_offset: int = 0):
        """
        Write KV cache data to specified blocks
        
        Args:
            block_indices: List of block indices to write to
            kv_data: KV data tensor [num_layers, 2, num_blocks, block_size, num_heads, head_size]
            layer_offset: Starting layer index (for chunked writes)
        """
        if self.kv_cache is None:
            raise RuntimeError("KV cache not initialized")
        
        assert kv_data.shape[2] == len(block_indices), \
            f"Block count mismatch: {kv_data.shape[2]} vs {len(block_indices)}"
        
        block_idx_tensor = torch.tensor(block_indices, dtype=torch.long, device=self.device)
        
        chunk_num_layers = kv_data.shape[0]
        for i in range(chunk_num_layers):
            layer_idx = layer_offset + i
            if layer_idx < len(self.kv_cache):
                self.kv_cache[layer_idx][:, block_idx_tensor, :, :, :] = kv_data[i]
    
    def extract_and_compress_kv(self, block_indices: List[int], request_id: str) -> Dict[str, Any]:
        """
        Extract KV blocks and optionally compress (for SIMULATION mode)
        Includes internal timing for compression
        
        Args:
            block_indices: List of block indices to extract
            request_id: Request ID for compression tracking
            
        Returns:
            Dict with compressed_kv/kv_data, bytes, compression_time_ms, extract_time_ms
        """
        import time
        
        # Extract KV
        t_extract_start = time.time()
        kv_data = self.extract_kv_blocks(block_indices)
        extract_time_ms = (time.time() - t_extract_start) * 1000.0
        
        # Compress if enabled
        if self.compression_manager:
            t_compress_start = time.time()
            # Create compression config (following remote API)
            config = CompressionConfig(
                enabled=self.compression_config.get("enabled", True),
                transformer_config=self.compression_config.get("transformer_config"),
                quantizer_config=self.compression_config.get("quantizer_config"),
                codec_config=self.compression_config.get("codec_config"),
                pipeline=self.compression_config.get("pipeline", []),
                min_compress_size=self.compression_config.get("min_compress_size", 0),
            )
            compressed = self.compression_manager.compress_all_layers(
                all_layers_data=kv_data,
                request_id=request_id,
                config=config,
                metadata={"block_indices": block_indices},
            )
            compression_time_ms = (time.time() - t_compress_start) * 1000.0
            
            if compressed:
                # Move compressed data to CPU to avoid GPU memory pressure in pickling/simulation
                if compressed.is_chunked:
                    # Chunked compression: move all chunks to CPU
                    for chunk in compressed.chunks:
                        if chunk.device.type == 'cuda':
                            chunk.data = chunk.cpu()
                else:
                    # Single tensor compression
                    if compressed.compressed_tensor is not None and compressed.compressed_tensor.device.type == 'cuda':
                        compressed.compressed_tensor = compressed.compressed_tensor.cpu()
                
                return {
                    'compressed_kv': compressed,
                    'bytes': compressed.compressed_size,
                    'original_bytes': compressed.original_size,
                    'compression_time_ms': compression_time_ms,
                    'extract_time_ms': extract_time_ms,
                }
            else:
                # Compression skipped - move to CPU before returning
                log_warning(f"[Worker-{self.worker_id}] Compression returned None for {request_id}")
                return {
                    'kv_data': kv_data.cpu(),  # Move to CPU
                    'bytes': kv_data.numel() * kv_data.element_size(),
                    'compression_time_ms': 0.0,
                    'extract_time_ms': extract_time_ms,
                }
        else:
            # No compression - move to CPU before returning
            return {
                'kv_data': kv_data.cpu(),  # Move to CPU
                'bytes': kv_data.numel() * kv_data.element_size(),
                'compression_time_ms': 0.0,
                'extract_time_ms': extract_time_ms,
            }
    
    def decompress_and_write_kv(self, compressed_data: Any, dst_blocks: List[int]) -> Dict[str, Any]:
        """
        Decompress and write KV blocks (for SIMULATION mode)
        Includes internal timing for decompression
        
        Args:
            compressed_data: CompressedKVData or raw kv_data tensor (on CPU)
            dst_blocks: Destination block indices
            
        Returns:
            Dict with decompression_time_ms, write_time_ms
        """
        import time
        import torch
        
        # Decompress if needed
        t_decompress_start = time.time()
        if self.compression_manager and hasattr(compressed_data, 'compressed_tensor'):
            # Ensure compressed tensor on GPU for decompression (handle both chunked and non-chunked)
            if compressed_data.is_chunked:
                # Move all chunks to GPU
                if compressed_data.chunks:
                    for chunk in compressed_data.chunks:
                        if chunk is not None and chunk.device.type == 'cpu':
                            chunk.data = chunk.to(self.device, non_blocking=True)
            else:
                if compressed_data.compressed_tensor is not None and compressed_data.compressed_tensor.device.type == 'cpu':
                    compressed_data.compressed_tensor = compressed_data.compressed_tensor.to(self.device, non_blocking=True)
            
            # Create compression config (following remote API)
            config = CompressionConfig(
                enabled=self.compression_config.get("enabled", True),
                transformer_config=self.compression_config.get("transformer_config"),
                quantizer_config=self.compression_config.get("quantizer_config"),
                codec_config=self.compression_config.get("codec_config"),
                pipeline=self.compression_config.get("pipeline", []),
                min_compress_size=self.compression_config.get("min_compress_size", 0),
            )
            kv_data = self.compression_manager.decompress_all_layers(compressed_data, config)
            decompression_time_ms = (time.time() - t_decompress_start) * 1000.0
        else:
            # Already decompressed - move to GPU if needed
            kv_data = compressed_data
            if kv_data.device.type == 'cpu':
                kv_data = kv_data.to(self.device, non_blocking=True)
            decompression_time_ms = 0.0
        
        # Write to KV cache (handle both single tensor and chunked list)
        t_write_start = time.time()
        
        if isinstance(kv_data, list):
            # Chunked decompression result: write chunk by chunk to save memory
            log_debug(f"[Worker] Writing {len(kv_data)} decompressed chunks to KV cache")
            current_layer_offset = 0
            for chunk_idx, chunk_tensor in enumerate(kv_data):
                chunk_num_layers = chunk_tensor.shape[0]
                self.write_kv_blocks(dst_blocks, chunk_tensor, layer_offset=current_layer_offset)
                current_layer_offset += chunk_num_layers
                # Release chunk immediately after writing
                del chunk_tensor
            log_debug(f"[Worker] Finished writing {current_layer_offset} total layers")
        else:
            # Single tensor: write all at once
            self.write_kv_blocks(dst_blocks, kv_data)
            del kv_data
        
        write_time_ms = (time.time() - t_write_start) * 1000.0
        
        # Final cleanup
        torch.cuda.empty_cache()
        
        return {
            'decompression_time_ms': decompression_time_ms,
            'write_time_ms': write_time_ms,
        }
    
    def p2p_send_kv(self, dst_rank: int, block_indices: List[int]) -> Dict[str, Any]:
        """
        Send KV cache blocks via PyTorch P2P
        
        TP Support: When tensor_parallel_size > 1, self.kv_cache contains sharded
        KV data (num_heads = total_heads / TP). The transfer automatically sends
        the sharded data, and destination worker will receive matching shards.
        """
        if not self.nccl_pg_initialized:
            return {"error": "Global NCCL not initialized", "bytes": 0, "blocks": 0}
        
        try:
            block_idx_tensor = torch.tensor(block_indices, dtype=torch.long, device=self.device)
            
            total_bytes = 0
            for layer_kv in self.kv_cache:
                layer_data = layer_kv[:, block_idx_tensor, :, :, :].contiguous()
                torch.distributed.send(layer_data, dst=dst_rank, group=self.p2p_process_group)
                total_bytes += layer_data.numel() * layer_data.element_size()
            
            torch.cuda.synchronize()
            return {"bytes": total_bytes, "blocks": len(block_indices)}
        except Exception as e:
            error_msg = f"NCCL send failed: {type(e).__name__}: {str(e)}"
            log_error(f"[Worker] {error_msg}")
            return {"error": error_msg, "bytes": 0, "blocks": 0}
    
    def p2p_recv_kv(self, src_rank: int, block_indices: List[int]) -> Dict[str, Any]:
        """
        Receive KV cache blocks via PyTorch P2P
        
        TP Support: Receives sharded KV data matching the sender's TP configuration.
        Ensure both sender and receiver use the same tensor_parallel_size.
        """
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
                torch.distributed.recv(layer_data, src=src_rank, group=self.p2p_process_group)
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
                torch.distributed.recv(layer_data, src=src_rank, group=self.p2p_process_group)
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
    
    def p2p_send_kv_compressed(
        self,
        dst_rank: int,
        block_indices: List[int],
        request_id: str,
    ) -> Dict[str, Any]:
        """
        Send KV cache blocks with compression (all layers together)
        
        Args:
            dst_rank: Destination NCCL rank
            block_indices: Block indices to send
            request_id: Request ID for tracking
            
        Returns:
            Dict with transfer statistics
        """
        if not self.nccl_pg_initialized:
            return {"error": "Global NCCL not initialized", "bytes": 0, "blocks": 0}
        
        if self.compression_manager is None:
            return {"error": "Compression manager not initialized", "bytes": 0, "blocks": 0}
        
        try:
            # Extract all layers KV data
            kv_data = self.extract_kv_blocks(block_indices)  # [num_layers, 2, num_blocks, ...]
            
            # TODO: Update compression config for every request
            config = CompressionConfig(
                enabled=self.compression_config.get("enabled", True),
                transformer_config=self.compression_config.get("transformer_config"),
                quantizer_config=self.compression_config.get("quantizer_config"),
                codec_config=self.compression_config.get("codec_config"),
                pipeline=self.compression_config.get("pipeline", []),
                min_compress_size=self.compression_config.get("min_compress_size", 0),
            )

            # Compress all layers together
            compressed_data = self.compression_manager.compress_all_layers(
                all_layers_data=kv_data,
                request_id=request_id,
                config=config,
                metadata={"block_indices": block_indices},
            )
            
            # Release original KV data after compression
            del kv_data
            
            if compressed_data is None:
                # Compression failed or disabled, fallback to uncompressed
                log_warning(f"[Worker-{self.worker_id}] Compression failed, fallback to uncompressed transfer")
                return self.p2p_send_kv(dst_rank, block_indices)
            
            original_size = compressed_data.original_size
            compressed_size = compressed_data.compressed_size
            
            # Prepare metadata for transfer
            if compressed_data.is_chunked:
                # Include chunk sizes + chunk metadata for receiver reconstruction
                chunk_sizes = [
                    chunk.numel() * chunk.element_size()
                    for chunk in (compressed_data.chunks or [])
                ]
                metadata = {
                    **compressed_data.metadata,
                    "compressed_size": compressed_size,
                    "chunk_sizes": chunk_sizes,
                    "chunk_metadata": compressed_data.chunk_metadata or [],
                }
            else:
                metadata = compressed_data.metadata

            metadata_tensor, metadata_size_tensor = EasyDist.pack_object(metadata)
            metadata_size = metadata_size_tensor.item()
            
            # Send metadata size first (int64)
            torch.distributed.send(metadata_size_tensor, dst=dst_rank)
            del metadata_size_tensor # Release immediately
            
            # Send metadata
            torch.distributed.send(metadata_tensor, dst=dst_rank)
            del metadata_tensor # Release immediately
            
            # Send compressed data
            if compressed_data.is_chunked:
                for chunk in compressed_data.chunks or []:
                    c_tensor = chunk
                    if c_tensor.device != self.device:
                        c_tensor = c_tensor.to(self.device)
                    torch.distributed.send(c_tensor, dst=dst_rank)
                    del c_tensor
            else:
                c_tensor = compressed_data.compressed_tensor
                if c_tensor is None:
                    raise RuntimeError("compressed_tensor is None for non-chunked data")
                if c_tensor.device != self.device:
                    c_tensor = c_tensor.to(self.device)
                torch.distributed.send(c_tensor, dst=dst_rank)
                del c_tensor
            del compressed_data
            gc.collect()
            torch.cuda.empty_cache()
            # torch.cuda.synchronize()
            
            total_bytes_sent = 8 + metadata_size + compressed_size
            compression_ratio = original_size / (compressed_size + metadata_size)
            
            log_info(f"[Worker-{self.worker_id}] Compressed send: {original_size} -> {compressed_size} bytes (ratio: {compression_ratio:.2f}x)")
            
            return {
                "bytes": total_bytes_sent,
                "blocks": len(block_indices),
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
            }
        
        except Exception as e:
            error_msg = f"Compressed NCCL send failed: {type(e).__name__}: {str(e)}"
            log_error(f"[Worker-{self.worker_id}] {error_msg}")
            import traceback
            traceback.print_exc()
            return {"error": error_msg, "bytes": 0, "blocks": 0}
    
    def p2p_recv_kv_compressed(
        self,
        src_rank: int,
        block_indices: List[int],
    ) -> Dict[str, Any]:
        """
        Receive KV cache blocks with decompression (all layers together)
        
        Args:
            src_rank: Source NCCL rank
            block_indices: Block indices to write to
            
        Returns:
            Dict with transfer statistics
        """
        if not self.nccl_pg_initialized:
            return {"error": "Global NCCL not initialized", "bytes": 0, "blocks": 0}
        
        if self.compression_manager is None:
            return {"error": "Compression manager not initialized", "bytes": 0, "blocks": 0}
        
        try:
            # Receive metadata size (int64)
            metadata_size_tensor = torch.empty(1, dtype=torch.int64, device=self.device)
            torch.distributed.recv(metadata_size_tensor, src=src_rank)
            metadata_size = metadata_size_tensor.item()
            
            # Receive metadata
            metadata_tensor = torch.empty(metadata_size, dtype=torch.uint8, device=self.device)
            torch.distributed.recv(metadata_tensor, src=src_rank)
            
            # Unpack metadata
            metadata = EasyDist.unpack_object(metadata_tensor)
            
            # Release tensor after use
            del metadata_tensor, metadata_size_tensor
            
            # Infer compressed size from metadata
            original_size = metadata.get("original_size")
            compressed_size = metadata.get("compressed_size")
            if compressed_size is None:
                error_msg = "compressed_size not in metadata"
                log_error(f"[Worker-{self.worker_id}] {error_msg}")
                return {"error": error_msg, "bytes": 0, "blocks": 0}

            # Receive compressed data (chunked or single)
            if metadata.get("is_chunked"):
                chunk_sizes = metadata.get("chunk_sizes", [])
                chunk_metadata = metadata.get("chunk_metadata", [])
                chunks = []
                for size in chunk_sizes:
                    chunk_tensor = torch.empty(size, dtype=torch.uint8, device=self.device)
                    torch.distributed.recv(chunk_tensor, src=src_rank)
                    chunks.append(chunk_tensor)

                num_layers = metadata.get("num_layers", len(self.kv_cache))
                compressed_data = CompressedKVData(
                    request_id=metadata.get("request_id", "unknown"),
                    layer_id=num_layers - 1,
                    compressed_tensor=None,
                    metadata=metadata,
                    original_size=metadata.get("original_size", 0),
                    compressed_size=compressed_size,
                    is_chunked=True,
                    chunks=chunks,
                    chunk_metadata=chunk_metadata,
                )
            else:
                compressed_tensor = torch.empty(compressed_size, dtype=torch.uint8, device=self.device)
                torch.distributed.recv(compressed_tensor, src=src_rank)

                # Reconstruct CompressedKVData
                num_layers = metadata.get("num_layers", len(self.kv_cache))
                compressed_data = CompressedKVData(
                    request_id=metadata.get("request_id", "unknown"),
                    layer_id=num_layers - 1,  # layer_end_id
                    compressed_tensor=compressed_tensor,
                    metadata=metadata,
                    original_size=metadata.get("original_size", 0),
                    compressed_size=compressed_size,
                )
                
                # Release local reference to tensor (it is held by compressed_data)
                del compressed_tensor
            
            # TODO: Update compression config for every request
            config = CompressionConfig(
                enabled=self.compression_config.get("enabled", True),
                transformer_config=self.compression_config.get("transformer_config"),
                quantizer_config=self.compression_config.get("quantizer_config"),
                codec_config=self.compression_config.get("codec_config"),
                pipeline=self.compression_config.get("pipeline", []),
                min_compress_size=self.compression_config.get("min_compress_size", 0),
            )

            # Decompress all layers
            kv_data = self.compression_manager.decompress_all_layers(
                compressed_data=compressed_data,
                config=config,
            )
            chunk_metadata = compressed_data.chunk_metadata or []
            
            if kv_data is None:
                error_msg = "Decompression failed"
                log_error(f"[Worker-{self.worker_id}] {error_msg}")
                return {"error": error_msg, "bytes": 0, "blocks": 0}
            
            # Write to KV cache
            if isinstance(kv_data, list):
                # Chunked decompression: write each chunk with proper layer offset
                for idx, chunk in enumerate(kv_data):
                    start_layer = 0
                    if idx < len(chunk_metadata):
                        start_layer = chunk_metadata[idx].get("start_layer_id", 0)
                    self.write_kv_blocks(block_indices, chunk, layer_offset=start_layer)
            else:
                self.write_kv_blocks(block_indices, kv_data)
            
            # Release compressed_data after use (metadata already copied)
            del compressed_data
            
            # Release kv_data after writing to cache
            del kv_data
            
            total_bytes_received = 8 + metadata_size + compressed_size
            compression_ratio = original_size / (compressed_size + metadata_size) if compressed_size > 0 else 1.0
            
            # [MEMORY FIX] metadata is a dict, should be garbage collected, but explicit del helps
            del metadata
            gc.collect()
            torch.cuda.empty_cache()
            # torch.cuda.synchronize()
            return {
                "bytes": total_bytes_received,
                "blocks": len(block_indices),
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
            }
        
        except Exception as e:
            error_msg = f"Compressed NCCL recv failed: {type(e).__name__}: {str(e)}"
            log_error(f"[Worker-{self.worker_id}] {error_msg}")
            import traceback
            traceback.print_exc()
            return {"error": error_msg, "bytes": 0, "blocks": 0}
    
    def p2p_coordinated_transfer_kv_compressed(
        self,
        src_worker_ref: Any,
        src_rank: int,
        src_blocks: List[int],
        dst_blocks: List[int],
        request_id: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Coordinated KV transfer with compression (all layers together)
        
        This method is called on the DESTINATION worker and coordinates with
        the source worker to transfer compressed KV cache.
        
        Args:
            src_worker_ref: Ray actor handle to source worker
            src_rank: NCCL rank of source worker
            src_blocks: Block indices to send from source
            dst_blocks: Block indices to write to in destination
            request_id: Request ID for tracking
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
            # ✅ STEP 1: Trigger compressed send on source worker
            send_future = src_worker_ref.p2p_send_kv_compressed.remote(
                torch.distributed.get_rank(),  # dst_rank = my rank
                src_blocks,
                request_id
            )
            
            # ✅ STEP 2: Immediately start receiving compressed data
            recv_result = self.p2p_recv_kv_compressed(src_rank, dst_blocks)
            
            if "error" in recv_result:
                return recv_result
            
            # Wait for send to complete
            send_result = ray.get(send_future)
            
            elapsed = time.time() - start_time
            
            if "error" in send_result:
                return {"error": f"Send failed: {send_result['error']}", "bytes": 0, "blocks": 0}
            
            return {
                **recv_result,
                "elapsed": elapsed,
            }
            
        except Exception as e:
            error_msg = f"Coordinated compressed transfer failed: {type(e).__name__}: {str(e)}"
            log_error(f"[Worker] {error_msg}")
            import traceback
            traceback.print_exc()
            return {"error": error_msg, "bytes": 0, "blocks": 0}

    # =========================================================================
    # 🚚 Ray-based KV transfer (CPU hop) for TP>1 when NCCL P2P is unavailable
    # =========================================================================
    
    def p2p_transfer_kv_async(
        self,
        src_worker_ref: Any,
        src_rank: int,
        src_blocks: List[int],
        dst_blocks: List[int],
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """
        🚀 MULTI-STREAM: Async KV transfer on communication stream
        
        This method performs transfer on a separate comm_stream, allowing
        compute operations to continue on the default stream concurrently.
        
        Args:
            src_worker_ref: Ray actor handle to source worker
            src_rank: NCCL rank of source worker
            src_blocks: Block indices to send from source
            dst_blocks: Block indices to write to in destination
            timeout: Transfer timeout in seconds
            
        Returns:
            Dict with transfer statistics and CUDA event for synchronization
        """
        import time
        
        if not self.nccl_pg_initialized:
            return {"error": "NCCL not initialized", "bytes": 0, "blocks": 0}
        
        if len(src_blocks) != len(dst_blocks):
            return {"error": f"Block count mismatch", "bytes": 0, "blocks": 0}
        
        if len(src_blocks) == 0:
            return {"bytes": 0, "blocks": 0, "event": None}
        
        # Initialize comm stream if not already done
        if self.comm_stream is None:
            self.init_comm_stream()
        
        start_time = time.time()
        
        try:
            # ✅ Execute transfer on communication stream
            with torch.cuda.stream(self.comm_stream):
                # STEP 1: Trigger send on source worker
                send_future = src_worker_ref.p2p_send_kv_on_stream.remote(
                    torch.distributed.get_rank(),
                    src_blocks
                )
                
                # STEP 2: Receive data
                block_idx_tensor = torch.tensor(dst_blocks, dtype=torch.long, device=self.device)
                
                num_kv = 2
                num_blocks = len(dst_blocks)
                block_size = self.kv_cache[0].shape[2]
                num_heads = self.kv_cache[0].shape[3]
                head_size = self.kv_cache[0].shape[4]
                kv_dtype = self.kv_cache[0].dtype
                
                layer_shape = (num_kv, num_blocks, block_size, num_heads, head_size)
                total_bytes = 0
                
                # Receive layer by layer
                for layer_idx, layer_kv in enumerate(self.kv_cache):
                    layer_data = torch.empty(layer_shape, dtype=kv_dtype, device=self.device)
                    torch.distributed.recv(layer_data, src=src_rank, group=self.p2p_process_group)
                    layer_kv[:, block_idx_tensor, :, :, :] = layer_data
                    total_bytes += layer_data.numel() * layer_data.element_size()
                
                # Record event (marks transfer completion point)
                self.comm_event.record(self.comm_stream)
            
            # ⚠️ IMPORTANT: Do NOT synchronize here - let scheduler decide when to sync
            # Transfer runs asynchronously on comm_stream
            # Will be synchronized in sync_comm_stream() before compute
            
            # Wait for send to complete
            send_result = ray.get(send_future)
            
            elapsed = time.time() - start_time
            
            if "error" in send_result:
                return {"error": f"Send failed: {send_result['error']}", "bytes": 0, "blocks": 0}
            
            return {
                "bytes": total_bytes,
                "blocks": len(dst_blocks),
                "elapsed": elapsed,
                "transfer_complete": False,  # Async transfer, needs sync later
            }
            
        except Exception as e:
            error_msg = f"Async transfer failed: {type(e).__name__}: {str(e)}"
            log_error(f"[Worker] {error_msg}")
            return {"error": error_msg, "bytes": 0, "blocks": 0}
    
    def sync_comm_stream(self) -> Dict[str, Any]:
        """
        🔄 Synchronize communication stream with compute stream
        Call this before compute to ensure KV transfer is complete
        """
        if self.comm_stream is None or self.comm_event is None:
            return {"synced": True, "elapsed": 0.0}
        
        import time
        start = time.time()
        
        # Make default stream wait for comm_stream
        self.comm_event.synchronize()
        
        elapsed = time.time() - start
        log_debug(f"[Worker {self.worker_id}] Synced comm_stream (waited {elapsed*1000:.2f}ms)")
        
        return {"synced": True, "elapsed": elapsed}
    
    def p2p_send_kv_on_stream(self, dst_rank: int, block_indices: List[int]) -> Dict[str, Any]:
        """Send KV cache blocks on communication stream"""
        if not self.nccl_pg_initialized:
            return {"error": "NCCL not initialized"}
        
        # Initialize comm stream if needed
        if self.comm_stream is None:
            self.init_comm_stream()
        
        try:
            with torch.cuda.stream(self.comm_stream):
                block_idx_tensor = torch.tensor(block_indices, dtype=torch.long, device=self.device)
                
                total_bytes = 0
                for layer_idx, layer_kv in enumerate(self.kv_cache):
                    layer_data = layer_kv[:, block_idx_tensor, :, :, :].contiguous()
                    torch.distributed.send(layer_data, dst=dst_rank, group=self.p2p_process_group)
                    total_bytes += layer_data.numel() * layer_data.element_size()
            
            return {"bytes": total_bytes, "blocks": len(block_indices)}
            
        except Exception as e:
            return {"error": str(e)}
    
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


