"""KV cache storage manager for simulator mode (CPU-based)"""

import torch
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KVPackage:
    """KV cache package stored in CPU memory"""
    request_id: str
    data: torch.Tensor  # CPU tensor (contiguous)
    num_blocks: int
    size_bytes: int
    block_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    is_compressed: bool = False
    compression_meta: Optional[Dict] = None


class KVStorage:
    """
    Manages KV cache storage in CPU memory for simulator mode.
    Provides pack (GPU->CPU) and unpack (CPU->GPU) operations with IO time estimation.
    """
    
    def __init__(
        self,
        pcie_gbps: float = 24.0,  # PCIe Gen4 x16 ~ 24 GB/s effective
        io_jitter_ms: float = 0.5,  # Random jitter for IO
    ):
        self.pcie_gbps = pcie_gbps
        self.io_jitter_ms = io_jitter_ms
        
        # Storage: {request_id: KVPackage}
        self.storage: Dict[str, KVPackage] = {}
    
    def pack_kv(
        self,
        request_id: str,
        kv_blocks: torch.Tensor,  # [num_blocks, 2, num_layers, num_heads, block_size, head_dim]
        block_indices: List[int],
        worker: Any = None,  # Optional: worker reference to fetch KV
    ) -> Tuple[KVPackage, float]:
        """
        Pack KV cache blocks from GPU to CPU memory.
        
        Returns:
            (package, io_time_ms): Packed KV and estimated IO time
        """
        start_time = time.perf_counter()
        
        # Move to CPU and make contiguous
        if kv_blocks.is_cuda:
            cpu_data = kv_blocks.cpu().contiguous()
        else:
            cpu_data = kv_blocks.contiguous()
        
        # Calculate size
        size_bytes = cpu_data.numel() * cpu_data.element_size()
        
        # Create package
        shape = kv_blocks.shape
        package = KVPackage(
            request_id=request_id,
            data=cpu_data,
            num_blocks=len(block_indices),
            size_bytes=size_bytes,
            block_size=shape[-2] if len(shape) >= 2 else 0,
            num_layers=shape[2] if len(shape) >= 3 else 0,
            num_heads=shape[3] if len(shape) >= 4 else 0,
            head_dim=shape[-1] if len(shape) >= 1 else 0,
            dtype=kv_blocks.dtype,
        )
        
        # Estimate IO time (GPU->CPU via PCIe)
        io_time_ms = self._estimate_io_time(size_bytes)
        
        # Store
        self.storage[request_id] = package
        
        return package, io_time_ms
    
    def unpack_kv(
        self,
        request_id: str,
        device: str = "cuda:0",
    ) -> Tuple[torch.Tensor, float]:
        """
        Unpack KV cache from CPU to GPU memory.
        
        Returns:
            (gpu_tensor, io_time_ms): GPU tensor and estimated IO time
        """
        if request_id not in self.storage:
            raise KeyError(f"KV package for {request_id} not found in storage")
        
        package = self.storage[request_id]
        
        start_time = time.perf_counter()
        
        # Move to GPU
        gpu_data = package.data.to(device, non_blocking=True)
        
        # Estimate IO time (CPU->GPU via PCIe)
        io_time_ms = self._estimate_io_time(package.size_bytes)
        
        return gpu_data, io_time_ms
    
    def _estimate_io_time(self, size_bytes: int) -> float:
        """
        Estimate PCIe transfer time (ms) with jitter.
        
        Args:
            size_bytes: Data size in bytes
            
        Returns:
            Estimated time in milliseconds
        """
        size_gb = size_bytes / (1024**3)
        base_time_s = size_gb / self.pcie_gbps
        jitter_s = random.uniform(-self.io_jitter_ms, self.io_jitter_ms) / 1000.0
        time_ms = max(0.1, (base_time_s + jitter_s) * 1000.0)
        return time_ms
    
    def get_size(self, request_id: str) -> int:
        """Get KV package size in bytes"""
        if request_id not in self.storage:
            return 0
        return self.storage[request_id].size_bytes
    
    def remove(self, request_id: str):
        """Remove KV package from storage"""
        if request_id in self.storage:
            del self.storage[request_id]
    
    def clear(self):
        """Clear all stored KV packages"""
        self.storage.clear()

