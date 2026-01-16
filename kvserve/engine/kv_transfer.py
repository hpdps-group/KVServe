"""KV Transfer Manager for PD separation engine"""

import asyncio
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from kvserve.engine.logger import log_debug, log_warning, log_error, log_info


class TransferMethod(Enum):
    """KV transfer method"""
    NCCL = "nccl"  # NCCL P2P transfer
    P2P_COPY = "p2p_copy"  # Direct GPU-to-GPU copy
    SIMULATION = "simulation"  # File-based simulation (no real transfer)


class KVTransferManager:
    """
    Manages KV cache transfers between Prefill and Decode stages
    """
    
    def __init__(
        self,
        transfer_method: TransferMethod = TransferMethod.NCCL,
        metrics_callback=None,
        simulation_dir: Optional[str] = None,  # For SIMULATION mode
    ):
        """
        Initialize KV transfer manager
        
        Args:
            transfer_method: Method for transferring KV cache
            metrics_callback: Callback function to record transfer metrics (transfer_time, num_blocks)
            simulation_dir: Directory for saving KV files in SIMULATION mode
        """
        self.transfer_method = transfer_method
        self.metrics_callback = metrics_callback
        
        # Worker registry using global_rank as key
        self.worker_registry: Dict[int, Any] = {}
        
        # Transfer statistics
        self.transfer_count = 0
        self.total_transfer_time = 0.0
        self.total_bytes_transferred = 0
        
        # Simulation mode
        self.simulation_dir = simulation_dir
        self.simulation_manifest: Dict[str, Dict] = {}  # {request_id: {file, bytes, ...}}
        if transfer_method == TransferMethod.SIMULATION:
            import os
            if not simulation_dir:
                raise ValueError("simulation_dir required for SIMULATION mode")
            os.makedirs(simulation_dir, exist_ok=True)
    
    def register_worker(self, global_rank: int, worker: Any) -> None:
        """
        Register a worker using global_rank as key
        
        Args:
            global_rank: Worker's global NCCL rank
            worker: Worker reference
        """
        self.worker_registry[global_rank] = worker
    
    def unregister_worker(self, global_rank: int) -> None:
        """Unregister a worker by global_rank"""
        if global_rank in self.worker_registry:
            del self.worker_registry[global_rank]
    
    async def transfer_kv_cache(
        self,
        request_id: str,
        src_rank: int,
        dst_rank: int,
        src_blocks: List[int],
        dst_blocks: List[int],
    ) -> bool:
        """
        Transfer KV cache from prefill to decode stage using global ranks (single-stream)
        
        Args:
            request_id: Request ID
            src_rank: Source worker's global NCCL rank (prefill)
            dst_rank: Destination worker's global NCCL rank (decode)
            src_blocks: Source block IDs
            dst_blocks: Destination block IDs
            
        Returns:
            True if transfer successful
        """
        if self.transfer_method == TransferMethod.NCCL:
            return await self._transfer_via_nccl_p2p(
                request_id,
                src_rank, src_blocks,
                dst_rank, dst_blocks,
            )
        elif self.transfer_method == TransferMethod.P2P_COPY:
            return await self._transfer_via_p2p_copy(
                request_id,
                src_rank, src_blocks,
                dst_rank, dst_blocks
            )
        elif self.transfer_method == TransferMethod.SIMULATION:
            return await self._transfer_via_simulation(
                request_id,
                src_rank, src_blocks,
                dst_rank, dst_blocks,
            )
        else:
            return False
    
    async def transfer_kv_cache_async(
        self,
        request_id: str,
        src_rank: int,
        dst_rank: int,
        src_blocks: List[int],
        dst_blocks: List[int],
    ) -> Dict[str, Any]:
        """
        🚀 TRUE ASYNC: Start transfer and return immediately (non-blocking)
        
        This method initiates transfer on comm_stream and returns ObjectRef
        without waiting. The caller decides when to wait for completion.
        
        Args:
            request_id: Request ID
            src_rank: Source worker's global NCCL rank
            dst_rank: Destination worker's global NCCL rank
            src_blocks: Source block IDs
            dst_blocks: Destination block IDs
            
        Returns:
            Dict with transfer handle:
            {
                "success": bool,
                "transfer_ref": ray.ObjectRef,  # Future to wait on
                "dst_worker": ray.ActorHandle,  # For sync_comm_stream()
                "dst_rank": int,
                "start_time": float,            # For timing
                "num_blocks": int,              # For stats
                "error": str (if failed)
            }
        """
        try:
            start_time = time.perf_counter()
            
            # Get worker references
            src_worker = self.worker_registry.get(src_rank)
            dst_worker = self.worker_registry.get(dst_rank)
            
            if src_worker is None or dst_worker is None:
                log_error(f"[KVTransfer] Worker not found (src_rank={src_rank}, dst_rank={dst_rank})")
                return {"success": False, "error": "Worker not found"}
            
            # ✅ Start async transfer - DO NOT AWAIT!
            transfer_ref = dst_worker.p2p_transfer_kv_async.remote(
                src_worker,
                src_rank,
                src_blocks,
                dst_blocks,
                timeout=10.0
            )
            
            log_debug(f"[KVTransfer] 🚀 Started non-blocking transfer for {request_id} "
                     f"({len(src_blocks)} blocks)")
            
            # ✅ Return ObjectRef immediately - caller decides when to wait
            return {
                "success": True,
                "transfer_ref": transfer_ref,    # Ray ObjectRef (future)
                "dst_worker": dst_worker,         # Needed for sync_comm_stream()
                "dst_rank": dst_rank,
                "start_time": start_time,
                "num_blocks": len(src_blocks),
                "request_id": request_id,         # For tracking
            }
            
        except Exception as e:
            log_error(f"[KVTransfer] Failed to start async transfer for {request_id}: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    async def _transfer_via_nccl_p2p(
        self,
        request_id: str,
        src_rank: int,
        src_blocks: List[int],
        dst_rank: int,
        dst_blocks: List[int],
    ) -> bool:
        """
        Transfer KV cache via NCCL P2P using coordinated transfer (optimized)
        This uses a single coordinated transfer method for better performance
        """
        try:
            import time
            start_time = time.perf_counter()
            
            # Get worker references
            src_worker = self.worker_registry.get(src_rank)
            dst_worker = self.worker_registry.get(dst_rank)
            
            if src_worker is None or dst_worker is None:
                log_error(f"[KVTransfer] Worker not found (src_rank={src_rank}, dst_rank={dst_rank})")
                return False
            
            # ✅ OPTIMIZED: Use coordinated transfer (single remote call)
            # This is more efficient than separate send/recv calls
            # Check if compression is enabled on source worker
            has_compression = await src_worker.compression_manager_is_enabled.remote()
            
            if has_compression:
                # Use compressed transfer
                recv_result = await dst_worker.p2p_coordinated_transfer_kv_compressed.remote(
                    src_worker,    # Pass src worker reference
                    src_rank,      # NCCL rank of source
                    src_blocks,    # Blocks to send from source
                    dst_blocks,    # Blocks to write in destination
                    request_id,    # Request ID for compression tracking
                    timeout=10.0   # 10s timeout for safety
                )
            else:
                # Use regular transfer
                recv_result = await dst_worker.p2p_coordinated_transfer_kv.remote(
                    src_worker,    # Pass src worker reference
                    src_rank,      # NCCL rank of source
                    src_blocks,    # Blocks to send from source
                    dst_blocks,    # Blocks to write in destination
                    timeout=10.0   # 10s timeout for safety
                )
            
            elapsed = time.perf_counter() - start_time
            
            # Check for errors
            if "error" in recv_result:
                error_msg = recv_result.get("error", "Unknown error")
                log_error(f"[KVTransfer] Transfer failed for {request_id}: {error_msg}")
                return False
            
            # Record metrics
            bytes_transferred = recv_result.get("bytes", 0)
            if bytes_transferred == 0:
                log_warning(f"[KVTransfer] Transfer returned 0 bytes for {request_id}")
                return False
            
            bandwidth_gbps = (bytes_transferred / 1e9) / elapsed if elapsed > 0 else 0
            
            self.transfer_count += 1
            self.total_transfer_time += elapsed
            self.total_bytes_transferred += bytes_transferred
            
            if self.metrics_callback:
                self.metrics_callback(elapsed, len(src_blocks))
            
            # Log for debugging (only first few transfers to avoid spam)
            if self.transfer_count <= 3:
                log_info(f"[KVTransfer] ✓ Coordinated P2P: {len(src_blocks)} blocks, "
                      f"{bytes_transferred/1e6:.2f} MB, {elapsed*1000:.2f} ms, {bandwidth_gbps:.2f} GB/s - {request_id}")
            
            return True
            
        except Exception as e:
            log_error(f"[KVTransfer] Error transferring KV cache for {request_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _transfer_via_p2p_copy(
        self,
        request_id: str,
        src_rank: int,
        src_blocks: List[int],
        dst_rank: int,
        dst_blocks: List[int],
    ) -> bool:
        """Transfer KV cache via direct GPU-to-GPU copy (fallback)"""
        # Similar to NCCL but with direct copy
        # For now, fallback to NCCL
        return await self._transfer_via_nccl_p2p(
            request_id,
            src_rank, src_blocks,
            dst_rank, dst_blocks,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transfer statistics"""
        return {
            "transfer_count": self.transfer_count,
            "total_transfer_time": self.total_transfer_time,
            "total_bytes_transferred": self.total_bytes_transferred,
            "avg_transfer_time": self.total_transfer_time / self.transfer_count if self.transfer_count > 0 else 0,
            "avg_bandwidth_gbps": (
                (self.total_bytes_transferred / 1e9) / self.total_transfer_time
                if self.total_transfer_time > 0 else 0
            ),  # GB/s (not Gbps)
        }

    async def _transfer_via_simulation(
        self,
        request_id: str,
        src_rank: int,
        src_blocks: List[int],
        dst_rank: int,
        dst_blocks: List[int],
    ) -> bool:
        """
        Simulation mode: extract and compress KV, save to file (prefill) OR load from file (decode)
        """
        try:
            import pickle
            import os
            
            # Check if this is decode stage (manifest already loaded)
            if request_id in self.simulation_manifest:
                # Decode stage: load from file
                info = self.simulation_manifest[request_id]
                kv_file = info['kv_file']
                
                if not os.path.exists(kv_file):
                    log_error(f"[KVTransfer-SIM] KV file not found: {kv_file}")
                    return False
                
                # Load KV data
                with open(kv_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Get destination worker and write KV
                dst_worker = self.worker_registry.get(dst_rank)
                if dst_worker is None:
                    log_error(f"[KVTransfer-SIM] Dest worker not found (rank={dst_rank})")
                    return False
                
                # Decompress and write
                result = await dst_worker.decompress_and_write_kv.remote(
                    data['compressed_kv'],
                    dst_blocks
                )
                
                # Update manifest with decompression time
                self.simulation_manifest[request_id]['decompression_time_ms'] = result['decompression_time_ms']
                self.simulation_manifest[request_id]['write_time_ms'] = result['write_time_ms']
                
                log_debug(f"[KVTransfer-SIM] Loaded KV for {request_id}: "
                         f"decompress={result['decompression_time_ms']:.2f}ms, "
                         f"write={result['write_time_ms']:.2f}ms")
                
                return True
            
            else:
                # Prefill stage: extract and save to file
                src_worker = self.worker_registry.get(src_rank)
                if src_worker is None:
                    log_error(f"[KVTransfer-SIM] Source worker not found (rank={src_rank})")
                    return False
                
                # Extract and compress KV (with internal timing)
                result = await src_worker.extract_and_compress_kv.remote(src_blocks, request_id)
                
                # Save to file (use async I/O to avoid blocking)
                kv_file = os.path.join(self.simulation_dir, f"{request_id}_kv.pkl")
                data_to_save = {
                    'request_id': request_id,
                    'compressed_kv': result.get('compressed_kv') or result.get('kv_data'),
                    'dst_blocks': dst_blocks,
                    'src_blocks': src_blocks,  # Save src_blocks for reference
                    'kv_bytes': result['bytes'],
                    'compression_time_ms': result.get('compression_time_ms', 0),
                    'extract_time_ms': result.get('extract_time_ms', 0),
                }
                
                # Use asyncio.to_thread to run file I/O in background thread
                def save_file():
                    with open(kv_file, 'wb') as f:
                        pickle.dump(data_to_save, f)
                
                await asyncio.to_thread(save_file)
                
                # Record in manifest
                original_bytes = result.get('original_bytes', result['bytes'])  # Fallback to bytes if no original
                self.simulation_manifest[request_id] = {
                    'kv_file': kv_file,
                    'kv_bytes': result['bytes'],
                    'original_bytes': original_bytes,
                    'dst_blocks': dst_blocks,
                    'src_blocks': src_blocks,
                    'dst_rank': dst_rank,
                    'compression_time_ms': result.get('compression_time_ms', 0),
                    'extract_time_ms': result.get('extract_time_ms', 0),
                }
                
                log_debug(f"[KVTransfer-SIM] Saved KV for {request_id}: {result['bytes']/1e6:.2f} MB, "
                         f"compression={result.get('compression_time_ms', 0):.2f}ms")
                
                return True
            
        except Exception as e:
            log_error(f"[KVTransfer-SIM] Failed for {request_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_simulation_manifest(self, manifest_file: str):
        """Save simulation manifest to file"""
        import pickle
        with open(manifest_file, 'wb') as f:
            pickle.dump(self.simulation_manifest, f)
        log_info(f"[KVTransfer-SIM] Saved manifest: {len(self.simulation_manifest)} requests")
    
    def load_simulation_manifest(self, manifest_file: str):
        """Load simulation manifest from file"""
        import pickle
        with open(manifest_file, 'rb') as f:
            self.simulation_manifest = pickle.load(f)
        log_info(f"[KVTransfer-SIM] Loaded manifest: {len(self.simulation_manifest)} requests")
    
    def get_simulation_manifest(self) -> Dict[str, Dict]:
        """Get simulation manifest"""
        return self.simulation_manifest


