"""
Dynamic online controller for adaptive compression profile selection.

Simplified version that works with DynamicProfileLibrary.
"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

from kvserve.controller.profile import Profile
from kvserve.controller.analytical_model import AnalyticalModel
from kvserve.controller.dynamic_profile_library import DynamicProfileLibrary
from kvserve.controller.bandit_state import BanditStateManager


class DynamicOnlineController:
    """
    Online controller for compression profile selection with dynamic profile loading.
    
    Workflow:
    1. Filter by accuracy requirement
    2. Filter by benefit condition (Theorem 1)
    3. Select top ≤3 candidates
    4. Use analytical model to predict latency
    5. Select best candidate (or use ε-greedy if bandit is enabled)
    """
    
    def __init__(
        self,
        model_name: str,
        dataset: str,
        prefill_machine: str,
        decode_machine: Optional[str] = None,
        profiles_dir: Optional[str] = None,
        speed_results_dir: Optional[str] = None,
        max_candidates: Optional[int] = None,
        bandit_state: Optional[BanditStateManager] = None,
        alpha: float = 0.2,
        min_compression_ratio: float = 7.0  # Minimum compression ratio threshold
    ):
        """
        Initialize dynamic online controller.
        
        Args:
            model_name: Model name
            dataset: Dataset name
            machine: Machine name
            profiles_dir: Directory containing profile JSON files
            speed_results_dir: Directory containing speed CSV files
            max_candidates: Maximum number of candidates to consider
        """
        base_dir = Path(__file__).resolve().parents[2]
        profiles_dir = profiles_dir or str(base_dir / "profiles")
        speed_results_dir = speed_results_dir or str(base_dir / "speed_results")
        self.library = DynamicProfileLibrary(
            model_name=model_name,
            dataset=dataset,
            prefill_machine=prefill_machine,
            decode_machine=decode_machine,
            profiles_dir=profiles_dir,
            speed_results_dir=speed_results_dir
        )
        
        self.model = AnalyticalModel()
        self.max_candidates = max_candidates
        self.min_compression_ratio = min_compression_ratio
        self.bandit = bandit_state or BanditStateManager(alpha=alpha)
        
        # Statistics
        self.total_decisions = 0
        self.selection_times = []  # Track selection time for each decision
        self.no_compression_count = 0
        self.exploit_count = 0

    def _select_candidates(
        self,
        input_length: int,
        acc_req: float
    ) -> Tuple[List[Profile], Optional[int], Dict[str, Any]]:
        """
        Simplified candidate selection:
        1. Filter by accuracy
        2. Filter by minimum compression ratio
        3. Group by transform type and select best from each group
        """
        all_profiles = self.library.get_profiles(input_length)
        
        # Filter by accuracy
        acc_filtered = [p for p in all_profiles if p.accuracy >= acc_req]
        if not acc_filtered:
            return [], None, {"reason": "no_accuracy_match"}
        
        # Filter by compression ratio
        cr_filtered = [p for p in acc_filtered if p.compression_ratio >= self.min_compression_ratio]
        if not cr_filtered:
            return [], None, {"reason": "no_cr_match"}
        
        # Return ALL filtered profiles as candidates
        # We'll evaluate speedup for all of them in select_profile()
        # This ensures we pick the TRUE best, not just best CR per group
        if not cr_filtered:
            return [], None, {"reason": "no_candidates"}
        
        # Return candidates with metadata
        return cr_filtered, 0, {
            "num_candidates": len(cr_filtered),
            "num_filtered": len(cr_filtered),
            "num_total": len(all_profiles)
        }
    
    def select_profile(
        self,
        input_length: int,
        V_bytes: float,
        B_mbps: float,
        T_model_ms: float,
        acc_req: float
    ) -> Tuple[Optional[Profile], Dict[str, Any]]:
        """
        Select optimal compression profile using simplified strategy.
        
        Strategy:
        1. Get 2 candidates (best from each transform group)
        2. Calculate actual speedup for each: speedup = (T_baseline - T_compressed) / T_baseline
        3. Select the one with highest speedup
        4. Bandit correction adjusts throughput prediction
        
        Args:
            input_length: Input sequence length (in tokens)
            V_bytes: KV cache volume in bytes
            B_mbps: Network bandwidth in MB/s
            T_model_ms: Model computation latency in milliseconds
            acc_req: Required accuracy (e.g., 0.95)
            
        Returns:
            (selected_profile, context)
        """
        start_time = time.perf_counter()
        self.total_decisions += 1
        
        # Get candidates (up to 2: best from each transform group)
        candidates, _, candidate_meta = self._select_candidates(
            input_length=input_length,
            acc_req=acc_req
        )

        if not candidates:
            selection_time = (time.perf_counter() - start_time) * 1000
            self.selection_times.append(selection_time)
            self.no_compression_count += 1
            return None, {
                **candidate_meta,
                'selection_time_ms': selection_time,
            }
        
        # Calculate baseline latency (no compression)
        baseline_latency = self.model.predict_baseline_latency(
            V_bytes, B_mbps, T_model_ms
        )
        
        # Evaluate each candidate with bandit correction
        V_mb = V_bytes / (1024 * 1024)
        best_profile = None
        best_speedup = 0
        best_info = {}
        
        for profile in candidates:
            # Get bandit throughput correction factor
            throughput_correction = self.bandit.get_throughput_correction(
                profile.profile_id
            )
            
            # Adjusted harmonic speed (actual vs offline profile)
            S_adjusted = profile.harmonic_speed * throughput_correction
            cr = profile.compression_ratio
            
            # Calculate compressed latency: T = T_model + V/S + V/(B*cr)
            T_codec = (V_mb / S_adjusted) * 1000.0  # Compression + decompression
            T_network = (V_mb / (B_mbps * cr)) * 1000.0  # Network transfer
            T_compressed = T_model_ms + T_codec + T_network
            
            # Calculate speedup
            speedup = (baseline_latency - T_compressed) / baseline_latency if baseline_latency > 0 else 0
            
            if speedup > best_speedup:
                best_speedup = speedup
                best_profile = profile
                best_info = {
                    'T_compressed': T_compressed,
                    'T_codec': T_codec,
                    'T_network': T_network,
                    'throughput_correction': throughput_correction,
                    'S_adjusted': S_adjusted
                }
        
        selection_time = (time.perf_counter() - start_time) * 1000
        self.selection_times.append(selection_time)
        
        # Decide whether to use compression
        if best_profile is None or best_speedup <= 0:
            self.no_compression_count += 1
            return None, {
                'reason': 'no_speedup',
                'baseline_latency': baseline_latency,
                'selection_time_ms': selection_time,
            }
        
        self.exploit_count += 1
        
        # Build context for bandit update
        context = {
            'profile_id': best_profile.profile_id,
            'predicted_throughput': best_profile.harmonic_speed,  # Store for update
            'speedup_predicted': best_speedup,
            'baseline_latency': baseline_latency,
            'selection_time_ms': selection_time,
            'num_candidates': len(candidates),
            **best_info,
            **candidate_meta
        }
        
        return best_profile, context
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get controller statistics.
        
        Returns:
            Dictionary containing decision statistics
        """
        if self.selection_times:
            avg_time = sum(self.selection_times) / len(self.selection_times)
            max_time = max(self.selection_times)
            min_time = min(self.selection_times)
        else:
            avg_time = max_time = min_time = 0
        
        return {
            'total_decisions': self.total_decisions,
            'avg_selection_time_ms': avg_time,
            'max_selection_time_ms': max_time,
            'min_selection_time_ms': min_time,
            'cached_input_lengths': self.library.get_cache_size(),
            'no_compression_count': self.no_compression_count,
            'exploit_count': self.exploit_count,
            'bandit_states': self.bandit.get_total_states(),
            'bandit_memory_bytes': self.bandit.get_memory_size()
        }

    def update(self, context: Dict[str, Any], metrics: Dict[str, float]):
        """
        Update bandit state with observed compression metrics.
        
        Args:
            context: Context from select_profile (contains profile_id, predicted_throughput)
            metrics: Observed metrics with keys:
                - compression_time_ms: Measured compression time
                - decompression_time_ms: Measured decompression time  
                - kv_size_bytes: KV cache size in bytes
        """
        if 'profile_id' not in context or 'predicted_throughput' not in context:
            return
        
        profile_id = context['profile_id']
        predicted_throughput = context['predicted_throughput']
        
        compression_time = metrics.get('compression_time_ms', 0)
        decompression_time = metrics.get('decompression_time_ms', 0)
        kv_size_bytes = metrics.get('kv_size_bytes', 0)
        
        if compression_time <= 0 or decompression_time <= 0 or kv_size_bytes <= 0:
            return
        
        # Calculate actual throughput (harmonic mean)
        kv_size_mb = kv_size_bytes / (1024 * 1024)
        compress_throughput = (kv_size_mb / compression_time) * 1000.0
        decompress_throughput = (kv_size_mb / decompression_time) * 1000.0
        actual_throughput = 2.0 / (1.0 / compress_throughput + 1.0 / decompress_throughput)
        
        # Update correction factor
        self.bandit.update_throughput_correction(
            profile_id,
            actual_throughput,
            predicted_throughput
        )
    
    def reset_statistics(self):
        """Reset decision statistics."""
        self.total_decisions = 0
        self.selection_times.clear()
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"DynamicOnlineController(\n"
            f"  Model: {self.library.model_name}\n"
            f"  Dataset: {self.library.dataset}\n"
            f"  Prefill machine: {self.library.prefill_machine}\n"
            f"  Decode machine: {self.library.decode_machine}\n"
            f"  Decisions: {stats['total_decisions']}\n"
            f"  Avg selection time: {stats['avg_selection_time_ms']:.3f} ms\n"
            f")"
        )
