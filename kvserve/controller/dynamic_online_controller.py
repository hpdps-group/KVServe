"""
Dynamic online controller for adaptive compression profile selection.

Simplified version that works with DynamicProfileLibrary.
"""

import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from kvserve.controller.profile import Profile
from kvserve.controller.analytical_model import AnalyticalModel
from kvserve.controller.dynamic_profile_library import DynamicProfileLibrary


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
        epsilon: float = 0.0,  # 0.0 = pure exploitation (no exploration)
        max_candidates: int = 3
    ):
        """
        Initialize dynamic online controller.
        
        Args:
            model_name: Model name
            dataset: Dataset name
            machine: Machine name
            profiles_dir: Directory containing profile JSON files
            speed_results_dir: Directory containing speed CSV files
            epsilon: ε-greedy exploration rate (0 = pure exploitation)
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
        self.epsilon = epsilon
        self.max_candidates = max_candidates
        
        # Statistics
        self.total_decisions = 0
        self.selection_times = []  # Track selection time for each decision
    
    def select_profile(
        self,
        input_length: int,
        V_bytes: float,
        B_mbps: float,
        T_model_ms: float,
        acc_req: float
    ) -> Tuple[Optional[Profile], Dict[str, Any]]:
        """
        Select optimal compression profile.
        
        Args:
            input_length: Input sequence length (in tokens)
            V_bytes: KV cache volume in bytes
            B_mbps: Network bandwidth in MB/s
            T_model_ms: Model computation latency in milliseconds
            acc_req: Required accuracy (e.g., 0.95)
            
        Returns:
            (selected_profile, context)
            - selected_profile: Selected Profile or None (no compression)
            - context: Decision context with timing and metadata
        """
        start_time = time.perf_counter()
        
        self.total_decisions += 1
        
        # Get candidate profiles
        candidates = self.library.get_candidate_profiles(
            input_length=input_length,
            acc_req=acc_req,
            B_mbps=B_mbps,
            max_candidates=self.max_candidates
        )
        
        if not candidates:
            selection_time = (time.perf_counter() - start_time) * 1000  # ms
            self.selection_times.append(selection_time)
            
            return None, {
                'reason': 'no_candidates',
                'selection_time_ms': selection_time,
                'num_candidates': 0
            }
        
        # Predict latency for each candidate
        baseline_latency = self.model.predict_baseline_latency(
            V_bytes, B_mbps, T_model_ms
        )
        
        predictions = {}
        for profile in candidates:
            T_hat = self.model.predict_latency(
                profile, V_bytes, B_mbps, T_model_ms
            )
            
            predictions[profile.profile_id] = {
                'profile': profile,
                'T_hat': T_hat,
                'speedup': baseline_latency / T_hat if T_hat > 0 else 0
            }
        
        # Select best candidate (exploitation) or random (exploration)
        if random.random() < self.epsilon and len(predictions) > 1:
            # Exploration
            selected_id = random.choice(list(predictions.keys()))
            action = 'explore'
        else:
            # Exploitation: select minimum latency
            selected_id = min(
                predictions.keys(),
                key=lambda pid: predictions[pid]['T_hat']
            )
            action = 'exploit'
        
        selected_profile = predictions[selected_id]['profile']
        
        selection_time = (time.perf_counter() - start_time) * 1000  # ms
        self.selection_times.append(selection_time)
        
        # Build context
        context = {
            'profile_id': selected_id,
            'T_hat': predictions[selected_id]['T_hat'],
            'speedup_predicted': predictions[selected_id]['speedup'],
            'baseline_latency': baseline_latency,
            'action': action,
            'num_candidates': len(candidates),
            'selection_time_ms': selection_time,
            'input_length': input_length,
            'V_bytes': V_bytes,
            'B_mbps': B_mbps,
            'T_model_ms': T_model_ms,
            'acc_req': acc_req
        }
        
        return selected_profile, context
    
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
            'cached_input_lengths': self.library.get_cache_size()
        }
    
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
