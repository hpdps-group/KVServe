"""
Online controller for adaptive compression profile selection.

Implements two-tier decision making:
1. Analytical model: Fast screening based on Theorems 1 & 2
2. ε-greedy bandit: Online learning of model residuals
"""

import random
from typing import Dict, Optional, Tuple, Any

from kvserve.controller.profile import Profile
from kvserve.controller.analytical_model import AnalyticalModel
from kvserve.controller.bandit_state import BanditStateManager
from kvserve.controller.profile_library import ProfileLibrary


class OnlineController:
    """
    Two-tier online controller for compression profile selection.
    
    Tier 1 (Analytical Model):
        - Filter by accuracy requirement
        - Partition bandwidth space (1/B)
        - Screen using benefit condition (Theorem 1)
        - Narrow down to 2-3 candidates (Theorem 2)
    
    Tier 2 (ε-greedy Bandit):
        - Predict latency using analytical model
        - Correct with learned residuals (EWMA)
        - Select using ε-greedy strategy
        - Update residuals from observations
    """
    
    def __init__(
        self,
        profile_library: ProfileLibrary,
        epsilon: float = 0.1,
        bandit_state: Optional[BanditStateManager] = None,
        alpha: float = 0.2
    ):
        """
        Initialize online controller.
        
        Args:
            profile_library: Pre-built profile library
            epsilon: ε-greedy exploration rate (0 < epsilon < 1)
            bandit_state: Optional pre-loaded bandit state
            alpha: EWMA learning rate for bandit (0 < alpha <= 1)
        """
        self.library = profile_library
        self.epsilon = epsilon
        self.bandit = bandit_state or BanditStateManager(alpha=alpha)
        self.model = AnalyticalModel()
        
        # Statistics
        self.total_decisions = 0
        self.exploit_count = 0
        self.explore_count = 0
        self.no_compression_count = 0
    
    def select_profile(
        self,
        V_bytes: float,
        B_mbps: float,
        T_model_ms: float,
        T_SLO_ms: float,
        acc_req: float
    ) -> Tuple[Optional[Profile], Dict[str, Any]]:
        """
        Select optimal compression profile using two-tier decision.
        
        Args:
            V_bytes: KV cache volume in bytes
            B_mbps: Network bandwidth in MB/s
            T_model_ms: Model computation latency in milliseconds
            T_SLO_ms: SLO constraint in milliseconds
            acc_req: Required accuracy (e.g., 0.92)
            
        Returns:
            (selected_profile, context)
            - selected_profile: Selected Profile or None (no compression)
            - context: Decision context for later update
        """
        self.total_decisions += 1
        
        # ========== Tier 1: Analytical Model Screening ==========
        
        # Step 1: Find accuracy bucket
        bucket = self.library.find_bucket(acc_req)
        if bucket is None:
            self.no_compression_count += 1
            return None, {
                'reason': 'no_matching_bucket',
                'acc_req': acc_req
            }
        
        # Step 2: Find bandwidth interval
        interval = self.library.find_interval(bucket, B_mbps)
        if interval is None:
            self.no_compression_count += 1
            return None, {
                'reason': 'no_matching_interval',
                'B_mbps': B_mbps
            }
        
        # Step 3: Get candidate profiles (already top-3 from Pareto frontier)
        candidates = self.library.get_candidate_profiles(bucket, interval)
        if not candidates:
            self.no_compression_count += 1
            return None, {
                'reason': 'no_candidates',
                'bucket_id': bucket['bucket_id'],
                'interval_id': interval['interval_id']
            }
        
        # Step 4: Filter by benefit condition (Theorem 1)
        candidates = [
            p for p in candidates 
            if self.model.check_benefit(p, B_mbps)
        ]
        
        if not candidates:
            self.no_compression_count += 1
            return None, {
                'reason': 'no_benefit',
                'B_mbps': B_mbps,
                'bucket_id': bucket['bucket_id'],
                'interval_id': interval['interval_id']
            }
        
        # ========== Tier 2: Bandit Selection ==========
        
        # Step 5: Predict latency with residual correction
        predictions = {}
        baseline_latency = self.model.predict_baseline_latency(
            V_bytes, B_mbps, T_model_ms
        )
        
        for profile in candidates:
            # Analytical model prediction
            T_hat = self.model.predict_latency(
                profile, V_bytes, B_mbps, T_model_ms
            )
            
            # Load historical residual
            _, delta_bar = self.bandit.get_stats(
                bucket['bucket_id'],
                interval['interval_id'],
                profile.profile_id
            )
            
            # Corrected effective latency
            T_eff = T_hat + delta_bar
            
            # Check SLO constraint
            if T_eff <= T_SLO_ms:
                predictions[profile.profile_id] = {
                    'profile': profile,
                    'T_hat': T_hat,
                    'T_eff': T_eff,
                    'delta_bar': delta_bar,
                    'speedup': baseline_latency / T_eff
                }
        
        if not predictions:
            self.no_compression_count += 1
            return None, {
                'reason': 'slo_violation',
                'T_SLO_ms': T_SLO_ms,
                'bucket_id': bucket['bucket_id'],
                'interval_id': interval['interval_id'],
                'candidates': [p.profile_id for p in candidates]
            }
        
        # Step 6: ε-greedy selection
        action_type = 'explore' if random.random() < self.epsilon else 'exploit'
        
        if action_type == 'explore':
            # Exploration: Random selection
            selected_id = random.choice(list(predictions.keys()))
            self.explore_count += 1
        else:
            # Exploitation: Select minimum T_eff
            selected_id = min(
                predictions.keys(),
                key=lambda pid: predictions[pid]['T_eff']
            )
            self.exploit_count += 1
        
        selected_profile = predictions[selected_id]['profile']
        
        # Step 7: Build context for update
        context = {
            'bucket_id': bucket['bucket_id'],
            'interval_id': interval['interval_id'],
            'profile_id': selected_id,
            'T_hat': predictions[selected_id]['T_hat'],
            'T_eff': predictions[selected_id]['T_eff'],
            'delta_bar': predictions[selected_id]['delta_bar'],
            'speedup_predicted': predictions[selected_id]['speedup'],
            'action': action_type,
            'num_candidates': len(candidates),
            'baseline_latency': baseline_latency,
            'V_bytes': V_bytes,
            'B_mbps': B_mbps,
            'T_model_ms': T_model_ms,
            'T_SLO_ms': T_SLO_ms
        }
        
        return selected_profile, context
    
    def update(self, context: Dict[str, Any], T_obs_ms: float):
        """
        Update bandit state with observed latency.
        
        Args:
            context: Decision context from select_profile
            T_obs_ms: Observed end-to-end latency in milliseconds
        """
        if 'profile_id' not in context:
            # No compression was used, nothing to update
            return
        
        # Compute residual
        T_hat = context['T_hat']
        delta_obs = T_obs_ms - T_hat
        
        # Update bandit state using EWMA
        self.bandit.update(
            context['bucket_id'],
            context['interval_id'],
            context['profile_id'],
            delta_obs
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get controller statistics.
        
        Returns:
            Dictionary containing decision statistics
        """
        return {
            'total_decisions': self.total_decisions,
            'exploit_count': self.exploit_count,
            'explore_count': self.explore_count,
            'no_compression_count': self.no_compression_count,
            'compression_rate': (
                (self.total_decisions - self.no_compression_count) / self.total_decisions
                if self.total_decisions > 0 else 0
            ),
            'exploitation_rate': (
                self.exploit_count / (self.exploit_count + self.explore_count)
                if (self.exploit_count + self.explore_count) > 0 else 0
            ),
            'bandit_states': self.bandit.get_total_states(),
            'bandit_memory_bytes': self.bandit.get_memory_size()
        }
    
    def reset_statistics(self):
        """Reset decision statistics (does not reset bandit state)."""
        self.total_decisions = 0
        self.exploit_count = 0
        self.explore_count = 0
        self.no_compression_count = 0
    
    def save_state(self, path: str):
        """Save bandit state to file."""
        self.bandit.save(path)
    
    def load_state(self, path: str):
        """Load bandit state from file."""
        self.bandit.load(path)
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"OnlineController(\n"
            f"  epsilon={self.epsilon:.2f},\n"
            f"  decisions={stats['total_decisions']},\n"
            f"  compression_rate={stats['compression_rate']:.2%},\n"
            f"  bandit_states={stats['bandit_states']}\n"
            f")"
        )



