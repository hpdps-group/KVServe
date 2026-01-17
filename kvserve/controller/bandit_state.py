"""
Bandit state management for online learning.

Maintains statistics for ε-greedy bandit:
- N: Number of times a profile has been used
- delta_bar: EWMA of prediction residuals (T_obs - T_hat)
"""

import json
from typing import Dict, Tuple
from pathlib import Path


class BanditStateManager:
    """
    Manages bandit state for online learning of model residuals.
    
    State structure:
        {(bucket_id, interval_id, profile_id): (N, delta_bar)}
    
    Where:
        - N: Usage count
        - delta_bar: Exponentially weighted moving average of residuals
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize bandit state manager.
        
        Args:
            alpha: EWMA learning rate (0 < alpha <= 1)
                   Higher alpha gives more weight to recent observations
        """
        self.alpha = alpha
        self.state: Dict[Tuple[int, int, str], Tuple[int, float]] = {}
    
    def get_stats(
        self,
        bucket_id: int,
        interval_id: int,
        profile_id: str
    ) -> Tuple[int, float]:
        """
        Get statistics for a specific (bucket, interval, profile) triplet.
        
        Args:
            bucket_id: Accuracy bucket ID
            interval_id: Bandwidth interval ID
            profile_id: Profile identifier
            
        Returns:
            (N, delta_bar): Usage count and residual EWMA
        """
        key = (bucket_id, interval_id, profile_id)
        return self.state.get(key, (0, 0.0))
    
    def update(
        self,
        bucket_id: int,
        interval_id: int,
        profile_id: str,
        delta_obs: float
    ):
        """
        Update statistics with new observation using EWMA.
        
        Formula:
            delta_bar_new = (1 - alpha) * delta_bar_old + alpha * delta_obs
        
        Args:
            bucket_id: Accuracy bucket ID
            interval_id: Bandwidth interval ID
            profile_id: Profile identifier
            delta_obs: Observed residual (T_obs - T_hat)
        """
        key = (bucket_id, interval_id, profile_id)
        N, delta_bar_old = self.state.get(key, (0, 0.0))
        
        # EWMA update
        delta_bar_new = (1 - self.alpha) * delta_bar_old + self.alpha * delta_obs
        
        # Increment usage count
        self.state[key] = (N + 1, delta_bar_new)
    
    def get_usage_count(
        self,
        bucket_id: int,
        interval_id: int,
        profile_id: str
    ) -> int:
        """Get usage count for a profile."""
        N, _ = self.get_stats(bucket_id, interval_id, profile_id)
        return N
    
    def get_residual(
        self,
        bucket_id: int,
        interval_id: int,
        profile_id: str
    ) -> float:
        """Get residual EWMA for a profile."""
        _, delta_bar = self.get_stats(bucket_id, interval_id, profile_id)
        return delta_bar
    
    def reset(self):
        """Clear all state."""
        self.state.clear()
    
    def save(self, path: str):
        """
        Save state to JSON file.
        
        Args:
            path: File path to save state
        """
        # Convert tuple keys to strings for JSON serialization
        serializable_state = {
            f"{bucket_id}_{interval_id}_{profile_id}": {
                'N': N,
                'delta_bar': delta_bar
            }
            for (bucket_id, interval_id, profile_id), (N, delta_bar) 
            in self.state.items()
        }
        
        data = {
            'alpha': self.alpha,
            'state': serializable_state
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """
        Load state from JSON file.
        
        Args:
            path: File path to load state from
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.alpha = data['alpha']
        
        # Convert string keys back to tuples
        self.state = {}
        for key_str, stats in data['state'].items():
            parts = key_str.split('_', 2)
            if len(parts) == 3:
                bucket_id = int(parts[0])
                interval_id = int(parts[1])
                profile_id = parts[2]
                key = (bucket_id, interval_id, profile_id)
                self.state[key] = (stats['N'], stats['delta_bar'])
    
    def get_total_states(self) -> int:
        """Get total number of tracked states."""
        return len(self.state)
    
    def get_memory_size(self) -> int:
        """
        Estimate memory usage in bytes.
        
        Returns:
            Approximate memory size in bytes
        """
        # Each state: (int, int, str, int, float)
        # Rough estimate: 3*8 (tuple overhead) + 8 (int) + 8 (float) + ~20 (string) = ~60 bytes
        return self.get_total_states() * 60
    
    def __repr__(self) -> str:
        return (f"BanditStateManager(alpha={self.alpha}, "
                f"states={self.get_total_states()}, "
                f"memory≈{self.get_memory_size()}B)")


