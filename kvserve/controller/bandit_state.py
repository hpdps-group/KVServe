"""
Simplified bandit state management for throughput correction.

Only tracks throughput correction factors for each profile:
- correction_factor: multiplier to adjust offline throughput to actual (EWMA)
"""

import json
from typing import Dict
from pathlib import Path


class BanditStateManager:
    """
    Simplified bandit state manager.
    
    Only tracks throughput correction factor for each profile:
        throughput_actual = throughput_offline × correction_factor
    
    State structure:
        {profile_id: correction_factor}
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize bandit state manager.
        
        Args:
            alpha: EWMA learning rate (0 < alpha <= 1)
                   Higher alpha gives more weight to recent observations
        """
        self.alpha = alpha
        self.state: Dict[str, float] = {}  # {profile_id: correction_factor}
    
    def get_throughput_correction(self, profile_id: str) -> float:
        """
        Get throughput correction factor for a profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Correction factor (default 1.0 = no correction)
        """
        return self.state.get(profile_id, 1.0)
    
    def update_throughput_correction(
        self,
        profile_id: str,
        actual_throughput: float,
        predicted_throughput: float
    ):
        """
        Update throughput correction factor using EWMA.
        
        Formula:
            correction_new = actual / predicted
            correction_factor = (1-α) × correction_old + α × correction_new
        
        Args:
            profile_id: Profile identifier
            actual_throughput: Measured throughput (MB/s)
            predicted_throughput: Predicted throughput from profile (MB/s)
        """
        if predicted_throughput <= 0:
            return
        
        # Calculate correction from this observation
        correction_obs = actual_throughput / predicted_throughput
        
        # EWMA update
        correction_old = self.state.get(profile_id, 1.0)
        correction_new = (1 - self.alpha) * correction_old + self.alpha * correction_obs
        
        self.state[profile_id] = correction_new
    
    def reset(self):
        """Clear all state."""
        self.state.clear()
    
    def save(self, path: str):
        """
        Save state to JSON file.
        
        Args:
            path: File path to save state
        """
        data = {
            'alpha': self.alpha,
            'state': self.state
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
        self.state = data['state']
    
    def get_total_states(self) -> int:
        """Get total number of tracked profiles."""
        return len(self.state)
    
    def get_memory_size(self) -> int:
        """
        Estimate memory usage in bytes.
        
        Returns:
            Approximate memory size in bytes
        """
        # Each state: (str, float) ≈ ~30 bytes
        return self.get_total_states() * 30
    
    def __repr__(self) -> str:
        return (f"BanditStateManager(alpha={self.alpha}, "
                f"profiles={self.get_total_states()}, "
                f"memory≈{self.get_memory_size()}B)")
