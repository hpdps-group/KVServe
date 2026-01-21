"""
Service configuration for runtime parameters
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ServiceConfig:
    """
    Service configuration for dynamic compression decisions
    
    These parameters are used by OnlineController to make runtime decisions.
    They can be updated during service lifetime to simulate changing conditions.
    """
    bandwidth_gbps: float = 0.1  # Network bandwidth in Gbps (bits/s)
    slo_ms: float = 5000.0  # Service Level Objective in milliseconds
    accuracy_requirement: float = 0.90  # Minimum accuracy requirement (0-1)
    model_name: Optional[str] = None  # Model name (e.g., "Llama-3.1-8B-Instruct")
    dataset: Optional[str] = None  # Dataset name (e.g., "longbench_qasper")
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'bandwidth_gbps': self.bandwidth_gbps,
            'slo_ms': self.slo_ms,
            'accuracy_requirement': self.accuracy_requirement,
            'model_name': self.model_name,
            'dataset': self.dataset,
        }
    
    def update(self, **kwargs):
        """Update service config parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


