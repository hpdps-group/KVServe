"""
Profile data structure for compression configurations.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class Profile:
    """
    Compression profile containing configuration and performance metrics.
    
    Attributes:
        profile_id: Unique identifier (e.g., "llama3.1-8b_longbench_T0_Q2_C1")
        metadata: Model name, dataset, description
        compression_config: Configuration for CompressionManager
        performance_metrics: Measured compression ratio, speeds, etc.
        quality_metrics: Accuracy impact
    """
    profile_id: str
    metadata: Dict[str, Any]
    compression_config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    quality_metrics: Dict[str, float]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Profile':
        """
        Load Profile from JSON dictionary.
        
        Args:
            data: Dictionary containing profile data
            
        Returns:
            Profile object
        """
        return cls(
            profile_id=data['profile_id'],
            metadata=data.get('metadata', {}),
            compression_config=data['compression_config'],
            performance_metrics=data['performance_metrics'],
            quality_metrics=data['quality_metrics']
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Profile to JSON-serializable dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'profile_id': self.profile_id,
            'metadata': self.metadata,
            'compression_config': self.compression_config,
            'performance_metrics': self.performance_metrics,
            'quality_metrics': self.quality_metrics
        }
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio (cr)"""
        return self.performance_metrics['compression_ratio']
    
    @property
    def harmonic_speed(self) -> float:
        """Harmonic speed S = (S1*S2)/(S1+S2) in MB/s"""
        return self.performance_metrics['harmonic_speed_mbps']
    
    @property
    def critical_bandwidth(self) -> float:
        """Critical bandwidth B_crit = (1 - 1/cr) * S"""
        return self.performance_metrics.get(
            'critical_bandwidth_mbps',
            (1 - 1/self.compression_ratio) * self.harmonic_speed
        )
    
    @property
    def accuracy(self) -> float:
        """Quality metric (e.g., ROUGE-L score)"""
        return self.quality_metrics['accuracy']
    
    def __repr__(self) -> str:
        return (f"Profile(id={self.profile_id}, "
                f"cr={self.compression_ratio:.2f}, "
                f"acc={self.accuracy:.3f})")



