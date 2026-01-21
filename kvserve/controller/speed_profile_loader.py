"""
Speed profile loader for efficient throughput lookup.

Loads component throughput data from CSV files and provides fast lookup
with automatic nearest-neighbor matching for input lengths.
"""

import csv
import bisect
from pathlib import Path
from typing import Dict, List, Tuple, Optional
class SpeedProfileLoader:
    """
    Loads and queries component throughput data from speed CSV files.
    
    Features:
    - Fast lookup by component type and input length
    - Automatic nearest-neighbor matching for input lengths
    - Cached data for efficiency
    - Harmonic mean computation for encode/decode speeds
    """
    
    def __init__(self, csv_path: str):
        """
        Load speed profile from CSV file.
        
        Args:
            csv_path: Path to speed CSV file (e.g., "5090_Qwen2.5-7B-Instruct_speed.csv")
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV format is invalid
        """
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Speed profile not found: {csv_path}")
        
        self.csv_path = csv_path
        
        # Data structure: {component_type: {input_length: (prefill_gbps, decode_gbps)}}
        self.data: Dict[str, Dict[int, Tuple[float, float]]] = {}
        
        # Sorted input lengths for each component (for binary search)
        self.sorted_lengths: Dict[str, List[int]] = {}
        
        self._load_csv()
    
    def _load_csv(self):
        """Load CSV data into memory."""
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                component_type = row['type'].strip()
                input_length = int(row['input_length'])
                
                # GB/s throughput
                prefill_throughput = float(row['prefill_throughput(GB/s)'])
                decode_throughput = float(row['decode_throughput(GB/s)'])
                
                if component_type not in self.data:
                    self.data[component_type] = {}
                
                self.data[component_type][input_length] = (prefill_throughput, decode_throughput)
        
        # Build sorted length arrays for binary search
        for component_type in self.data:
            self.sorted_lengths[component_type] = sorted(self.data[component_type].keys())
    
    def get_harmonic_speed(
        self,
        component_type: str,
        input_length: int
    ) -> Optional[float]:
        """
        Get harmonic mean speed for a component at given input length.
        
        Uses nearest-neighbor matching if exact length not found.
        
        Args:
            component_type: Component type (e.g., "quantizer", "ans", "cachegen")
            input_length: Input sequence length (in tokens)
            
        Returns:
            Harmonic speed in GB/s, or None if component not found
        """
        if component_type not in self.data:
            return None
        
        # Find nearest input length
        nearest_length = self._find_nearest_length(component_type, input_length)
        if nearest_length is None:
            return None
        
        prefill_gbps, decode_gbps = self.data[component_type][nearest_length]
        
        # Harmonic mean: 2 / (1/prefill + 1/decode)
        if prefill_gbps > 0 and decode_gbps > 0:
            harmonic_speed = 2.0 / (1.0 / prefill_gbps + 1.0 / decode_gbps)
        elif prefill_gbps > 0:
            harmonic_speed = prefill_gbps
        elif decode_gbps > 0:
            harmonic_speed = decode_gbps
        else:
            return None
        
        # Convert GB/s to MB/s
        return harmonic_speed * 1024.0

    def get_component_throughput_mbps(
        self,
        component_type: str,
        input_length: int
    ) -> Optional[Tuple[float, float]]:
        """
        Get component throughput in MB/s (prefill, decode).

        Args:
            component_type: Component type (e.g., "quantizer", "ans")
            input_length: Input sequence length (in tokens)

        Returns:
            (prefill_mbps, decode_mbps) or None if component not found
        """
        throughput = self.get_throughput(component_type, input_length)
        if throughput is None:
            return None
        prefill_gbps, decode_gbps = throughput
        return (prefill_gbps * 1024.0, decode_gbps * 1024.0)
    
    def get_throughput(
        self,
        component_type: str,
        input_length: int
    ) -> Optional[Tuple[float, float]]:
        """
        Get raw throughput data (prefill, decode) in GB/s.
        
        Args:
            component_type: Component type
            input_length: Input sequence length
            
        Returns:
            (prefill_gbps, decode_gbps) or None if not found
        """
        if component_type not in self.data:
            return None
        
        nearest_length = self._find_nearest_length(component_type, input_length)
        if nearest_length is None:
            return None
        
        return self.data[component_type][nearest_length]
    
    def _find_nearest_length(self, component_type: str, target_length: int) -> Optional[int]:
        """
        Find nearest input length in data using binary search.
        
        Args:
            component_type: Component type
            target_length: Target input length
            
        Returns:
            Nearest available length, or None if component not found
        """
        if component_type not in self.sorted_lengths:
            return None
        
        lengths = self.sorted_lengths[component_type]
        if not lengths:
            return None
        
        # Binary search for insertion point
        idx = bisect.bisect_left(lengths, target_length)
        
        # Handle edge cases
        if idx == 0:
            return lengths[0]
        if idx == len(lengths):
            return lengths[-1]
        
        # Find nearest (left or right)
        left = lengths[idx - 1]
        right = lengths[idx]
        
        if abs(target_length - left) <= abs(target_length - right):
            return left
        else:
            return right
    
    def get_pipeline_harmonic_speed(
        self,
        pipeline_components: List[str],
        input_length: int
    ) -> Optional[float]:
        """
        Compute pipeline harmonic speed from multiple serial components.
        
        For serial pipeline: 1/S_total = 1/S1 + 1/S2 + ...
        
        Args:
            pipeline_components: List of component types in order
            input_length: Input sequence length
            
        Returns:
            Pipeline harmonic speed in MB/s, or None if any component not found
        """
        reciprocal_sum = 0.0
        
        for component in pipeline_components:
            speed = self.get_harmonic_speed(component, input_length)
            if speed is None:
                return None
            reciprocal_sum += 1.0 / speed
        
        return 1.0 / reciprocal_sum
    
    def get_available_components(self) -> List[str]:
        """Get list of available component types."""
        return list(self.data.keys())
    
    def get_length_range(self, component_type: str) -> Optional[Tuple[int, int]]:
        """Get min and max input lengths for a component."""
        if component_type not in self.sorted_lengths:
            return None
        lengths = self.sorted_lengths[component_type]
        if not lengths:
            return None
        return (lengths[0], lengths[-1])
    
    def __repr__(self) -> str:
        components = list(self.data.keys())
        return f"SpeedProfileLoader(components={components}, csv={Path(self.csv_path).name})"


class SpeedProfileCache:
    """
    Cache for multiple speed profiles (different machines/models).
    
    Manages loading and caching of speed profiles to avoid redundant I/O.
    """
    
    def __init__(self, speed_results_dir: str = "/root/lzd/kvserve_project/speed_results"):
        """
        Initialize speed profile cache.
        
        Args:
            speed_results_dir: Directory containing speed CSV files
        """
        self.speed_results_dir = Path(speed_results_dir)
        self._cache: Dict[Tuple[str, str], SpeedProfileLoader] = {}
    
    def get_loader(self, machine: str, model_name: str) -> SpeedProfileLoader:
        """
        Get speed profile loader for a specific machine and model.
        
        Args:
            machine: Machine name (e.g., "5090", "H100")
            model_name: Model name (e.g., "Qwen2.5-7B-Instruct")
            
        Returns:
            SpeedProfileLoader instance
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
        """
        cache_key = (machine, model_name)
        
        if cache_key not in self._cache:
            csv_path = self.speed_results_dir / f"{machine}_{model_name}_speed.csv"
            self._cache[cache_key] = SpeedProfileLoader(str(csv_path))
        
        return self._cache[cache_key]
    
    def clear_cache(self):
        """Clear all cached loaders."""
        self._cache.clear()
    
    def __repr__(self) -> str:
        return f"SpeedProfileCache(cached={len(self._cache)}, dir={self.speed_results_dir})"
