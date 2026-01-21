"""
Dynamic profile library that builds from raw configs and speed data.

Replaces the static ProfileLibrary with a dynamic version that:
- Loads raw configs from profiles/{model}/{dataset}.json
- Queries throughput from speed_results/{machine}_{model}_speed.csv
- Builds profiles on-the-fly based on input length
"""

from typing import Dict, List, Optional
from pathlib import Path

from kvserve.controller.profile import Profile
from kvserve.controller.profile_builder import ProfileBuilder


class DynamicProfileLibrary:
    """
    Dynamic profile library for online compression profile selection.
    
    Features:
    - Loads Pareto-optimal configs from profiles directory
    - Queries throughput dynamically based on input length
    - Fast accuracy-based filtering
    - Minimal memory footprint
    """
    
    def __init__(
        self,
        model_name: str,
        dataset: str,
        prefill_machine: str,
        decode_machine: Optional[str] = None,
        profiles_dir: str = "/root/lzd/kvserve_project/profiles",
        speed_results_dir: str = "/root/lzd/kvserve_project/speed_results"
    ):
        """
        Initialize dynamic profile library.
        
        Args:
            model_name: Model name (e.g., "Qwen2.5-7B-Instruct")
            dataset: Dataset name (e.g., "qasper")
            machine: Machine name (e.g., "5090", "H100")
            profiles_dir: Directory containing profile JSON files
            speed_results_dir: Directory containing speed CSV files
        """
        self.model_name = model_name
        self.dataset = dataset
        self.prefill_machine = prefill_machine
        self.decode_machine = decode_machine or prefill_machine
        self.profiles_dir = profiles_dir
        self.speed_results_dir = speed_results_dir
        
        # Cache for built profiles at different input lengths
        # {input_length: List[Profile]}
        self._profile_cache: Dict[int, List[Profile]] = {}
    
    def get_profiles(self, input_length: int) -> List[Profile]:
        """
        Get all profiles for a given input length.
        
        Profiles are built on-demand and cached.
        
        Args:
            input_length: Input sequence length (in tokens)
            
        Returns:
            List of Profile objects
        """
        if input_length not in self._profile_cache:
            # Build profiles for this input length
            builder = ProfileBuilder(
                model_name=self.model_name,
                dataset=self.dataset,
                prefill_machine=self.prefill_machine,
                decode_machine=self.decode_machine,
                input_length=input_length,
                profiles_dir=self.profiles_dir,
                speed_results_dir=self.speed_results_dir
            )
            
            profiles = builder.build_all_profiles()
            self._profile_cache[input_length] = profiles
        
        return self._profile_cache[input_length]
    
    def filter_by_accuracy(
        self,
        profiles: List[Profile],
        acc_req: float
    ) -> List[Profile]:
        """
        Filter profiles by accuracy requirement.
        
        Args:
            profiles: List of profiles to filter
            acc_req: Minimum required accuracy
            
        Returns:
            Filtered list of profiles meeting accuracy requirement
        """
        return [p for p in profiles if p.accuracy >= acc_req]
    
    def filter_by_benefit(
        self,
        profiles: List[Profile],
        B_mbps: float
    ) -> List[Profile]:
        """
        Filter profiles by benefit condition (Theorem 1).
        
        Compression benefits when: B < B_crit = (1 - 1/cr) * S
        
        Args:
            profiles: List of profiles to filter
            B_mbps: Network bandwidth in MB/s
            
        Returns:
            Filtered list of profiles that benefit from compression
        """
        return [p for p in profiles if B_mbps < p.critical_bandwidth]
    
    def select_top_candidates(
        self,
        profiles: List[Profile],
        max_candidates: int = 3
    ) -> List[Profile]:
        """
        Select top candidates based on compression ratio.
        
        Higher compression ratio = better for bandwidth-constrained scenarios.
        
        Args:
            profiles: List of profiles
            max_candidates: Maximum number of candidates to return
            
        Returns:
            Top candidates (up to max_candidates)
        """
        # Sort by compression ratio (descending)
        sorted_profiles = sorted(
            profiles,
            key=lambda p: p.compression_ratio,
            reverse=True
        )
        
        return sorted_profiles[:max_candidates]
    
    def get_candidate_profiles(
        self,
        input_length: int,
        acc_req: float,
        B_mbps: float,
        max_candidates: int = 3
    ) -> List[Profile]:
        """
        Get candidate profiles for online selection.
        
        Workflow:
        1. Load profiles for input_length
        2. Filter by accuracy requirement
        3. Filter by benefit condition
        4. Select top candidates
        
        Args:
            input_length: Input sequence length
            acc_req: Minimum required accuracy
            B_mbps: Network bandwidth in MB/s
            max_candidates: Maximum number of candidates
            
        Returns:
            List of candidate profiles (≤ max_candidates)
        """
        # Get all profiles for this input length
        all_profiles = self.get_profiles(input_length)
        
        # Filter by accuracy
        acc_filtered = self.filter_by_accuracy(all_profiles, acc_req)
        
        # Filter by benefit condition
        benefit_filtered = self.filter_by_benefit(acc_filtered, B_mbps)
        
        # Select top candidates
        candidates = self.select_top_candidates(benefit_filtered, max_candidates)
        
        return candidates
    
    def get_profile_by_id(self, profile_id: str, input_length: int) -> Optional[Profile]:
        """
        Get a specific profile by ID.
        
        Args:
            profile_id: Profile identifier
            input_length: Input sequence length
            
        Returns:
            Profile object or None if not found
        """
        profiles = self.get_profiles(input_length)
        for profile in profiles:
            if profile.profile_id == profile_id:
                return profile
        return None
    
    def clear_cache(self):
        """Clear profile cache to free memory."""
        self._profile_cache.clear()
    
    def get_cache_size(self) -> int:
        """Get number of cached input lengths."""
        return len(self._profile_cache)
    
    def summary(self) -> str:
        """
        Get library summary.
        
        Returns:
            Human-readable summary string
        """
        # Count total unique profiles (use a representative input length)
        sample_length = 4096
        try:
            profiles = self.get_profiles(sample_length)
            num_profiles = len(profiles)
        except:
            num_profiles = 0
        
        return (
            f"DynamicProfileLibrary(\n"
            f"  Model: {self.model_name}\n"
            f"  Dataset: {self.dataset}\n"
            f"  Prefill machine: {self.prefill_machine}\n"
            f"  Decode machine: {self.decode_machine}\n"
            f"  Profiles: {num_profiles}\n"
            f"  Cached lengths: {self.get_cache_size()}\n"
            f")"
        )
    
    def __repr__(self) -> str:
        return (
            f"DynamicProfileLibrary(model={self.model_name}, "
            f"dataset={self.dataset}, prefill_machine={self.prefill_machine}, "
            f"decode_machine={self.decode_machine})"
        )
