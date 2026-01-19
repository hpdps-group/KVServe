"""
Profile library management.

Loads pre-profiled compression configurations organized by:
- Accuracy buckets
- Bandwidth intervals (in 1/B space)
- Pareto frontier candidates
"""

import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from kvserve.controller.profile import Profile


class ProfileLibrary:
    """
    Manages a library of compression profiles organized for fast lookup.
    
    Structure:
        - Accuracy buckets: Divides quality space
        - Bandwidth intervals: Divides 1/B space (Theorem 2)
        - Candidate profiles: 2-3 profiles per (bucket, interval)
    """
    
    def __init__(self, json_path: str):
        """
        Load profile library from JSON file.
        
        Args:
            json_path: Path to profile library JSON file
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            ValueError: If JSON format is invalid
        """
        if not Path(json_path).exists():
            raise FileNotFoundError(f"Profile library not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Load metadata
        self.metadata = data.get('library_metadata', {})
        
        # Load accuracy buckets
        self.accuracy_buckets: List[Dict] = data.get('accuracy_buckets', [])
        
        # Build profile index for fast lookup
        self._build_profile_index()
        
        # Validate library structure
        self._validate()
    
    def _build_profile_index(self):
        """Build profile_id -> Profile mapping for fast access."""
        self.profile_map: Dict[str, Profile] = {}
        
        for bucket in self.accuracy_buckets:
            pareto_profiles = bucket.get('pareto_profiles', [])
            for profile_dict in pareto_profiles:
                profile = Profile.from_dict(profile_dict)
                self.profile_map[profile.profile_id] = profile
    
    def _validate(self):
        """Validate library structure."""
        if not self.accuracy_buckets:
            raise ValueError("Profile library has no accuracy buckets")
        
        for bucket in self.accuracy_buckets:
            if 'bucket_id' not in bucket:
                raise ValueError("Bucket missing 'bucket_id'")
            if 'acc_range' not in bucket:
                raise ValueError("Bucket missing 'acc_range'")
            if 'bandwidth_intervals' not in bucket:
                raise ValueError("Bucket missing 'bandwidth_intervals'")
    
    def find_bucket(self, acc_req: float) -> Optional[Dict]:
        """
        Find accuracy bucket containing the required accuracy.
        
        Args:
            acc_req: Required accuracy (e.g., 0.92)
            
        Returns:
            Bucket dictionary or None if no matching bucket
        """
        for bucket in self.accuracy_buckets:
            acc_lower, acc_upper = bucket['acc_range']
            if acc_lower <= acc_req < acc_upper:
                return bucket
        
        # If exact match not found, return closest bucket
        # (useful for acc_req at boundary)
        if self.accuracy_buckets:
            closest = min(
                self.accuracy_buckets,
                key=lambda b: min(
                    abs(b['acc_range'][0] - acc_req),
                    abs(b['acc_range'][1] - acc_req)
                )
            )
            return closest
        
        return None
    
    def find_interval(self, bucket: Dict, B_mbps: float) -> Optional[Dict]:
        """
        Find bandwidth interval for given bandwidth.
        
        Uses 1/B space partitioning (Theorem 2).
        
        Args:
            bucket: Accuracy bucket dictionary
            B_mbps: Network bandwidth in MB/s
            
        Returns:
            Interval dictionary or None if no matching interval
        """
        x = 1.0 / B_mbps  # Transform to 1/B space
        
        intervals = bucket.get('bandwidth_intervals', [])
        for interval in intervals:
            x_lower, x_upper = interval['x_range']
            if x_lower <= x < x_upper:
                return interval
        
        # Handle edge case: x >= max x_range (very low bandwidth)
        if intervals:
            # Return the last interval (lowest bandwidth range)
            return intervals[-1]
        
        return None
    
    def get_profile(self, profile_id: str) -> Optional[Profile]:
        """
        Get profile by ID.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Profile object or None if not found
        """
        return self.profile_map.get(profile_id)
    
    def get_candidate_profiles(
        self,
        bucket: Dict,
        interval: Dict
    ) -> List[Profile]:
        """
        Get candidate profiles for a (bucket, interval) pair.
        
        Args:
            bucket: Accuracy bucket dictionary
            interval: Bandwidth interval dictionary
            
        Returns:
            List of Profile objects (typically 2-3 profiles)
        """
        candidate_ids = interval.get('candidate_profile_ids', [])
        candidates = []
        
        for profile_id in candidate_ids:
            profile = self.get_profile(profile_id)
            if profile is not None:
                candidates.append(profile)
        
        return candidates
    
    def get_optimal_profile(self, bucket: Dict, interval: Dict) -> Optional[Profile]:
        """
        Get theoretically optimal profile for a (bucket, interval) pair.
        
        Args:
            bucket: Accuracy bucket dictionary
            interval: Bandwidth interval dictionary
            
        Returns:
            Profile object or None if not found
        """
        optimal_id = interval.get('model_optimal_profile_id')
        if optimal_id:
            return self.get_profile(optimal_id)
        return None
    
    def get_all_profiles(self) -> List[Profile]:
        """Get all profiles in the library."""
        return list(self.profile_map.values())
    
    def get_num_profiles(self) -> int:
        """Get total number of profiles."""
        return len(self.profile_map)
    
    def get_num_buckets(self) -> int:
        """Get number of accuracy buckets."""
        return len(self.accuracy_buckets)
    
    def get_bucket_info(self, bucket_id: int) -> Optional[Dict]:
        """Get bucket by ID."""
        for bucket in self.accuracy_buckets:
            if bucket['bucket_id'] == bucket_id:
                return bucket
        return None
    
    def summary(self) -> str:
        """
        Get library summary.
        
        Returns:
            Human-readable summary string
        """
        total_intervals = sum(
            len(bucket.get('bandwidth_intervals', []))
            for bucket in self.accuracy_buckets
        )
        
        total_candidates = sum(
            len(interval.get('candidate_profile_ids', []))
            for bucket in self.accuracy_buckets
            for interval in bucket.get('bandwidth_intervals', [])
        )
        
        return (
            f"ProfileLibrary(\n"
            f"  Model: {self.metadata.get('model_name', 'Unknown')}\n"
            f"  Dataset: {self.metadata.get('dataset', 'Unknown')}\n"
            f"  Profiles: {self.get_num_profiles()}\n"
            f"  Accuracy buckets: {self.get_num_buckets()}\n"
            f"  Bandwidth intervals: {total_intervals}\n"
            f"  Total candidates: {total_candidates}\n"
            f")"
        )
    
    def __repr__(self) -> str:
        return f"ProfileLibrary(profiles={self.get_num_profiles()}, buckets={self.get_num_buckets()})"



