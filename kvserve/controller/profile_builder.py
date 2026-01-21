"""
Profile builder for constructing complete Profile objects from raw configs.

Combines:
- Raw compression configs from profiles/{model}/{dataset}.json
- Throughput data from speed_results/{machine}_{model}_speed.csv
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from kvserve.controller.profile import Profile
from kvserve.controller.speed_profile_loader import SpeedProfileLoader


class ProfileBuilder:
    """
    Builds complete Profile objects from raw compression configs and speed data.
    
    Workflow:
    1. Load raw configs from profiles/{model}/{dataset}.json
    2. Map config components to speed CSV component types
    3. Query throughput data from speed CSV
    4. Compute performance metrics (harmonic_speed, critical_bandwidth)
    5. Build Profile objects
    """
    
    # Component type mapping
    QUANTIZER_SPLIT_TYPE_MAP = {
        "head": "quantizer",
        "layer": "cachegen"
    }
    
    CODEC_TYPE_MAP = {
        "ANS": "ans",
        "ans": "ans",
        "BitComp": "bitcomp",
        "bitcomp": "bitcomp",
        "LZ4": "lz4",
        "lz4": "lz4"
    }
    
    TRANSFORMER_TYPE_MAP = {
        "Hadamard": "hadamard",
        "hadamard": "hadamard",
        "none": None,
        None: None
    }
    
    def __init__(
        self,
        model_name: str,
        dataset: str,
        prefill_machine: str,
        input_length: int,
        decode_machine: Optional[str] = None,
        profiles_dir: str = "/root/lzd/kvserve_project/profiles",
        speed_results_dir: str = "/root/lzd/kvserve_project/speed_results"
    ):
        """
        Initialize profile builder.
        
        Args:
            model_name: Model name (e.g., "Qwen2.5-7B-Instruct")
            dataset: Dataset name (e.g., "qasper")
            machine: Machine name (e.g., "5090", "H100")
            input_length: Input sequence length (for throughput lookup)
            profiles_dir: Directory containing profile JSON files
            speed_results_dir: Directory containing speed CSV files
        """
        self.model_name = model_name
        self.dataset = dataset
        self.prefill_machine = prefill_machine
        self.decode_machine = decode_machine or prefill_machine
        self.input_length = input_length
        
        # Load raw configs
        config_path = Path(profiles_dir) / model_name / f"{dataset}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Profile config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.raw_configs: List[Dict[str, Any]] = json.load(f)
        
        # Load speed profile
        prefill_csv = Path(speed_results_dir) / f"{self.prefill_machine}_{model_name}_speed.csv"
        if not prefill_csv.exists():
            raise FileNotFoundError(f"Speed profile not found: {prefill_csv}")
        
        self.prefill_speed_loader = SpeedProfileLoader(str(prefill_csv))
        
        if self.decode_machine == self.prefill_machine:
            self.decode_speed_loader = self.prefill_speed_loader
        else:
            decode_csv = Path(speed_results_dir) / f"{self.decode_machine}_{model_name}_speed.csv"
            if not decode_csv.exists():
                raise FileNotFoundError(f"Speed profile not found: {decode_csv}")
            self.decode_speed_loader = SpeedProfileLoader(str(decode_csv))
    
    def build_all_profiles(self) -> List[Profile]:
        """
        Build all Profile objects from raw configs.
        
        Returns:
            List of Profile objects with complete performance metrics
        """
        profiles = []
        
        for raw_config in self.raw_configs:
            try:
                profile = self.build_profile(raw_config)
                profiles.append(profile)
            except Exception as e:
                # Log warning but continue
                config_id = raw_config.get('config_id', 'unknown')
                print(f"Warning: Failed to build profile {config_id}: {e}")
        
        return profiles
    
    def build_profile(self, raw_config: Dict[str, Any]) -> Profile:
        """
        Build a single Profile from raw config.
        
        Args:
            raw_config: Raw config dictionary from JSON
            
        Returns:
            Complete Profile object
        """
        config_id = raw_config['config_id']
        
        # Generate profile ID
        profile_id = f"{self.model_name}_{self.dataset}_config{config_id}"
        
        # Extract compression config components
        transformer_config = raw_config.get('transformer_config', {})
        quantizer_config = raw_config.get('quantizer_config', {})
        codec_config = raw_config.get('codec_config', {})
        
        # Build pipeline component list
        pipeline_components = self._build_pipeline_components(
            transformer_config,
            quantizer_config,
            codec_config
        )
        
        # Query harmonic speed from CSV
        harmonic_speed_mbps = self._get_pipeline_harmonic_speed(
            pipeline_components,
            self.input_length
        )
        
        if harmonic_speed_mbps is None:
            raise ValueError(
                f"Failed to get harmonic speed for pipeline {pipeline_components} "
                f"at input_length={self.input_length}"
            )
        
        # Extract quality metrics
        accuracy = raw_config.get('accuracy', 100.0)
        compression_ratio = raw_config.get('compression_ratio', 1.0)
        
        # Compute critical bandwidth
        critical_bandwidth_mbps = (1.0 - 1.0 / compression_ratio) * harmonic_speed_mbps
        
        # Build compression config for CompressionManager
        compression_config = self._build_compression_config(
            transformer_config,
            quantizer_config,
            codec_config
        )
        
        # Build performance metrics
        performance_metrics = {
            'compression_ratio': compression_ratio,
            'harmonic_speed_mbps': harmonic_speed_mbps,
            'critical_bandwidth_mbps': critical_bandwidth_mbps,
            'pipeline_components': pipeline_components,
            'input_length': self.input_length
        }
        
        # Build quality metrics
        quality_metrics = {
            'accuracy': accuracy,
            'metric_name': 'accuracy'  # Or could be ROUGE-L, etc.
        }
        
        # Build metadata
        metadata = {
            'model_name': self.model_name,
            'dataset': self.dataset,
            'prefill_machine': self.prefill_machine,
            'decode_machine': self.decode_machine,
            'config_id': config_id
        }
        
        return Profile(
            profile_id=profile_id,
            metadata=metadata,
            compression_config=compression_config,
            performance_metrics=performance_metrics,
            quality_metrics=quality_metrics
        )
    
    def _build_pipeline_components(
        self,
        transformer_config: Dict,
        quantizer_config: Dict,
        codec_config: Dict
    ) -> List[str]:
        """
        Build pipeline component list for speed lookup.
        
        Args:
            transformer_config: Transformer config
            quantizer_config: Quantizer config
            codec_config: Codec config
            
        Returns:
            List of component names (e.g., ["quantizer", "ans"])
        """
        components = []
        
        # Transformer (if enabled)
        transform_type = transformer_config.get('transform_type')
        if transform_type and transform_type != 'none':
            mapped_type = self.TRANSFORMER_TYPE_MAP.get(transform_type)
            if mapped_type:
                components.append(mapped_type)
        
        # Quantizer (if present)
        if quantizer_config:
            split_type = quantizer_config.get('split_type', 'head')
            mapped_type = self.QUANTIZER_SPLIT_TYPE_MAP.get(split_type, 'quantizer')
            components.append(mapped_type)
        
        # Codec (if present)
        if codec_config:
            codec_type = codec_config.get('nvcomp_algorithm') or codec_config.get('codec_type')
            if codec_type:
                mapped_type = self.CODEC_TYPE_MAP.get(codec_type)
                if mapped_type:
                    components.append(mapped_type)
        
        return components
    
    def _build_compression_config(
        self,
        transformer_config: Dict,
        quantizer_config: Dict,
        codec_config: Dict
    ) -> Dict[str, Any]:
        """
        Build compression config for CompressionManager.
        
        Args:
            transformer_config: Transformer config
            quantizer_config: Quantizer config
            codec_config: Codec config
            
        Returns:
            Compression config dictionary
        """
        # Build pipeline list
        pipeline = []
        if transformer_config and transformer_config.get('transform_type') != 'none':
            pipeline.append('transformer')
        if quantizer_config:
            pipeline.append('quantizer')
        if codec_config:
            pipeline.append('codec')
        
        transformer_config = self._normalize_transformer_config(transformer_config)

        return {
            'enabled': True,
            'pipeline': pipeline,
            'transformer_config': transformer_config if transformer_config else None,
            'quantizer_config': quantizer_config if quantizer_config else None,
            'codec_config': codec_config if codec_config else None
        }

    def _normalize_transformer_config(self, transformer_config: Dict) -> Dict:
        """
        Normalize transformer config values (e.g., seed types).

        Ensures Hadamard seed is int even if provided as string.
        """
        if not transformer_config:
            return transformer_config
        normalized = dict(transformer_config)
        seed = normalized.get("seed")
        if isinstance(seed, str):
            try:
                normalized["seed"] = int(seed, 0)
            except ValueError:
                # Fall back to default if malformed
                normalized["seed"] = 0xC0FEBABE
        return normalized

    def _get_component_harmonic_speed(
        self,
        component_type: str,
        input_length: int
    ) -> Optional[float]:
        """
        Compute component harmonic speed using prefill/decode machines.

        Args:
            component_type: Component type
            input_length: Input length

        Returns:
            Harmonic speed in MB/s
        """
        prefill_throughput = self.prefill_speed_loader.get_component_throughput_mbps(
            component_type, input_length
        )
        decode_throughput = self.decode_speed_loader.get_component_throughput_mbps(
            component_type, input_length
        )
        if prefill_throughput is None or decode_throughput is None:
            return None
        prefill_mbps = prefill_throughput[0]
        decode_mbps = decode_throughput[1]
        if prefill_mbps > 0 and decode_mbps > 0:
            return 2.0 / (1.0 / prefill_mbps + 1.0 / decode_mbps)
        if prefill_mbps > 0:
            return prefill_mbps
        if decode_mbps > 0:
            return decode_mbps
        return None

    def _get_pipeline_harmonic_speed(
        self,
        pipeline_components: List[str],
        input_length: int
    ) -> Optional[float]:
        """
        Compute pipeline harmonic speed from serial components.
        """
        reciprocal_sum = 0.0
        for component in pipeline_components:
            speed = self._get_component_harmonic_speed(component, input_length)
            if speed is None:
                return None
            reciprocal_sum += 1.0 / speed
        return 1.0 / reciprocal_sum
    
    def get_num_configs(self) -> int:
        """Get total number of raw configs."""
        return len(self.raw_configs)
    
    def __repr__(self) -> str:
        return (
            f"ProfileBuilder(model={self.model_name}, dataset={self.dataset}, "
            f"prefill_machine={self.prefill_machine}, decode_machine={self.decode_machine}, "
            f"configs={len(self.raw_configs)})"
        )
