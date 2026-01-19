"""
Analytical cost model for predicting compression latency.

Based on theoretical analysis:
- T_p = T_model + V/S + V/(B*cr)
- T_0 = T_model + V/B

Where:
- T_model: Model computation latency
- V: KV cache volume
- S: Harmonic speed (encode/decode)
- B: Network bandwidth
- cr: Compression ratio
"""

from kvserve.controller.profile import Profile


class AnalyticalModel:
    """
    Analytical model for predicting end-to-end latency with compression.
    
    Implements Theorem 1 (benefit condition) and Theorem 2 (piecewise-optimal).
    """
    
    @staticmethod
    def predict_latency(
        profile: Profile,
        V_bytes: float,
        B_mbps: float,
        T_model_ms: float
    ) -> float:
        """
        Predict end-to-end latency using analytical model.
        
        Formula:
            T_p = T_model + V/S + V/(B*cr)
        
        Args:
            profile: Compression profile
            V_bytes: KV cache volume in bytes
            B_mbps: Network bandwidth in MB/s
            T_model_ms: Model computation latency in milliseconds
            
        Returns:
            T_hat: Predicted latency in milliseconds
        """
        # Convert bytes to MB
        V_mb = V_bytes / (1024 * 1024)
        
        # Get profile metrics
        S_mbps = profile.harmonic_speed
        cr = profile.compression_ratio
        
        # T_codec = T_compress + T_decompress = V/S
        T_codec_ms = (V_mb / S_mbps) * 1000
        
        # T_transfer = (compressed_volume) / B = (V*cr) / B
        T_transfer_ms = (V_mb * cr / B_mbps) * 1000
        
        # Total latency
        T_hat = T_model_ms + T_codec_ms + T_transfer_ms
        
        return T_hat
    
    @staticmethod
    def predict_baseline_latency(V_bytes: float, B_mbps: float, T_model_ms: float) -> float:
        """
        Predict baseline latency without compression.
        
        Formula:
            T_0 = T_model + V/B
        
        Args:
            V_bytes: KV cache volume in bytes
            B_mbps: Network bandwidth in MB/s
            T_model_ms: Model computation latency in milliseconds
            
        Returns:
            T_0: Baseline latency in milliseconds
        """
        V_mb = V_bytes / (1024 * 1024)
        T_transfer_ms = (V_mb / B_mbps) * 1000
        T_0 = T_model_ms + T_transfer_ms
        return T_0
    
    @staticmethod
    def check_benefit(profile: Profile, B_mbps: float) -> bool:
        """
        Theorem 1: Check if compression brings speedup at given bandwidth.
        
        Benefit condition:
            T_p < T_0
            => V/S + V/(B*cr) < V/B
            => 1/S < (cr-1)/(B*cr)
            => B < (1 - 1/cr) * S
        
        Args:
            profile: Compression profile
            B_mbps: Network bandwidth in MB/s
            
        Returns:
            True if compression accelerates, False otherwise
        """
        cr = profile.compression_ratio
        S = profile.harmonic_speed
        
        # Critical bandwidth: B_crit = (1 - 1/cr) * S
        B_crit = (1 - 1/cr) * S
        
        # Compression benefits when B < B_crit
        return B_mbps < B_crit
    
    @staticmethod
    def compute_speedup(
        profile: Profile,
        V_bytes: float,
        B_mbps: float,
        T_model_ms: float
    ) -> float:
        """
        Compute speedup ratio: T_0 / T_p
        
        Args:
            profile: Compression profile
            V_bytes: KV cache volume
            B_mbps: Network bandwidth
            T_model_ms: Model computation latency
            
        Returns:
            Speedup ratio (>1 means acceleration, <1 means slowdown)
        """
        T_0 = AnalyticalModel.predict_baseline_latency(V_bytes, B_mbps, T_model_ms)
        T_p = AnalyticalModel.predict_latency(profile, V_bytes, B_mbps, T_model_ms)
        return T_0 / T_p



