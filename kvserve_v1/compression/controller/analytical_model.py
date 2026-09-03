"""Analytical cost model for compression latency.

    T_p = T_model + V/S + V/(B * cr)
    T_0 = T_model + V/B
    B*  = (1 - 1/cr) * S     # compression helps iff B < B*
"""

from kvserve_v1.compression.controller.profile import Profile

# SI: 1 GB/s = 1000 MB/s. 5 Gbps = 625 MB/s.
GB_TO_MB = 1000.0


def gbps_to_mbps(gbps: float) -> float:
    return gbps * 1000.0 / 8.0


def mbps_to_gbps(mbps: float) -> float:
    return mbps * 8.0 / 1000.0


class AnalyticalModel:
    """Theorem 1 (B*) and the paper latency equations. V in bytes, B/S in MB/s, T in ms."""

    @staticmethod
    def t0_ms(V_bytes: float, B_mbps: float, T_model_ms: float) -> float:
        V_mb = V_bytes / (1024 * 1024)
        return T_model_ms + (V_mb / B_mbps) * 1000.0

    @staticmethod
    def tp_ms(
        V_bytes: float,
        B_mbps: float,
        T_model_ms: float,
        cr: float,
        S_mbps: float,
    ) -> float:
        V_mb = V_bytes / (1024 * 1024)
        t_codec = (V_mb / S_mbps) * 1000.0
        t_xfer = (V_mb / (B_mbps * cr)) * 1000.0
        return T_model_ms + t_codec + t_xfer

    @staticmethod
    def b_star_mbps(cr: float, S_mbps: float) -> float:
        return (1.0 - 1.0 / cr) * S_mbps

    @classmethod
    def benefits(cls, cr: float, S_mbps: float, B_mbps: float) -> bool:
        return B_mbps < cls.b_star_mbps(cr, S_mbps)

    @staticmethod
    def predict_latency(
        profile: Profile,
        V_bytes: float,
        B_mbps: float,
        T_model_ms: float,
    ) -> float:
        return AnalyticalModel.tp_ms(
            V_bytes, B_mbps, T_model_ms,
            profile.compression_ratio, profile.harmonic_speed,
        )

    @staticmethod
    def predict_baseline_latency(V_bytes: float, B_mbps: float, T_model_ms: float) -> float:
        return AnalyticalModel.t0_ms(V_bytes, B_mbps, T_model_ms)

    @staticmethod
    def check_benefit(profile: Profile, B_mbps: float) -> bool:
        return AnalyticalModel.benefits(
            profile.compression_ratio, profile.harmonic_speed, B_mbps,
        )

    @staticmethod
    def compute_speedup(
        profile: Profile,
        V_bytes: float,
        B_mbps: float,
        T_model_ms: float,
    ) -> float:
        T_0 = AnalyticalModel.predict_baseline_latency(V_bytes, B_mbps, T_model_ms)
        T_p = AnalyticalModel.predict_latency(profile, V_bytes, B_mbps, T_model_ms)
        return T_0 / T_p if T_p > 0 else 1.0
