"""
Lightweight pipeline simulator for Prefill → (simulated KV transfer) → Decode.

Goals:
- No heavy integration; pure Python computation (no sleeping).
- Model timeline events: t0 (arrival), t_p_end (prefill done),
  t_net_arrive (KV available at decode), t_d_end (decode done).
- Network transfer model: limited concurrent transfers, bandwidth with headroom,
  small random jitter.
- IO model: CPU→GPU copy time estimated from size / PCIe_bw.

Usage:
    from kvserve.simulator.pipeline_simulator import (
        RequestSpec, PipelineSimulator, NetworkSimulator, IOSpec
    )

    requests = [
        RequestSpec(
            request_id="req1",
            arrival_ms=0.0,
            kv_size_bytes=128 * 1024 * 1024,  # 128MB
            prefill_compute_ms=120.0,
            decode_compute_ms=200.0,
        ),
        ...
    ]

    net = NetworkSimulator(throughput_gbps=80.0, max_concurrent=2, jitter_frac=0.05)
    io = IOSpec(pcie_gbps=24.0)  # typical PCIe4 x16 ~ 24-28 GB/s

    sim = PipelineSimulator(net_sim=net, io_spec=io)
    results = sim.run(requests)

Outputs per request:
    {
        "request_id": str,
        "t0": float,
        "t_p_end": float,
        "t_net_start": float,
        "t_net_end": float,
        "t_net_arrive": float,
        "t_d_start": float,
        "t_d_end": float,
        "prefill_compute_ms": float,
        "decode_compute_ms": float,
        "transfer_ms": float,
        "queue_ms": float,
        "io_ms": float,
        "total_latency_ms": float,
    }
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


# ------------------------------
# Data models
# ------------------------------

@dataclass
class RequestSpec:
    request_id: str
    arrival_ms: float
    kv_size_bytes: int
    prefill_compute_ms: float
    decode_compute_ms: float


@dataclass
class IOSpec:
    # PCIe bandwidth in GB/s (effective). Default ~24 GB/s (PCIe4 x16).
    pcie_gbps: float = 24.0
    jitter_frac: float = 0.02  # ±2%

    def io_time_ms(self, size_bytes: int) -> float:
        gb = size_bytes / (1024 ** 3)
        base = gb / self.pcie_gbps * 1000.0
        jitter = 1.0 + random.uniform(-self.jitter_frac, self.jitter_frac)
        return base * jitter


# ------------------------------
# Network simulator
# ------------------------------

class NetworkSimulator:
    """
    Simple network/PCIe simulator:
    - throughput_gbps: link bandwidth
    - max_concurrent: how many transfers can run in parallel
    - jitter_frac: multiplicative jitter on transfer time
    """

    def __init__(
        self,
        throughput_gbps: float = 80.0,
        max_concurrent: int = 2,
        jitter_frac: float = 0.05,
    ):
        self.throughput_gbps = throughput_gbps
        self.max_concurrent = max_concurrent
        self.jitter_frac = jitter_frac
        # track (end_time_ms) of inflight transfers
        self._inflight: List[float] = []

    def _cleanup(self, now_ms: float):
        self._inflight = [t for t in self._inflight if t > now_ms]

    def schedule(self, ready_ms: float, size_bytes: int) -> Dict[str, float]:
        """
        Schedule a transfer starting no earlier than ready_ms.
        Returns dict with net_start_ms, net_end_ms, queue_ms, transfer_ms.
        """
        self._cleanup(ready_ms)

        # If inflight < max_concurrent, can start at ready_ms
        if len(self._inflight) < self.max_concurrent:
            start_ms = ready_ms
        else:
            # Wait until the earliest inflight finishes
            earliest_done = min(self._inflight)
            start_ms = max(ready_ms, earliest_done)
            self._cleanup(start_ms)

        # transfer time with jitter
        gb = size_bytes / (1024 ** 3)
        base_ms = gb / self.throughput_gbps * 1000.0
        transfer_ms = base_ms * (1.0 + random.uniform(-self.jitter_frac, self.jitter_frac))
        end_ms = start_ms + transfer_ms

        self._inflight.append(end_ms)

        return {
            "net_start_ms": start_ms,
            "net_end_ms": end_ms,
            "queue_ms": max(0.0, start_ms - ready_ms),
            "transfer_ms": transfer_ms,
        }


# ------------------------------
# Pipeline simulator
# ------------------------------

class PipelineSimulator:
    def __init__(
        self,
        net_sim: NetworkSimulator,
        io_spec: IOSpec = IOSpec(),
    ):
        self.net_sim = net_sim
        self.io_spec = io_spec

    def run(self, requests: List[RequestSpec]) -> List[Dict[str, Any]]:
        """
        Run simulation for a batch of requests.
        Returns list of per-request dicts with timeline metrics.
        """
        results: List[Dict[str, Any]] = []
        # Sort by arrival time to ensure deterministic scheduling
        requests_sorted = sorted(requests, key=lambda r: r.arrival_ms)

        for req in requests_sorted:
            t0 = req.arrival_ms

            # Prefill compute
            t_p_end = t0 + req.prefill_compute_ms

            # Network transfer (simulate KV movement)
            net_info = self.net_sim.schedule(t_p_end, req.kv_size_bytes)
            t_net_arrive = net_info["net_end_ms"]

            # IO (CPU->GPU) cost before decode
            io_ms = self.io_spec.io_time_ms(req.kv_size_bytes)
            t_d_start = t_net_arrive + io_ms

            # Decode compute
            t_d_end = t_d_start + req.decode_compute_ms

            results.append({
                "request_id": req.request_id,
                "t0": t0,
                "t_p_end": t_p_end,
                "t_net_start": net_info["net_start_ms"],
                "t_net_end": net_info["net_end_ms"],
                "t_net_arrive": t_net_arrive,
                "t_d_start": t_d_start,
                "t_d_end": t_d_end,
                "prefill_compute_ms": req.prefill_compute_ms,
                "decode_compute_ms": req.decode_compute_ms,
                "transfer_ms": net_info["transfer_ms"],
                "queue_ms": net_info["queue_ms"],
                "io_ms": io_ms,
                "total_latency_ms": t_d_end - t0,
            })

        return results


# ------------------------------
# Convenience helper to build RequestSpec from measured data
# ------------------------------

def build_requests_from_measurements(
    ids: List[str],
    arrivals_ms: List[float],
    kv_sizes: List[int],
    prefill_times_ms: List[float],
    decode_times_ms: List[float],
) -> List[RequestSpec]:
    assert len({len(ids), len(arrivals_ms), len(kv_sizes), len(prefill_times_ms), len(decode_times_ms)}) == 1
    return [
        RequestSpec(
            request_id=i,
            arrival_ms=arrivals_ms[idx],
            kv_size_bytes=kv_sizes[idx],
            prefill_compute_ms=prefill_times_ms[idx],
            decode_compute_ms=decode_times_ms[idx],
        )
        for idx, i in enumerate(ids)
    ]


