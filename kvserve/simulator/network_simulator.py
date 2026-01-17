"""Network transfer simulator with bandwidth and concurrency limits"""

import time
import random
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TransferTask:
    """A single KV transfer task (all times in milliseconds)"""
    request_id: str
    size_bytes: int
    submit_time_ms: float  # When transfer was requested (t_p_end, in ms)
    
    # Filled by simulator (ms)
    queue_time_ms: float = 0.0  # Time spent waiting in queue
    transfer_time_ms: float = 0.0  # Actual transfer time
    complete_time_ms: float = 0.0  # When transfer completes (t_net_arrive)


class NetworkSimulator:
    """
    Simulates network transfer with:
    - Bandwidth limit (e.g., 80 Gbps PCIe, 200 Gbps IB)
    - Concurrency limit (e.g., 2 concurrent transfers)
    - Queueing when concurrent limit reached
    - Random jitter
    """
    
    def __init__(
        self,
        throughput_gbps: float = 80.0,  # PCIe Gen4 x16 ~ 80 Gbps effective
        max_concurrent: int = 2,  # Max concurrent transfers
        efficiency: float = 0.8,  # Effective bandwidth utilization
        jitter_ms: float = 1.0,  # Random jitter (±ms)
    ):
        self.throughput_gbps = throughput_gbps
        self.max_concurrent = max_concurrent
        self.efficiency = efficiency
        self.jitter_ms = jitter_ms
        
        # Active transfers: (complete_time_ms, task)
        self.active: List[Tuple[float, TransferTask]] = []
    
    def simulate_transfer(self, task: TransferTask) -> TransferTask:
        """
        Simulate a single transfer, considering queue and bandwidth.
        All time units are milliseconds.
        """
        current_time_ms = task.submit_time_ms
        
        # Remove completed transfers
        self.active = [(t, tk) for t, tk in self.active if t > current_time_ms]
        
        # If at capacity, wait for earliest to complete
        if len(self.active) >= self.max_concurrent:
            earliest_complete = min(t for t, _ in self.active)
            task.queue_time_ms = earliest_complete - current_time_ms
            current_time_ms = earliest_complete
            self.active = [(t, tk) for t, tk in self.active if t > current_time_ms]
        
        # Calculate transfer time (ms)
        # Note: size_gb is in GB (bytes), throughput_gbps is in Gbps (bits)
        # Convert GB to Gb: 1 GB = 8 Gb
        size_gb = task.size_bytes / (1024**3)  # Convert to GB
        size_gb_in_bits = size_gb * 8  # Convert GB to Gb
        base_time_ms = size_gb_in_bits / (self.throughput_gbps * self.efficiency) * 1000.0
        jitter_ms = random.uniform(-self.jitter_ms, self.jitter_ms)
        task.transfer_time_ms = max(1.0, base_time_ms + jitter_ms)  # At least 1ms
        
        # Complete time
        task.complete_time_ms = current_time_ms + task.transfer_time_ms
        
        # Add to active
        self.active.append((task.complete_time_ms, task))
        
        return task
    
    def reset(self):
        """Reset simulator state"""
        self.active.clear()




