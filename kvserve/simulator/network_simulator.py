"""Network transfer simulator with bandwidth and concurrency limits"""

from dataclasses import dataclass
from typing import List, Tuple
import random
import time


@dataclass
class TransferTask:
    """A single KV transfer task (all times in milliseconds)"""
    request_id: str
    size_bytes: int
    submit_time_ms: float
    queue_time_ms: float = 0.0
    transfer_time_ms: float = 0.0
    complete_time_ms: float = 0.0


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
        throughput_gbps: float,
        max_concurrent: int = 2,
        efficiency: float = 0.9,
        jitter_ms: float = 0.0
    ):
        self.throughput_gbps = throughput_gbps
        self.max_concurrent = max_concurrent
        self.efficiency = efficiency
        self.jitter_ms = jitter_ms
        self.active: List[TransferTask] = []
    
    def simulate_transfer(self, task: TransferTask) -> TransferTask:
        """
        Simulate a single transfer, considering queue and bandwidth.
        All time units are milliseconds.
        """
        current_time_ms = task.submit_time_ms
        
        # If we're at capacity, wait for earliest to complete
        if len(self.active) >= self.max_concurrent:
            earliest_complete = min((t.complete_time_ms for t in self.active))
            current_time_ms = max(current_time_ms, earliest_complete)
            # Remove completed transfers
            self.active = [t for t in self.active if t.complete_time_ms > current_time_ms]
        
        task.queue_time_ms = current_time_ms - task.submit_time_ms
        
        # Calculate transfer time: bytes -> GB -> bits -> seconds -> ms
        size_gb = task.size_bytes / (1024**3)
        size_gb_in_bits = size_gb * 8
        base_time_ms = (size_gb_in_bits / self.throughput_gbps * self.efficiency) * 1000.0
        
        # Add jitter
        if self.jitter_ms > 0:
            base_time_ms += random.uniform(-self.jitter_ms, self.jitter_ms)
        
        task.transfer_time_ms = max(0.0, base_time_ms)
        task.complete_time_ms = current_time_ms + task.transfer_time_ms
        
        self.active.append(task)
        return task
    
    def reset(self):
        """Reset simulator state"""
        self.active.clear()

