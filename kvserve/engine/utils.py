"""
Utility classes and functions for PD separation engine
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import asyncio
import socket


def pick_free_port(start_port: int, max_tries: int = 50) -> int:
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    return start_port


class EngineStage(Enum):
    """Engine stage enumeration"""
    PREFILL = "prefill"
    DECODING = "decoding"


class EngineStatus(Enum):
    """Engine status enumeration"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    MIGRATING = "migrating"
    ERROR = "error"


class KVTransferStatus(Enum):
    """KV cache transfer status for multi-stream optimization"""
    NOT_STARTED = "not_started"      # Not yet started (in prefill stage)
    PENDING = "pending"              # Waiting to transfer
    TRANSFERRING = "transferring"   # Transfer in progress
    READY = "ready"                  # Transfer complete, ready for compute
    FAILED = "failed"                # Transfer failed


@dataclass
class Request:
    """Request representation for PD separation engine"""
    request_id: str
    prompt: str
    max_tokens: Optional[int] = None  # None means no limit
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 means disabled
    do_sample: bool = True
    stop: Optional[List[str]] = None  # Stop sequences
    
    # Sequence management
    prompt_token_ids: Optional[List[int]] = None
    output_token_ids: List[int] = None
    
    # Status tracking
    is_finished: bool = False
    finish_reason: Optional[str] = None
    
    # Timing information (for profiling)
    arrival_time: Optional[float] = None
    prefill_start_time: Optional[float] = None
    prefill_end_time: Optional[float] = None
    decoding_start_time: Optional[float] = None
    decoding_end_time: Optional[float] = None
    total_decode_compute_time: float = 0.0
    kv_transfer_time: Optional[float] = None
    
    # Multi-stream KV transfer (for async transfer optimization)
    kv_transfer_status: KVTransferStatus = field(default=KVTransferStatus.NOT_STARTED)
    kv_transfer_task: Optional[asyncio.Task] = field(default=None, repr=False, compare=False)  # Async transfer task (not serializable)
    kv_transfer_complete: bool = False  # Whether transfer is complete (synced in worker)
    kv_transfer_retry_count: int = 0  # Transfer retry counter
    assigned_worker_rank: Optional[int] = None  # Assigned decode worker rank
    
    def __post_init__(self):
        if self.output_token_ids is None:
            self.output_token_ids = []
        if self.arrival_time is None:
            self.arrival_time = time.time()
    
    def __getstate__(self):
        """Custom serialization: exclude non-serializable fields"""
        state = self.__dict__.copy()
        # Remove asyncio.Task which cannot be pickled
        state['kv_transfer_task'] = None
        return state
    
    def __setstate__(self, state):
        """Custom deserialization"""
        self.__dict__.update(state)
    
    def get_input_len(self) -> int:
        """Get input sequence length"""
        if self.prompt_token_ids:
            return len(self.prompt_token_ids)
        return 0
    
    def get_output_len(self) -> int:
        """Get output sequence length"""
        return len(self.output_token_ids)
    
    def get_total_len(self) -> int:
        """Get total sequence length"""
        return self.get_input_len() + self.get_output_len()
    
    def append_token(self, token_id: int):
        """Append a new token to output"""
        self.output_token_ids.append(token_id)
    
    def check_finished(self, eos_token_id: int) -> bool:
        """Check if request is finished"""
        # Check max_tokens limit (if set)
        if self.max_tokens is not None and self.get_output_len() >= self.max_tokens:
            self.is_finished = True
            self.finish_reason = "length"
            return True
        
        # Check for EOS token
        if self.output_token_ids and self.output_token_ids[-1] == eos_token_id:
            self.is_finished = True
            self.finish_reason = "stop"
            return True
        
        return False


@dataclass
class BatchedRequests:
    """Batched requests for execution"""
    requests: List[Request]
    
    def __len__(self):
        return len(self.requests)
    
    def __iter__(self):
        return iter(self.requests)
    
    def is_empty(self) -> bool:
        return len(self.requests) == 0


@dataclass
class MigratingRequest:
    """Request being migrated from Prefill to Decode stage"""
    req: Request
    
    # Block indexes for KV cache
    kv_block_indexes: Optional[List[int]] = None
    
    # Generated tokens from prefill stage
    output_token_ids: Optional[List[int]] = None
    
    # Expanded prompt tokens (for cross-process transfer)
    expanded_prompt_token_ids: Optional[List[int]] = None
    
    # Source stage information
    source_stage: Optional[EngineStage] = None
    source_worker_id: Optional[int] = None
    
    # Target stage information
    target_stage: Optional[EngineStage] = None
    target_worker_id: Optional[int] = None


@dataclass
class StepOutput:
    """Output from a single execution step"""
    request_id: str
    output_token_ids: List[int]
    finished: bool
    finish_reason: Optional[str] = None
    
    # Logprobs (optional)
    logprobs: Optional[Dict[int, float]] = None
    
    # Profiling information
    step_start_time: Optional[float] = None
    step_end_time: Optional[float] = None
    num_input_tokens: Optional[int] = None  # For prefill
    num_output_tokens: Optional[int] = None  # For decode iterations


# Constants
GB = 1024 ** 3
MB = 1024 ** 2
KB = 1024


