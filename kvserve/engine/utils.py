"""
Utility classes and functions for PD separation engine
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time


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


@dataclass
class Request:
    """Request representation for PD separation engine"""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 means disabled
    
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
    
    def __post_init__(self):
        if self.output_token_ids is None:
            self.output_token_ids = []
        if self.arrival_time is None:
            self.arrival_time = time.time()
    
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
        if self.get_output_len() >= self.max_tokens:
            self.is_finished = True
            self.finish_reason = "length"
            return True
        
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


