"""
Evaluation module for KVServe
Compatible with lm-evaluation-harness interface
"""

from kvserve.eval.evaluator import KVServeEvaluator
from kvserve.eval.runner import run_evaluation

__all__ = [
    "KVServeEvaluator",
    "run_evaluation",
]


