"""
Evaluation runner for KVServe
Wraps lm-evaluation-harness simple_evaluate function
"""

import logging
from typing import List, Optional, Union, Dict, Any
import asyncio
import ray

from kvserve.engine.backend import PDBackend
from kvserve.eval.evaluator import KVServeEvaluator

eval_logger = logging.getLogger(__name__)


def run_evaluation(
    backend: PDBackend,
    tasks: Union[str, List[str]],
    model_path: Optional[str] = None,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[int] = None,
    limit: Optional[Union[int, float]] = None,
    verbosity: str = "INFO",
    output_path: Optional[str] = None,
    apply_chat_template: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Run evaluation on KVServe backend using lm-evaluation-harness
    
    Args:
        backend: Initialized PDBackend instance
        tasks: Task name(s) to evaluate on (e.g., "hellaswag", ["hellaswag", "mmlu"])
        model_path: Path to model (for tokenizer loading)
        num_fewshot: Number of few-shot examples
        batch_size: Batch size for evaluation
        limit: Limit number of examples (int) or fraction (float)
        verbosity: Logging verbosity level
        output_path: Path to save results JSON
        **kwargs: Additional arguments passed to simple_evaluate
    
    Returns:
        Dictionary containing evaluation results
    """
    # Import lm_eval here to avoid requiring it as a hard dependency
    try:
        from lm_eval import simple_evaluate
        from lm_eval.tasks import TaskManager
    except ImportError:
        raise ImportError(
            "lm-evaluation-harness is required for evaluation. "
            "Install it with: pip install lm-eval"
        )
    
    # Initialize evaluator
    evaluator = KVServeEvaluator(
        backend=backend,
        tokenizer=backend.tokenizer if hasattr(backend, 'tokenizer') else None,
        apply_chat_template=apply_chat_template,
        batch_size=batch_size or 1
    )
    
    # Convert tasks to list if string
    if isinstance(tasks, str):
        tasks = [tasks]
    
    # Create task manager
    task_manager = TaskManager(verbosity=verbosity)
    
    # Run evaluation using lm-eval's simple_evaluate
    results = simple_evaluate(
        model=evaluator,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size or 1,
        limit=limit,
        verbosity=verbosity,
        task_manager=task_manager,
        **kwargs
    )
    
    # Save results if output_path specified
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        eval_logger.info(f"Results saved to {output_path}")
    
    return results


async def run_evaluation_async(
    backend: PDBackend,
    tasks: Union[str, List[str]],
    model_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Async version of run_evaluation
    
    Args:
        backend: Initialized PDBackend instance
        tasks: Task name(s) to evaluate on
        model_path: Path to model
        **kwargs: Additional arguments passed to run_evaluation
    
    Returns:
        Dictionary containing evaluation results
    """
    # Run in executor to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: run_evaluation(backend, tasks, model_path, **kwargs)
    )


def main():
    """
    Command-line interface for running evaluation
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate KVServe backend on tasks")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help="Task names to evaluate (e.g., hellaswag mmlu)"
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples"
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit number of examples (int) or fraction (float)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save results JSON"
    )
    parser.add_argument(
        "--num_prefill_workers",
        type=int,
        default=1,
        help="Number of prefill workers"
    )
    parser.add_argument(
        "--num_decoding_workers",
        type=int,
        default=1,
        help="Number of decoding workers"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="WARNING",
        help="Log level (ERROR, WARNING, INFO, DEBUG)"
    )
    
    args = parser.parse_args()
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_gpus=args.num_prefill_workers + args.num_decoding_workers,
        )
    
    # Create backend
    backend = PDBackend(
        model_path=args.model_path,
        num_prefill_workers=args.num_prefill_workers,
        num_decoding_workers=args.num_decoding_workers,
        block_size=16,
        max_num_gpu_blocks=3000,
        dtype="float16",
        gpu_memory_utilization=0.85,
        kv_transfer_method="nccl",
        nccl_init_method="tcp://localhost:29500",
        log_level=args.log_level,
    )
    
    # Initialize backend
    async def run():
        await backend.initialize()
        await backend.start()
        
        try:
            # Run evaluation
            results = run_evaluation(
                backend=backend,
                tasks=args.tasks,
                model_path=args.model_path,
                num_fewshot=args.num_fewshot,
                limit=args.limit,
                output_path=args.output_path,
                verbosity="INFO",
            )
            
            # Print results
            print("\n" + "="*50)
            print("Evaluation Results")
            print("="*50)
            for task_name, task_results in results.get("results", {}).items():
                print(f"\n{task_name}:")
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
            
        finally:
            await backend.stop()
    
    asyncio.run(run())


if __name__ == "__main__":
    main()


