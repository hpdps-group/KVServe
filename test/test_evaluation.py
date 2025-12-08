#!/usr/bin/env python3
"""
KVServe Evaluation Example
Standard usage example for running evaluation with KVServe backend.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from kvserve.eval.cli import main

if __name__ == "__main__":
    # Standard usage: modify these parameters as needed
    model_path = "/root/ssd/Llama3.1-8B-Instruct"
    
    sys.argv = [
        'kvserve_eval',
        '--model', 'kvserve',
        '--model_args', f'pretrained={model_path},num_prefill_workers=1,num_decoding_workers=1,max_model_len=32768,max_batch_size=32,temperature=0.0,top_p=1.0,top_k=-1',
        '--tasks', 'longbench_2wikimqa',
        '--num_fewshot', '0',
        '--limit', '5',
        '--verbosity', 'INFO'
    ]
    
    main()
