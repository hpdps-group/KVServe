# DuoConfig Generation

This module provides tools to generate DuoAttention configuration for KVServe's quantizer.

## Overview

DuoAttention is a method to identify which attention heads are important for long-context retrieval ("Retrieval Heads") and which are only attending to local context ("Streaming Heads"). This module computes the attention scores for each layer's KV heads to help configure the DuoAttention mechanism in KVServe.

Specifically, based on the computed DuoAttention scores, we categorize the attention heads into **High-Precision Compression Heads** and **Low-Precision Compression Heads**:

- **High-Precision Compression Heads**: These heads are critical for retrieving information from long contexts. They require higher precision quantization to preserve retrieval capabilities.
- **Low-Precision Compression Heads**: These heads primarily focus on local context or have less impact on global attention. They can be compressed with lower precision to maximize memory savings without significantly degrading performance.


## Usage

### Command Line Interface

You can run the script to generate scores for your model:

```bash
python -m kvserve.config.duo_config.run_duo_config \
    --model_name /path/to/model \
    --num_samples 10 \
    --q_len 1024 \
    --max_tokens 2048
```

### Programmatic Usage

You can also use the `DuoConfigGenerator` class in your Python code to generate scores or load existing ones:

```python
from kvserve.config.duo_config.get_config import DuoConfigGenerator
from transformers import AutoModelForCausalLM

# 1. Generate scores on the fly
model = AutoModelForCausalLM.from_pretrained("path/to/model")
scores = DuoConfigGenerator.duo_attention_on_the_fly(
    model, 
    num_samples=10, 
    q_len=1024, 
    max_tokens=2048
)

# 2. Load existing scores from CSV
# Looks for <model_basename>_scores.csv in current directory or script directory
scores = DuoConfigGenerator.get_scores_from_csv("/path/to/model/Llama-3.1-8B-Instruct")
```

### Arguments

- `--model_name`: The model name or path (default: `Llama-3.1-8B-Instruct`).
- `--num_samples`: Number of samples from BookSum dataset to use. If not specified, uses all samples (default: None).
- `--q_len`: Query length for attention calculation (default: 1024).
- `--max_tokens`: Maximum tokens to read from each sample (default: 2048).

### Output

The script outputs a CSV file named `<model_name>_scores.csv` in the current directory. This file contains the computed attention scores for each layer and KV head group. These scores are used to determine which heads should be high-precision/low-precision.

## Implementation Details

The `duo_attention_on_the_fly` function:
1. Loads samples from the `kmfoda/booksum` dataset.
2. Computes mean query and key vectors.
3. Calculates attention weights and the area under the cumulated attention curve.
4. Aggregates scores across samples to identify head importance.

This implementation references the method from [KVPress](https://github.com/NVIDIA/kvpress).

