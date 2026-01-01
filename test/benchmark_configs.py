"""
Benchmark configurations for different scenarios
"""

# Configuration 1: Short context (baseline - no benefit expected)
CONFIG_SHORT = {
    "name": "short_context",
    "num_requests": 100,
    "prompt_length": 100,
    "max_output_tokens": 50,
    "description": "短 context - Multi-stream 收益很小",
}

# Configuration 2: Medium context (moderate benefit)
CONFIG_MEDIUM = {
    "name": "medium_context",
    "num_requests": 200,
    "prompt_length": 500,
    "max_output_tokens": 100,
    "description": "中等 context - Multi-stream 开始有收益",
}

# Configuration 3: Long context (significant benefit expected)
CONFIG_LONG = {
    "name": "long_context",
    "num_requests": 100,
    "prompt_length": 4000,  # Long prompt for testing multi-stream benefits
    "max_output_tokens": 100,
    "description": "长 context (4000 tokens) - Multi-stream 收益显著",
}

# Configuration 4: Very long context (maximum benefit)
# NOTE: Requires max_model_len >= 1800 in backend config
CONFIG_VERY_LONG = {
    "name": "very_long_context",
    "num_requests": 100,
    "prompt_length": 1700,  # Reduced to fit max_model_len=2000
    "max_output_tokens": 100,
    "description": "超长 context - Multi-stream 收益最大",
}

# Configuration 5: High concurrency
CONFIG_HIGH_CONCURRENCY = {
    "name": "high_concurrency",
    "num_requests": 500,
    "prompt_length": 1000,
    "max_output_tokens": 50,
    "description": "高并发 - Multi-stream 并行传输收益",
}

# All configurations
ALL_CONFIGS = [
    # CONFIG_SHORT,      # Skip - already tested
    CONFIG_MEDIUM,
    CONFIG_LONG,
    # CONFIG_VERY_LONG,  # Optional - takes longer
    # CONFIG_HIGH_CONCURRENCY,  # Optional
]

# Recommended configuration for demo
RECOMMENDED_CONFIG = CONFIG_LONG

