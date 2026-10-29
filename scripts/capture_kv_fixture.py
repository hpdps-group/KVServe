#!/usr/bin/env python3
"""Capture a real model KV fixture for bench_accelerated_stages.py.

Requires a local Transformers model/tokenizer, CUDA or ROCm PyTorch and an
input text file. The resulting file contains model activations and may be large;
share it only with people who are allowed to receive the input-derived data.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local model directory")
    parser.add_argument("--text", required=True, help="UTF-8 input text file")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.tokens < 1 or not torch.cuda.is_available():
        parser.error("A positive token count and a CUDA/ROCm GPU are required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = tokenizer(
        Path(args.text).read_text(), return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    if ids.shape[1] < args.tokens:
        raise ValueError(f"Input contains {ids.shape[1]} tokens, need {args.tokens}")
    device = torch.device(args.device)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    with torch.inference_mode():
        out = model(input_ids=ids[:, : args.tokens].to(device), use_cache=True)
        cache = out.past_key_values
        if hasattr(cache, "layers"):
            layers = [(layer.keys, layer.values) for layer in cache.layers]
        else:
            layers = [(layer[0], layer[1]) for layer in cache]
        if not layers or any(k.ndim != 4 or v.shape != k.shape for k, v in layers):
            raise ValueError("Expected dense [batch,heads,tokens,head_dim] K/V caches")
        if any(k.shape[0] != 1 or k.shape[2] < args.tokens for k, _ in layers):
            raise ValueError("Cache is too short or has an unsupported batch size")
        keys = torch.stack(
            [
                k[0, :, : args.tokens].permute(1, 0, 2).contiguous().cpu()
                for k, _ in layers
            ]
        )
        values = torch.stack(
            [
                v[0, :, : args.tokens].permute(1, 0, 2).contiguous().cpu()
                for _, v in layers
            ]
        )
    if keys.dtype != torch.bfloat16 or values.dtype != torch.bfloat16:
        raise ValueError("This benchmark requires BF16 K/V caches")
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "key": keys,
            "value": values,
            "model": str(args.model),
            "tokens": args.tokens,
            "dtype": "bfloat16",
            "layout": "layers,tokens,heads,head_dim",
        },
        target,
    )
    print(f"Wrote {target}; key/value shape={list(keys.shape)}")


if __name__ == "__main__":
    main()
