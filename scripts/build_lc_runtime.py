#!/usr/bin/env python3
"""
Build an in-process GPU LC runtime shared library for KVServe.

The generated library exposes C ABI functions that operate on CUDA device
pointers. KVServe's Python LCCodec calls these functions through ctypes, so
online inference does not go through LC's file-based command line interface.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


RUNTIME_CU = r'''
#define USE_GPU

#include <cuda_runtime.h>
#include <cstdio>
#include "lc.h"

extern "C" long long lc_max_compressed_size(long long insize)
{
  const long long chunks = (insize + CS - 1) / CS;
  return 2 * (long long)sizeof(long long) + chunks * (long long)sizeof(unsigned short) + chunks * (long long)CS;
}

static int lc_blocks()
{
  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) return -1;
  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, dev);
  if (err != cudaSuccess) return -1;
  return prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / TPB);
}

// Persistent per-device scratch: eliminates cudaMalloc/Free on the hot path.
struct LcScratch {
  int device = -1;
  long long* d_outsize = nullptr;
  long long* d_fullcarry = nullptr;
  long long carry_chunks = 0;
};

static thread_local LcScratch g_scratch;

static int lc_ensure_scratch(long long chunks)
{
  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) return -1;

  if (g_scratch.device != dev) {
    if (g_scratch.d_outsize) cudaFree(g_scratch.d_outsize);
    if (g_scratch.d_fullcarry) cudaFree(g_scratch.d_fullcarry);
    g_scratch = LcScratch{};
    g_scratch.device = dev;
    err = cudaMalloc((void**)&g_scratch.d_outsize, sizeof(long long));
    if (err != cudaSuccess) return -2;
  }
  if (chunks > g_scratch.carry_chunks) {
    if (g_scratch.d_fullcarry) cudaFree(g_scratch.d_fullcarry);
    g_scratch.d_fullcarry = nullptr;
    err = cudaMalloc((void**)&g_scratch.d_fullcarry, chunks * sizeof(long long));
    if (err != cudaSuccess) return -3;
    g_scratch.carry_chunks = chunks;
  }
  return 0;
}

extern "C" int lc_encode_device(
    unsigned long long chain,
    const void* input,
    long long insize,
    void* output,
    long long output_capacity,
    long long* host_outsize)
{
  if (input == nullptr || output == nullptr || host_outsize == nullptr || insize <= 0) return -1;
  const long long chunks = (insize + CS - 1) / CS;
  const long long maxsize = lc_max_compressed_size(insize);
  if (output_capacity < maxsize) return -2;

  const int blocks = lc_blocks();
  if (blocks <= 0) return -3;
  if (lc_ensure_scratch(chunks) != 0) return -4;

  d_reset<<<1, 1>>>();
  cudaError_t err = cudaMemset(g_scratch.d_fullcarry, 0, chunks * sizeof(long long));
  if (err != cudaSuccess) return -5;
  d_encode<<<blocks, TPB>>>(
      chain,
      (const byte*)input,
      insize,
      (byte*)output,
      g_scratch.d_outsize,
      g_scratch.d_fullcarry);

  err = cudaGetLastError();
  if (err == cudaSuccess) {
    // Still need compressed size for the tight slice; one D2H remains.
    err = cudaMemcpy(host_outsize, g_scratch.d_outsize, sizeof(long long), cudaMemcpyDeviceToHost);
  }
  if (err != cudaSuccess) return -6;
  return 0;
}

extern "C" int lc_decode_device(
    unsigned long long chain,
    const void* input,
    long long insize,
    void* output,
    long long output_capacity,
    long long* host_outsize)
{
  if (input == nullptr || output == nullptr || host_outsize == nullptr || insize <= 0) return -1;
  if (output_capacity <= 0) return -3;

  const int blocks = lc_blocks();
  if (blocks <= 0) return -4;
  // Decode only needs d_outsize scratch (carry unused).
  if (lc_ensure_scratch(1) != 0) return -5;

  // Fast path: caller (KVServe) already knows the expected decoded size via
  // original_shape, so skip the header D2H peek and the trailing size D2H.
  // Kernels remain ordered on the current CUDA stream.
  d_reset<<<1, 1>>>();
  d_decode<<<blocks, TPB>>>(
      chain,
      (const byte*)input,
      (byte*)output,
      g_scratch.d_outsize);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return -6;
  *host_outsize = output_capacity;
  return 0;
}
'''


def run(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_component_ids(header: Path) -> dict[str, int]:
    text = header.read_text(encoding="utf-8")
    # Newer LC releases use `enum LC_GPUcomponents { ... }`; older used anonymous enum.
    match = re.search(r"enum(?:\s+\w+)?\s*\{([^}]+)\};", text, flags=re.S)
    if not match:
        raise RuntimeError(f"Could not parse component enum from {header}")
    names = [item.strip() for item in match.group(1).replace("\n", " ").split(",")]
    ids: dict[str, int] = {}
    for idx, name in enumerate(names):
        if not name:
            continue
        name = name.split("=")[0].strip()
        if name in {"NULGPUcomponents", "NUL_GPUcomponents"}:
            ids["NUL"] = idx
            ids[name] = idx
        else:
            ids[name] = idx
    return ids


def algorithm_to_chain(algorithm: str, component_ids: dict[str, int]) -> int:
    chain = 0
    parts = [part for part in algorithm.split() if part]
    if len(parts) > 8:
        raise ValueError("LC supports at most 8 stages")
    for offset, name in enumerate(parts):
        if name not in component_ids:
            raise KeyError(f"Unknown LC component {name!r}")
        chain |= int(component_ids[name]) << (8 * offset)
    return chain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lc-dir", default="LC-framework")
    parser.add_argument("--output-dir", default="build/lc_runtime")
    parser.add_argument("--algorithm", action="append", required=True)
    parser.add_argument("--arch", default="sm_120", help="CUDA arch for RTX 5090; use sm_90/sm_89 on older GPUs")
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    lc_dir = Path(args.lc_dir)
    lc_dir = lc_dir.resolve() if lc_dir.is_absolute() else (repo / lc_dir).resolve()
    output_dir = Path(args.output_dir)
    output_dir = (
        output_dir.resolve() if output_dir.is_absolute() else (repo / output_dir).resolve()
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "include").mkdir(parents=True, exist_ok=True)

    run(
        [
            "python3",
            "generate_Device_LC-Framework.py",
            "--output_dir",
            str(output_dir),
            "--base_file",
            "framework.h",
            "--main_file",
            "framework.cu",
        ],
        cwd=lc_dir,
    )

    runtime_cu = output_dir / "lc_runtime.cu"
    runtime_cu.write_text(RUNTIME_CU, encoding="utf-8")

    lib_path = output_dir / "liblc_runtime.so"
    compile_cmd = [
        args.nvcc,
        "-O3",
        "-std=c++17",
        f"-arch={args.arch}",
        "-Xcompiler",
        "-fPIC",
        "-shared",
        "-DUSE_GPU",
        "-I",
        str(output_dir),
        "-I",
        str(lc_dir),
        "-o",
        str(lib_path),
        str(runtime_cu),
    ]
    run(compile_cmd, cwd=output_dir)

    component_ids = parse_component_ids(output_dir / "components/include/GPUcomponents.h")
    algorithms = {
        algorithm: algorithm_to_chain(algorithm, component_ids)
        for algorithm in args.algorithm
    }
    meta = {
        "library": str(lib_path),
        "component_ids": component_ids,
        "algorithms": algorithms,
        "arch": args.arch,
    }
    meta_path = output_dir / "lc_runtime_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nBuilt: {lib_path}")
    print(f"Metadata: {meta_path}")
    for algorithm, chain in algorithms.items():
        print(f"  {algorithm}: chain=0x{chain:x}")


if __name__ == "__main__":
    main()
