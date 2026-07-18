# KVServe (`fused` branch)

TileLang fused quantizer + LC lossless codec for vLLM V1 PD KV transfer.

## Fastest path: run `tests/test_kvserve.py`

Needs: **2 GPUs**, working **vLLM**, **TileLang**, and a local **LC-framework** tree.

### 1. Checkout

```bash
git clone git@github.com:hpdps-group/KVServe.git
cd KVServe
git checkout fused
export PYTHONPATH="$(pwd)"
```

### 2. Build LC runtime (required once per machine)

`build/` is gitignored — each machine must compile its own `.so`.

```bash
# LC-framework: clone or point to an existing checkout
# git clone https://github.com/<your-org>/LC-framework.git /path/to/LC-framework

python3 scripts/build_lc_runtime.py \
  --lc-dir /path/to/LC-framework \
  --output-dir build/lc_runtime \
  --arch sm_120 \
  --algorithm "TUPL8_1 BIT_8 RZE_2"

export KVSERVE_LC_META_PATH="$(pwd)/build/lc_runtime/lc_runtime_meta.json"
```

Change `--arch` for your GPU (`sm_90`, `sm_89`, …). Confirm with:

```bash
python3 scripts/test_lc_runtime_codec.py
```

### 3. Smoke `test_kvserve` (TileLang + LC)

```bash
python3 tests/test_kvserve.py \
  --mode tilelang_lc \
  --model /path/to/your-model \
  --num-requests 2 \
  --max-tokens 8 \
  --prefill-gpu 0 \
  --decode-gpu 1 \
  --kv-port 25020 \
  --gpu-mem-util 0.75 \
  --output-dir sim_outputs/tilelang_lc_smoke
```

Expect `PASS:` near the end of the log. If GPU memory is tight, lower `--gpu-mem-util` further (e.g. `0.6`).

Equivalent config form:

```bash
python3 tests/test_kvserve.py \
  --mode custom \
  --compression-config configs/compression/fused_top_lc.json \
  --model /path/to/your-model \
  --num-requests 2 \
  --max-tokens 8 \
  --prefill-gpu 0 \
  --decode-gpu 1 \
  --kv-port 25020 \
  --gpu-mem-util 0.75 \
  --output-dir sim_outputs/fused_top_lc_smoke
```

## Optional: 4-way HotpotQA

```bash
MODEL=/path/to/your-model \
DATA=/path/to/hotpotqa.jsonl \
NUM_REQUESTS=20 \
GPU_MEM_UTIL=0.75 \
bash scripts/run_hotpotqa_4way.sh
```

Runs `original_nvcomp`, `original_lc`, `fused_nvcomp`, `fused_lc`. Logs under `sim_outputs/bench_4way_hotpotqa_n20/`.

## Configs

| File | Path |
|------|------|
| `configs/compression/fused_top_lc.json` | TileLang + LC |
| `configs/compression/fused_top_nvcomp.json` | TileLang + nvCOMP |
| `configs/compression/original_top_lc.json` | Hadamard+quant + LC |
| `configs/compression/original_top_nvcomp.json` | Hadamard+quant + nvCOMP |

Default codec wire layout is **permuted** (keeps LC ratio). Override with `KVSERVE_CODEC_LAYOUT=native` only if you accept a possible ratio change.
