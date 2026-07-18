#!/usr/bin/env bash
# Reproduce KVServe-main README 4-way HotpotQA online comparison on KVServe_fused.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export KVSERVE_LC_META_PATH="${KVSERVE_LC_META_PATH:-$ROOT/build/lc_runtime/lc_runtime_meta.json}"
export KVSERVE_LC_ALGORITHM="${KVSERVE_LC_ALGORITHM:-TUPL8_1 BIT_8 RZE_2}"

MODEL="${MODEL:-/data/models/Llama-3-8B-Instruct}"
DATA="${DATA:-$ROOT/datasets/LongBench/data/hotpotqa.jsonl}"
OUT="${OUT:-$ROOT/sim_outputs/bench_4way_hotpotqa_n20}"
NUM_REQUESTS="${NUM_REQUESTS:-20}"
MAX_TOKENS="${MAX_TOKENS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
BASE_PORT="${BASE_PORT:-28301}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"

mkdir -p "$OUT"

run_one() {
  local name="$1"
  local cfg="$2"
  local port="$3"

  mkdir -p "$OUT/$name"
  echo "===== RUNNING $name ====="
  python3 tests/test_kvserve.py \
    --mode custom \
    --compression-config "$cfg" \
    --model "$MODEL" \
    --data-path "$DATA" \
    --num-requests "$NUM_REQUESTS" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --prefill-gpu "$PREFILL_GPU" \
    --decode-gpu "$DECODE_GPU" \
    --kv-port "$port" \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    --output-dir "$OUT/$name" \
    >"$OUT/$name/run.log" 2>&1

  echo "===== DONE $name ====="
  grep -A 20 'SUMMARY' "$OUT/$name/run.log" | tail -n 25 || true
  grep -E 'PASS:|FAIL:' "$OUT/$name/run.log" | tail -n 5 || true
}

run_one original_nvcomp "$ROOT/configs/compression/original_top_nvcomp.json" "$BASE_PORT"
run_one original_lc     "$ROOT/configs/compression/original_top_lc.json"     "$((BASE_PORT + 1))"
run_one fused_nvcomp    "$ROOT/configs/compression/fused_top_nvcomp.json"    "$((BASE_PORT + 2))"
run_one fused_lc        "$ROOT/configs/compression/fused_top_lc.json"        "$((BASE_PORT + 3))"

OUT="$OUT" python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
rows = []
ref = {
    "original_nvcomp": 10.118,
    "original_lc": 6.690,
    "fused_nvcomp": 8.826,
    "fused_lc": 6.113,
}
print(f"{'method':<18} {'ours_ratio':>10} {'ref_ratio':>10} {'delta':>8} {'n':>4} {'pass':>6} {'comp_MB':>10}")
for name in ["original_nvcomp", "original_lc", "fused_nvcomp", "fused_lc"]:
    cand = list((out / name).glob("compression_stats_*.jsonl"))
    ratios, comp = [], 0.0
    n = 0
    if cand:
        for line in cand[0].read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            o = float(r.get("original_bytes", 0))
            c = float(r.get("compressed_bytes", 0))
            if o > 0 and c > 0:
                ratios.append(o / c)
                comp += c
                n += 1
    avg = sum(ratios) / len(ratios) if ratios else float("nan")
    ref_r = ref[name]
    delta = avg - ref_r if ratios else float("nan")
    log = out / name / "run.log"
    passed = "Y" if log.exists() and "PASS:" in log.read_text(errors="ignore") else "N"
    print(f"{name:<18} {avg:10.3f} {ref_r:10.3f} {delta:8.3f} {n:4d} {passed:>6} {comp/1e6:10.2f}")
    rows.append({
        "method": name,
        "ours_ratio": avg,
        "ref_ratio": ref_r,
        "delta": delta,
        "n": n,
        "pass": passed,
        "compressed_MB": comp / 1e6,
    })
summary = out / "summary_4way.csv"
with summary.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("wrote", summary)
PY
