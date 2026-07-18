# KVServe (`fused`)

## Run TileLang + LC PD smoke test

```bash
git clone git@github.com:hpdps-group/KVServe.git && cd KVServe
git checkout fused
export PYTHONPATH=$PWD

# 1) build LC once (need local LC-framework; set --arch for your GPU)
python3 scripts/build_lc_runtime.py \
  --lc-dir /path/to/LC-framework \
  --output-dir build/lc_runtime \
  --arch sm_120 \
  --algorithm "TUPL8_1 BIT_8 RZE_2"
export KVSERVE_LC_META_PATH=$PWD/build/lc_runtime/lc_runtime_meta.json

# 2) two-GPU smoke
python3 tests/test_kvserve.py \
  --mode tilelang_lc \
  --model /path/to/model \
  --num-requests 2 --max-tokens 8 \
  --prefill-gpu 0 --decode-gpu 1 \
  --kv-port 25020 --gpu-mem-util 0.75
```

See `PASS:` at the end. OOM → lower `--gpu-mem-util`.
