# KVServe Artifact Evaluation (SIGCOMM 2026)

This README is the AE document for HotCRP. It is written for the
**Artifacts Available** and **Artifacts Evaluated — Functional** badges.

**Paper.** Zedong Liu et al., *KVServe: Service-Aware KV Cache Compression
for Communication-Efficient Disaggregated LLM Serving*, ACM SIGCOMM 2026.
The accepted PDF was uploaded at AE registration. Public copy:
https://arxiv.org/abs/2605.13734

**Code.** https://github.com/hpdps-group/KVServe, branch `ae`

The `sigcomm-ae` container `/workspace` is this branch. Reviewers do not
need to clone, pip-install, download weights, or rebuild the image.

No malicious or destructive operations. No analytics/tracking on the
artifact during evaluation.

---

## 1. Access

Testbed SSH login is **not** stored in this repository. It is sent to AEC
privately through HotCRP.

After you are logged in on the prefill host:

```bash
docker exec -it sigcomm-ae bash
cd /workspace
export PYTHONPATH=/workspace
```

Reviewers only log in once (prefill host). C1 starts decode on the second
machine over SSH by itself. If `sigcomm-ae` is not running, start that
existing container; do not create a new one.

---

## 2. Environment (already in the container)

| Item | Value |
| --- | --- |
| GPUs | 8× RTX 5090 (32 GB) on each host |
| Interconnect (C1) | 50 Gb/s Ethernet |
| Image | `kvserve-env:v2` (not rebuilt for AE) |
| Workspace | `/workspace` |
| Model | `/data/models/Qwen2.5-7B-Instruct` |
| Python | `/opt/kvs-venv` (3.12, torch 2.10+cu128, vLLM 0.18, nvCOMP) |
| Dataset | `datasets/hotpotqa_ae.jsonl` (20 HotpotQA prefixes) |

C1 and C2 need GPUs 0 and 1 free. Do not run them at the same time.

---

## 3. Tests ↔ paper claims / figures

AE checks **mechanism** on Qwen2.5-7B. These are reduced functional tests,
not paper-scale reproductions (7B vs 8B/32B, 50 Gb/s vs shaped 5–100 Gb/s,
20 requests, simulated SSD on C2).

| Cmd | Track | Paper claim | Figures | Time |
| --- | --- | --- | --- | --- |
| `./ae/scripts/run_compression.sh` | B1 | Unified encode/decode pipeline, not a library sketch | Fig. 6, Fig. 7 | 2–5 min |
| `./ae/scripts/run_profiler_small.sh` | B2 | Bayesian search: candidates, GP/BO, feasible/infeasible, prune | Fig. 8, Fig. 9, Fig. 10 | ~5 s |
| `./ae/scripts/run_controller.sh` | B3 | Online pick from bandwidth / quality / SLO; bandit residual | Fig. 4, Fig. 11, Fig. 16 | ~5 s |
| `./ae/scripts/run_pd.sh` | C1 | Compressed PD KV transfer over a real cross-machine link | Supports the mechanism evaluated in Fig. 13 | ~15 min |
| `./ae/scripts/run_kv_reuse.sh` | C2 | Compressed prefix-KV reuse vs recompute / uncompressed remote KV | Supports the mechanism evaluated in Fig. 14 | 10–15 min |

Suggested order: B2, B3 (CPU), then B1, C2, C1 (GPU).

```bash
./ae/scripts/run_profiler_small.sh
./ae/scripts/run_controller.sh
./ae/scripts/run_compression.sh
./ae/scripts/run_kv_reuse.sh
./ae/scripts/run_pd.sh
```

---

## 4. Expected output

### B1 — compression pipeline (`run_compression.sh`)

One node, GPUs 0 and 1. Real path:
KV → Hadamard → hybrid quant → nvCOMP/ANS → transfer → decode → reconstruct.

```
KVServe AE Track B1 — Compression Pipeline

INPUT
  model       Qwen2.5-7B-Instruct
  GPUs        prefill=0  decode=1
  requests    1 measured (+1 warm-up)
  pipeline    Hadamard -> quantizer -> nvCOMP/ANS

OUTPUT
  completed   1/1 requests
  KV payload  ~6.1 -> ~1.3 MiB  (~4.6x)
  checks      Hadamard fwd+inv / shape / encode+decode OK

KVServe AE Track B1: PASSED
```

Functional PASS: 1 measured request completed, CR > 1, shape match, encode
and decode both ran.

Summary: `results/smoke/summary.json`

### B2 — mini Bayesian profiling (`run_profiler_small.sh`)

CPU. Loads 144 cached CR candidates and runs BO with **proxy** accuracy
(no 7B forward). This is not a measured paper profile.

```
[SEARCH][cold 1/2] ... status=INFEASIBLE
[SEARCH][cold 2/2] ... status=FEASIBLE
[SEARCH][bo 1/50] propose ...
KVServe AE Track B2: PASS
  CR candidates: 144
  accuracy evaluations: ~49
  feasible: ~22
  best compression ratio: ~9.7x
  note: proxy accuracy validates search behavior only
```

Functional PASS: `status=complete`, both feasible and infeasible decisions
appear.
Summary: `results/profiler_small/Qwen2.5-7B-Instruct/summary.json`

### B3 — online controller (`run_controller.sh`)

CPU. Pick a profile from bandwidth / accuracy / machine / SLO, then inject
4× latency and show the controller leaving that profile.

```
KVServe AE Track B3 — Online controller
  B=5 / 25 / 150 Gbps  →  compress / compress / no_compression
  B=25 Gbps  observe 4x predicted latency  →  switches profile
KVServe AE Track B3: PASSED
```

Functional PASS: low B compresses; 150 Gbps disables compression; 4×
observation moves the choice. Stdout only.

### C1 — cross-machine PD (`run_pd.sh`)

Same SL3060 shell. Decode is SL3061. Two concurrent streams on GPUs 0 and 1,
10 HotpotQA requests each, `none` vs `default`, shared 50 Gb/s NIC.

```
KVServe AE Track C1 — Cross-machine PD
  none      makespan~16s   payload~15600 MiB  cr=1.00x
  default   makespan~11s   payload~1780 MiB   cr~8.7x
  speedup vs uncompressed (wall)     ~1.4x
  speedup vs uncompressed (payload)  ~8.8x
KVServe AE Track C1: PASSED
```

Functional PASS: both modes complete successfully; compressed mode
transfers fewer KV bytes; encode/decode path completes correctly.

Expected performance trend: compressed mode normally reduces wall time
under the provided 50 Gb/s setup. Wall speedup is modest here because 7B
GQA prefill still dominates. Speedups are reported for inspection, not as
a PASS gate.

Summary: `results/pd/summary.json`

### C2 — prefix cache (`run_kv_reuse.sh`)

This is a **reduced functional test**. One node, GPUs 0 and 1. 20 full
HotpotQA prefixes. Cache-hit (compress + decompress) vs cold prefill vs a
**simulated** remote SSD at 5 Gb/s.

It exercises KV compression/reconstruction and the prefix-reuse path; it
does **not** reproduce the paper-scale KV-disaggregation topology in
Fig. 14.

```
KVServe AE Track C2 — Prefix cache
  recompute (cold prefill)       ~1100 ms
  remote SSD uncompressed        ~1235 ms   @ 5 Gbps
  KVServe default (cache hit)     ~186 ms   cr~8.7x
  speedup vs recompute         ~5.9x
  speedup vs remote SSD        ~6.6x
KVServe AE Track C2: PASSED
```

Functional PASS: recompute, simulated SSD accounting, and KVServe cache-hit
all complete; compressed hit path reconstructs KV (CR > 1, decode ran).

Expected performance trend: KVServe cache-hit is normally faster than
recompute and uncompressed SSD in this setup. Ratios vary with GPU load.

Summary: `results/kv_reuse/summary.json`

---

## 5. Paper-scale profiling

Full paper-scale profiling is not required for AE because it takes ~20 h.
The full configuration and entry point are provided in `configs/paper/` and
`ae/scripts/run_profiler_full.sh`.

B2 default smoke mode uses proxy accuracy; it is not a measured LongBench
or paper profile. AE checks mechanism and direction. Exact speedups need
not match the paper.
