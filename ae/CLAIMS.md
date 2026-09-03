# Claims → AE tracks

AE checks direction and mechanism. It does not re-run the paper's 20-hour
profiler or every bandwidth point.

| ID | Paper claim | AE track | What the reviewer should see |
| --- | --- | --- | --- |
| C1 | KV compression pipeline is a real encode/decode path, not a library sketch | B1 | KV → transform → quant → nvCOMP → transfer → decode → reconstruct; shape OK; CR > 1 |
| C2 | Offline search finds accuracy-feasible, high-CR configs | B2 | Default smoke run validates candidates / GP / BO / feasibility / pruning with proxy accuracy; optional `model` mode performs measured LongBench evaluation. The paper-scale profile library is not regenerated. |
| C3 | Online controller picks compression from bandwidth, quality, machine, and SLO | B3 | Printed INPUT (B, acc_req, machine, SLO, V, length) → OUTPUT: low B compresses, 150 Gbps turns compression off; tighter quality/SLO shrinks the feasible set |
| C4 | Controller corrects offline/online residual (bandit) | B3 | Inject slower-than-predicted latency on the chosen profile; later decisions move off it |
| C5 | Under constrained bandwidth, KVServe cuts KV transfer cost vs no/static compression; as bandwidth rises, compression is relaxed or turned off | C1 | PD, 3 systems × {10, 25/50, 100} Gbps |
| C6 | Compressed remote KV beats uncompressed remote KV, both beat cold prefill | C2 | TTFT(KVServe) < TTFT(uncompressed remote KV) << TTFT(cold prefill) |

## Explicit non-claims for AEC

- Not treating AE magnitudes as paper-scale end-to-end results.
- Not reproducing the full profiler wall-clock or the full bandwidth sweep.
- Not treating B2 smoke-mode proxy accuracy as a measured model result.
- Not requiring multi-node topology.

If a representative E2E ratio disagrees in **direction** with C5/C6, that is
an AE failure. Magnitude may differ from the paper because the AE workload,
GPU, and search budget are smaller.

## Mapping onto existing code

No new core modules. AE only wraps what is already in the repo:

| Track | Existing surface |
| --- | --- |
| B1 | `kvserve_v1/compression/*`, `tests/test_kvserve.py` |
| B2 | `kvserve_v1/offline_search/evaluation/param_search/test_search.py` |
| B3 | `kvserve_v1/compression/controller/dynamic_online_controller.py` + `profiles/` |
| C1 | `ae/scripts/run_pd.sh` (`tests/test_kvserve_remote.py`, SL3060→SL3061) |
| C2 | `ae/scripts/run_kv_reuse.sh` (`tests/test_kvserve.py --mode default`) |

## Controller bandwidth units

Paper text uses Gbps. `OnlineController.select_profile(..., B_mbps=...)` uses
MB/s. AE configs and `validate.sh` must convert explicitly.
