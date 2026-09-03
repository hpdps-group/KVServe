# AE Environment

Prepared runtime for the SIGCOMM 2026 AE. Reviewers use the existing
`sigcomm-ae` container; they do not rebuild the image or recreate the
container. Login is provided privately via HotCRP.

| Item | Value |
| --- | --- |
| GPUs | 8× RTX 5090 (32 GB) per host |
| Interconnect (C1) | 50 Gb/s Ethernet |
| Image | `kvserve-env:v2` |
| Container | `sigcomm-ae` |
| Workspace | `/workspace` (this `ae` branch) |
| Model | `/data/models/Qwen2.5-7B-Instruct` |
| Python | `/opt/kvs-venv` (3.12, torch 2.10+cu128, vLLM 0.18, nvCOMP) |

Enter (after HotCRP SSH):

```bash
docker exec -it sigcomm-ae bash
cd /workspace
export PYTHONPATH=/workspace
```
