# KVServe

SIGCOMM 2026 artifact-evaluation snapshot (`ae` branch).

**Reviewer guide:** [`ae/README.md`](ae/README.md)

That document is the AE README: how to enter the prepared `sigcomm-ae`
container, which commands to run, how each test maps to paper claims/figures,
and what output to expect.

This branch matches the evaluation container (`/workspace`). Code, models,
datasets, and dependencies for the five functional tests (B1–B3, C1–C2) are
already there. Testbed login is provided privately via HotCRP, not in git.

KVServe itself is a vLLM KV-connector for service-aware KV-cache compression
in disaggregated LLM serving. Paper: https://arxiv.org/abs/2605.13734
