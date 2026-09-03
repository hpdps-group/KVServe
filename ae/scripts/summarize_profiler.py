#!/usr/bin/env python3
"""Print and validate the compact Track B2 AE result."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.is_file():
        raise SystemExit(f"Track B2 summary is missing: {summary_path}")
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("status") != "complete":
        raise SystemExit(f"Track B2 did not complete: {summary.get('status', 'unknown')}")

    best = summary.get("best_compression_ratio")
    best_text = "n/a" if best is None else f"{best:.4f}x"
    print("KVServe AE Track B2: PASS")
    print(f"  evaluation mode: {summary['evaluation_mode']} (measured profile: {summary['measured_profile']})")
    print(f"  model/task: {summary['model']} / {','.join(summary['tasks'])}")
    print(f"  CR candidates: {summary['candidate_count']}")
    print(f"  accuracy evaluations: {summary['accuracy_evaluations']}")
    print(f"  feasible (relative accuracy >= {summary['accuracy_threshold']:.1f}%): {summary['feasible_count']}")
    print(f"  best compression ratio: {best_text}")
    print(f"  elapsed: {summary['elapsed_seconds']:.1f}s")
    print(f"  summary: {summary_path}")
    print(f"  detailed log: {Path(args.log)}")
    if not summary["measured_profile"]:
        print("  note: proxy accuracy validates search behavior only; no paper profile was produced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
