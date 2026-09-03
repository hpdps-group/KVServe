#!/usr/bin/env python3
"""Track B2 AE driver: stream the search, validate artifacts, and summarize."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SEARCH_SCRIPT = ROOT / "kvserve_v1/offline_search/evaluation/param_search/test_search.py"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KVServe Track B2 offline-search AE")
    parser.add_argument(
        "--config", type=Path,
        default=Path(os.environ.get(
            "CONFIG", str(ROOT / "configs/representative/profiler_small.json"))),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(os.environ.get("OUT_DIR", str(ROOT / "results/profiler_small"))),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _search_environment() -> dict[str, str]:
    env = os.environ.copy()
    source_root = str(ROOT / "kvserve_v1")
    env["PYTHONPATH"] = source_root + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_search(config_path: Path, output_dir: Path, log_path: Path) -> int:
    command = [
        sys.executable, str(SEARCH_SCRIPT),
        "--config", str(config_path),
        "--results-dir", str(output_dir),
    ]
    print("[AE] Track B2 offline-search validation")
    print(f"[AE] Results -> {output_dir}")
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command, cwd=ROOT, env=_search_environment(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    if return_code:
        print(f"KVServe AE Track B2: FAILED (search exited {return_code})", file=sys.stderr)
    return return_code


def _validate_and_print(summary_path: Path, log_path: Path) -> int:
    if not summary_path.is_file():
        print(f"Track B2 summary is missing: {summary_path}", file=sys.stderr)
        return 1
    summary = _load_json(summary_path)
    if summary.get("status") != "complete":
        print(f"Track B2 did not complete: {summary.get('status', 'unknown')}", file=sys.stderr)
        return 1
    evaluated_path = Path(str(summary.get("evaluated_results", "")))
    feasible_path = Path(str(summary.get("feasible_results", "")))
    if not evaluated_path.is_file() or not feasible_path.is_file():
        print("Track B2 result files referenced by summary are missing", file=sys.stderr)
        return 1

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
    print(f"  detailed log: {log_path}")
    if not summary["measured_profile"]:
        print("  note: proxy accuracy validates search behavior only; no paper profile was produced")
    return 0


def main() -> int:
    args = _args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    config = _load_json(config_path)
    model_name = config.get("model_name")
    if not model_name:
        print(f"model_name is missing from {config_path}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    summary_path = output_dir / str(model_name) / "summary.json"
    return_code = _run_search(config_path, output_dir, log_path)
    if return_code:
        return return_code
    return _validate_and_print(summary_path, log_path)


if __name__ == "__main__":
    raise SystemExit(main())
