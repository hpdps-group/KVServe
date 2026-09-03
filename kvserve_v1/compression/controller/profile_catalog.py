"""Search and load ProfileLibrary JSONs under profiles/."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from kvserve_v1.compression.controller.profile_library import ProfileLibrary

_PROFILES = Path(__file__).resolve().parents[3] / "profiles"
_CATALOG = _PROFILES / "catalog.json"


def catalog() -> Dict[str, Any]:
    return json.loads(_CATALOG.read_text())


def search(
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    machine: Optional[str] = None,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    hits = []
    for e in catalog()["entries"]:
        if model and model.lower() not in e["model"].lower():
            continue
        if dataset and dataset.lower() not in e["dataset"].lower():
            continue
        if machine and machine.lower() not in str(e["machine"]).lower():
            continue
        if source and source.lower() not in e["source"].lower():
            continue
        hits.append(e)
    return hits


def library_path(
    model: str,
    dataset: str = "qasper",
    machine: str = "5090",
) -> Path:
    hits = search(model=model, dataset=dataset, machine=machine)
    exact = [
        h for h in hits
        if h["model"] == model and h["dataset"] == dataset and h["machine"] == machine
    ]
    chosen = exact[0] if exact else (hits[0] if hits else None)
    if chosen is None:
        raise FileNotFoundError(
            f"no library for model={model!r} dataset={dataset!r} machine={machine!r}"
        )
    return _PROFILES / chosen["path"]


def load(
    model: str = "Qwen2.5-7B-Instruct",
    dataset: str = "qasper",
    machine: str = "5090",
) -> ProfileLibrary:
    return ProfileLibrary(str(library_path(model, dataset, machine)))


def default_library() -> ProfileLibrary:
    info = catalog()["default"]
    return load(info["model"], info["dataset"], info["machine"])


def default_path() -> Path:
    info = catalog()["default"]
    return _PROFILES / info["path"]


def _main() -> None:
    p = argparse.ArgumentParser(description="Search profile libraries")
    p.add_argument("--model", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--machine", default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--load", action="store_true")
    args = p.parse_args()
    hits = search(model=args.model, dataset=args.dataset, machine=args.machine, source=args.source)
    if not hits:
        print("no matches")
        return
    for h in hits:
        print(f"{h['model']:28s} {h['dataset']:12s} {h['machine']:6s}  "
              f"n={h['n_profiles']:3d}  {h['path']}")
    if args.load:
        h = hits[0]
        lib = load(h["model"], h["dataset"], h["machine"])
        print()
        print(lib.summary())


if __name__ == "__main__":
    _main()
