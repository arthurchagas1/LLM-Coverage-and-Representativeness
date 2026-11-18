#!/usr/bin/env python3
"""Compare regenerated results with the committed ones, numerically.

    python scripts/compare_results.py                      # against the last commit (git HEAD)
    python scripts/compare_results.py --reference DIR      # against another checkout or a copy of the results
    python scripts/compare_results.py --release 20261105_rolefix --reference DIR

It checks every CSV, JSON, Markdown, TeX and text file under deliverables/<release>/, outputs/<release>/ (tables,
appendix_c, diagnostics, counterfactual_panels) and annotation/results/. CSV cells are compared as numbers when both
sides parse as numbers (relative tolerance --rtol, absolute --atol) and as strings otherwise. Images and PDFs are
only checked for existence, since their bytes embed creation dates.

Wall-clock timings are machine dependent, so they are reported but never counted as mismatches: the whole
P1_6_scalability folder, pipeline_stage_times.csv, and keys or columns named like *seconds*, *runtime*, *time_s*.
Exit status 1 means at least one other file differs.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEXT = {".csv", ".json", ".md", ".tex", ".txt"}
BINARY = {".png", ".pdf", ".parquet", ".pkl"}
TIMING_PATH = re.compile(r"P1_6_scalability/|pipeline_stage_times|run_summary")
TIMING_NAME = re.compile(r"seconds|runtime|time_s\b|wall|elapsed", re.I)


def tracked_files(release: str, reference: Path | None) -> list[str]:
    roots = [f"deliverables/{release}", f"outputs/{release}/tables", f"outputs/{release}/appendix_c",
             f"outputs/{release}/diagnostics", f"outputs/{release}/counterfactual_panels", "annotation/results"]
    if reference is None:
        out = subprocess.run(["git", "ls-files", *roots], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return [line for line in out.splitlines() if line]
    return sorted(str(p.relative_to(reference)) for r in roots for p in (reference / r).rglob("*") if p.is_file())


def read_reference(path: str, reference: Path | None) -> bytes | None:
    if reference is not None:
        target = reference / path
        return target.read_bytes() if target.exists() else None
    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else None


def compare_csv(old: bytes, new: bytes, rtol: float, atol: float) -> str | None:
    a = pd.read_csv(io.BytesIO(old), dtype=str, keep_default_na=False)
    b = pd.read_csv(io.BytesIO(new), dtype=str, keep_default_na=False)
    if list(a.columns) != list(b.columns):
        return f"columns differ: {sorted(set(a.columns) ^ set(b.columns))}"
    if len(a) != len(b):
        return f"rows {len(a)} -> {len(b)}"
    problems = []
    for column in a.columns:
        if TIMING_NAME.search(column):
            continue
        x, y = a[column], b[column]
        if x.equals(y):
            continue
        xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        numeric = xn.notna() & yn.notna()
        text_bad = (~numeric) & (x != y)
        close = np.isclose(xn[numeric].astype(float), yn[numeric].astype(float), rtol=rtol, atol=atol, equal_nan=True)
        num_bad = int((~close).sum())
        if num_bad or text_bad.any():
            worst = float(np.nanmax(np.abs(xn[numeric].astype(float) - yn[numeric].astype(float)))) if numeric.any() else 0.0
            problems.append(f"{column}: {num_bad + int(text_bad.sum())} cells (max abs diff {worst:.3g})")
    return "; ".join(problems[:5]) if problems else None


def strip_timings(value):
    if isinstance(value, dict):
        return {k: strip_timings(v) for k, v in value.items() if not TIMING_NAME.search(str(k))}
    if isinstance(value, list):
        return [strip_timings(v) for v in value]
    return value


def compare_json(old: bytes, new: bytes, rtol: float, atol: float) -> str | None:
    def flat(value, prefix=""):
        if isinstance(value, dict):
            for k, v in value.items():
                yield from flat(v, f"{prefix}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                yield from flat(v, f"{prefix}[{i}]")
        else:
            yield prefix, value
    a = dict(flat(strip_timings(json.loads(old))))
    b = dict(flat(strip_timings(json.loads(new))))
    bad = []
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if not np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
                bad.append(key)
        elif x != y:
            bad.append(key)
    return f"{len(bad)} keys differ, e.g. {bad[:3]}" if bad else None


def compare_text(old: bytes, new: bytes) -> str | None:
    """Text reports print rounded numbers, so exact equality is the right test; timings lines are ignored."""
    keep = lambda raw: [l for l in raw.decode("utf-8", "replace").splitlines() if not TIMING_NAME.search(l)]
    a, b = keep(old), keep(new)
    if a == b:
        return None
    changed = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    first = next((f"{x!r} -> {y!r}" for x, y in zip(a, b) if x != y), "")
    return f"{changed} lines differ; first: {first[:200]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", default="20261105")
    parser.add_argument("--reference", type=Path, help="directory with the reference results (default: git HEAD)")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-9)
    args = parser.parse_args()
    reference = args.reference.resolve() if args.reference else None

    files = tracked_files(args.release, reference)
    if not files:
        raise SystemExit("No reference files found.")
    mismatches, timing_only, missing, same = [], [], [], 0
    for path in files:
        suffix = Path(path).suffix.lower()
        current = ROOT / path
        if not current.exists():
            missing.append(path)
            continue
        if suffix in BINARY or suffix not in TEXT:
            same += 1
            continue
        old = read_reference(path, reference)
        new = current.read_bytes()
        if old is None or old == new:
            same += 1
            continue
        if suffix == ".csv":
            problem = compare_csv(old, new, args.rtol, args.atol)
        elif suffix == ".json":
            problem = compare_json(old, new, args.rtol, args.atol)
        else:
            problem = compare_text(old, new)
        if problem is None:
            same += 1
        elif TIMING_PATH.search(path):
            timing_only.append((path, problem))
        else:
            mismatches.append((path, problem))

    print(f"{len(files)} reference files: {same} match, {len(mismatches)} differ, {len(missing)} missing, "
          f"{len(timing_only)} timing-only differences")
    for path, problem in timing_only:
        print(f"  timing   {path}: {problem}")
    for path in missing:
        print(f"  MISSING  {path}")
    for path, problem in mismatches:
        print(f"  DIFFERS  {path}: {problem}")
    sys.exit(1 if mismatches or missing else 0)


if __name__ == "__main__":
    main()
