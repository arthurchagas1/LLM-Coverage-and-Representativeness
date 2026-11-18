#!/usr/bin/env python3
"""Single entry point: build one data release and every result from its cleaned Hub population.

    python run_all_experiments.py                      # every offline step of the paper release
    python run_all_experiments.py --resume             # continue after a crash, same command every time
    python run_all_experiments.py --from rq1           # rerun from a step, and everything after it
    python run_all_experiments.py --only rq4 annotation
    python run_all_experiments.py --release 20261105_rolefix
    python run_all_experiments.py --with-clean         # also rebuild the cleaned population from the raw snapshot
    python run_all_experiments.py --network            # also refresh the network caches (Hub configs, arXiv dates, OpenAlex)
    python run_all_experiments.py --list

By default the run is offline and deterministic: the steps that call the network (`configs`, `paper_dates`) are
skipped, OpenAlex answers come from data/corpus_frame_cache.json only, and the cleaned population shipped in
data/oome_clean_<release>/ is the starting point. `--with-clean` first rebuilds that population from the raw snapshot
in data/hf_snapshots_<release>/ (about 65 min).

Each step writes logs/<release>/<step>.log, and the run stops at the first failure.
logs/run_summary_<release>.json records status and wall time per step. Collecting a new snapshot is separate
(data_collection/coleta.py), because a new snapshot is a new release. See README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import release_config

ROOT = Path(__file__).resolve().parent

# (name, what it produces, arguments after the interpreter, extra environment, kind)
# kind: "" always runs; "clean" needs the raw snapshot (--with-clean); "network" refreshes a cache (--network).
STEPS = [
    ("clean", "filter the raw Hub snapshot to the textual population", ["data_collection/clean.py"], {}, "clean"),
    ("lineage", "family, series, base, and variant for each repository", ["data_collection/p4_families.py"], {}, ""),
    ("validation", "resolver, panels, targets, simulations, foundation cache", ["src/build_foundation.py"], {"COVERAGE_FAST_MODE": "0"}, ""),
    ("foundation_check", "cache reproduces the source metrics", ["src/validate_foundation.py"], {}, ""),
    ("figures", "adoption-gap figure from the validation table", ["make_figures.py"], {}, ""),
    ("configs", "context length of representative repositories (network, cached)", ["data_collection/fetch_repo_configs.py"], {}, "network"),
    ("paper_dates", "first public dates and full text of the papers (network, cached)", ["data_collection/fetch_paper_dates.py"], {}, "network"),
    ("diag_papers", "per-paper diagnostics", ["src/build_paper_diagnostics.py"], {}, ""),
    ("diag_resolver", "resolver sensitivity", ["src/build_resolver_ablation.py"], {}, ""),
    ("diag_counterfactual", "same-budget swaps", ["src/build_counterfactual_panels.py"], {}, ""),
    ("diag_temporal_tables", "time-consistent null calibration", ["src/build_temporal_tables.py"], {}, ""),
    ("diag_temporal_audit", "temporal audit", ["src/build_temporal_audit.py"], {}, ""),
    ("feasibility", "F_P by paper and the shared cutoffs (paper_contexts.pkl)", ["analysis/f_feasibility.py"], {}, ""),
    ("appendix_c", "eligibility, audit, and claim-conditioned populations, on the F_P cutoffs", ["build_appendix_c.py"], {}, ""),
    ("p1_1", "validation under model reuse", ["analysis/p1_1_validation.py"], {}, ""),
    ("p1_2", "adoption-target sensitivity", ["analysis/p1_2_target_sensitivity.py"], {}, ""),
    ("p1_3", "temporal reconstruction", ["analysis/p1_3_temporal.py"], {}, ""),
    ("p1_4", "value of the Wasserstein guarantee", ["analysis/p1_4_bound.py"], {}, ""),
    ("p1_5", "auditable sample of papers", ["analysis/p1_5_corpus.py"], {}, ""),
    ("p1_6", "scalability (wall-clock timings, machine dependent)", ["analysis/p1_6_scalability.py"], {}, ""),
    ("rq1", "RQ1 coverage of existing panels", ["analysis/rq1_audit.py"], {}, ""),
    ("rq2", "RQ2 sensitivity to modelling choices", ["analysis/rq2_sensitivity.py"], {}, ""),
    ("rq3", "RQ3 coverage-aware construction", ["analysis/rq3_construction.py"], {}, ""),
    ("rq4", "RQ4 behavioral validation with fixed-inclusion designs", ["analysis/rq4_behavioral.py"], {}, ""),
    ("paper_figures", "figures of the paper (notebooks/paper_figures.ipynb)", ["scripts/run_notebook.py", "notebooks/paper_figures.ipynb"], {}, ""),
    ("annotation", "human validation of the taxonomy: agreement and accuracy", ["annotation/agreement.py"], {}, ""),
]


def run_step(name: str, args: list[str], environment: dict[str, str], log_dir: Path) -> float:
    log_path = log_dir / f"{name}.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - started
    if result.returncode != 0:
        print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]))
        raise SystemExit(f"step '{name}' failed (exit {result.returncode}); see {log_path}")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    names = [step[0] for step in STEPS]
    parser.add_argument("--release", default=release_config.DEFAULT_RELEASE, choices=list(release_config.RELEASES))
    parser.add_argument("--only", nargs="+", choices=names, help="run just these steps")
    parser.add_argument("--from", dest="start", choices=names, help="start from this step")
    parser.add_argument("--skip", nargs="+", choices=names, default=[], help="skip these steps")
    parser.add_argument("--resume", action="store_true", help="skip the steps this release has already completed")
    parser.add_argument("--with-clean", action="store_true", help="rebuild the cleaned population from the raw snapshot")
    parser.add_argument("--network", action="store_true", help="refresh the network caches (results may then change)")
    parser.add_argument("--list", action="store_true", help="list the steps and exit")
    args = parser.parse_args()

    summary_path = ROOT / "logs" / f"run_summary_{args.release}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    done = {row["step"] for row in summary if row.get("status") == "ok"}

    if args.list:
        for name, description, _, _, kind in STEPS:
            flag = {"clean": "  [--with-clean]", "network": "  [--network]"}.get(kind, "")
            print(f"{'done' if name in done else '    '}  {name:22s} {description}{flag}")
        return

    selected = STEPS
    if args.start:
        selected = STEPS[names.index(args.start):]
        # Later steps build on this one, so their earlier completion no longer counts.
        rerun = {s[0] for s in selected}
        summary = [row for row in summary if row["step"] not in rerun]
        summary_path.parent.mkdir(exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.only:
        selected = [s for s in STEPS if s[0] in args.only]
    else:
        selected = [s for s in selected if (s[4] != "clean" or args.with_clean) and (s[4] != "network" or args.network)]
    selected = [s for s in selected if s[0] not in args.skip]
    if args.resume:
        selected = [s for s in selected if s[0] not in done]

    environment = {**os.environ, **release_config.release_env(args.release), "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
                   "PYTHONHASHSEED": "0", "DMKD_OFFLINE": "0" if args.network else "1"}
    snapshot = Path(environment["HF_SNAPSHOT_DIR"])
    if any(s[4] == "clean" for s in selected) and not (list(snapshot.glob("*.flat.parquet")) or list(snapshot.glob("*.flat.part-*.parquet"))):
        raise SystemExit(f"--with-clean needs the raw snapshot in {snapshot}; see data/README.md")
    log_dir = ROOT / "logs" / args.release
    log_dir.mkdir(parents=True, exist_ok=True)
    for name, description, command, extra, _ in selected:
        print(f"[run] {name:22s} {description}", flush=True)
        elapsed = run_step(name, command, {**environment, **extra}, log_dir)
        print(f"[ok ] {name:22s} {elapsed:8.1f}s", flush=True)
        summary = [row for row in summary if row["step"] != name] + [{"step": name, "seconds": round(elapsed, 1), "status": "ok"}]
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAll requested steps completed for release {args.release}.")


if __name__ == "__main__":
    main()
