#!/usr/bin/env python3
"""P1.6: scalability of tree construction, metrics, and greedy selection.

Each measurement runs in a fresh process, so peak memory belongs to that operation alone.
Populations below the real size are random subsets of its variants. Larger ones replicate the real table
under renamed families, keeping its hierarchy shape. The full-kernel breadth greedy is
capped where its matrix would not fit reasonably in memory; capped cells are reported as such.

    .venv/bin/python analysis/p1_6_scalability.py            # runs the grid, writes the deliverable
    .venv/bin/python analysis/p1_6_scalability.py --worker N K OP   # one measurement, as JSON

Outputs in deliverables/<release>/P1_6_scalability/.
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import common
from common import OUTPUT, RELEASE, deliverable, foundation, markdown_table, save_figure

import numpy as np
import pandas as pd

HIERARCHY = ["family_id", "series_id", "base_model_id", "variant_id"]
BETA, LAMBDA0 = 0.15, 0.25
POPULATIONS = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
PANEL_SIZES = [5, 10, 20, 50, 100]
KERNEL_LIMIT = 32000
OPERATIONS = ["tree", "adoption_gap", "breadth", "structural", "coverage", "greedy_adoption", "greedy_breadth"]
COMPLEXITY = {
    "tree": ("O(V log V)", "O(V)"),
    "adoption_gap": ("O(k)", "O(E) once, for the target branch masses"),
    "breadth": ("O(k V)", "O(V)"),
    "structural": ("O(k + C)", "O(C)"),
    "coverage": ("O(k)", "O(V)"),
    "greedy_adoption": ("O(k V)", "O(V + E)"),
    "greedy_breadth": ("O(k V C) with a stored kernel", "O(V^2) kernel"),
}


def scaled_table(F, n: int) -> pd.DataFrame:
    """A variant table of about n rows with the real table's hierarchy shape."""
    table = F.variant[HIERARCHY + ["size_bucket", "model_role", "training_stage", "downloads_30d"]].copy()
    rng = np.random.default_rng(common.SEED)
    if n <= len(table):
        return table.iloc[np.sort(rng.choice(len(table), n, replace=False))].reset_index(drop=True)
    copies = int(np.ceil(n / len(table)))
    parts = []
    for copy in range(copies):
        part = table.copy()
        for column in HIERARCHY:
            part[column] = f"c{copy}::" + part[column].astype(str)
        parts.append(part)
    return pd.concat(parts, ignore_index=True).iloc[:n].reset_index(drop=True)


def build_tree(table: pd.DataFrame, pi: np.ndarray) -> Dict[str, Any]:
    """Same construction as TreeGeometry in the validation source, with the variant-count geometry."""
    n = len(table)
    paths = np.empty((n, 4), dtype=int)
    offset = 0
    for depth in range(4):
        labels = table[HIERARCHY[: depth + 1]].astype(str).agg(" / ".join, axis=1)
        codes, uniques = pd.factorize(labels, sort=True)
        paths[:, depth] = offset + codes
        offset += len(uniques)
    count = np.zeros(offset)
    for depth in range(4):
        np.add.at(count, paths[:, depth], 1.0)
    raw = np.zeros(offset)
    for depth in range(4):
        edge = paths[:, depth]
        parent = np.full(n, float(n)) if depth == 0 else count[paths[:, depth - 1]]
        ratio = np.clip(count[edge] / np.maximum(parent, 1e-15), 1e-15, 1.0)
        raw[edge] = (1.0 - BETA) * (-np.log(ratio)) + BETA * LAMBDA0
    scale = float(np.dot(pi, raw[paths].sum(axis=1)))
    lengths = raw / max(scale, 1e-15)
    cumulative = np.cumsum(lengths[paths], axis=1)
    return {"paths": paths, "lengths": lengths, "cumulative": cumulative, "root_distance": cumulative[:, -1], "n_edges": offset, "n": n}


def branch_mass(tree: Dict[str, Any], mass: np.ndarray) -> np.ndarray:
    out = np.zeros(tree["n_edges"])
    for depth in range(4):
        np.add.at(out, tree["paths"][:, depth], mass)
    return out


def distances_to(tree: Dict[str, Any], idx: int) -> np.ndarray:
    lca = np.zeros(tree["n"])
    same = np.ones(tree["n"], dtype=bool)
    for depth in range(4):
        same &= tree["paths"][:, depth] == tree["paths"][idx, depth]
        lca[same] = tree["cumulative"][same, depth]
    return np.maximum(tree["root_distance"] + tree["root_distance"][idx] - 2.0 * lca, 0.0)


def worker(population: int, k: int, operation: str) -> Dict[str, Any]:
    F = foundation()
    table = scaled_table(F, population)
    del F
    downloads = pd.to_numeric(table["downloads_30d"], errors="coerce").fillna(0.0).to_numpy(float)
    pi = 0.9 * downloads / max(downloads.sum(), 1.0) + 0.1 / len(table)
    pi /= pi.sum()
    rng = np.random.default_rng(common.SEED + population + k)
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    started = time.perf_counter()
    tree = build_tree(table, pi)
    tree_seconds = time.perf_counter() - started
    panel = rng.choice(len(table), min(k, len(table)), replace=False)
    tau = 0.43
    repeats = 1
    started = time.perf_counter()
    if operation == "tree":
        elapsed = tree_seconds
    elif operation == "adoption_gap":
        target_edge = branch_mass(tree, pi)
        base = float(np.dot(tree["lengths"], target_edge))
        started = time.perf_counter()
        repeats = 200
        for _ in range(repeats):
            edges, counts = np.unique(tree["paths"][panel].ravel(), return_counts=True)
            _ = base + float(np.dot(tree["lengths"][edges], np.abs(target_edge[edges] - counts / k) - target_edge[edges]))
        elapsed = (time.perf_counter() - started) / repeats
    elif operation == "breadth":
        best = np.zeros(tree["n"])
        for idx in panel:
            best = np.maximum(best, np.exp(-distances_to(tree, int(idx)) / tau))
        _ = float(np.dot(pi, best))
        elapsed = time.perf_counter() - started
    elif operation == "structural":
        cells = pd.factorize(table["size_bucket"].astype(str) + "|" + table["model_role"].astype(str))[0]
        pop = np.bincount(cells, weights=pi)
        started = time.perf_counter()
        repeats = 200
        for _ in range(repeats):
            _ = np.abs(np.bincount(cells[panel], minlength=len(pop)) / k - pop).max()
        elapsed = (time.perf_counter() - started) / repeats
    elif operation == "coverage":
        started = time.perf_counter()
        repeats = 200
        for _ in range(repeats):
            _ = float(pi[np.unique(panel)].sum())
        elapsed = (time.perf_counter() - started) / repeats
    elif operation == "greedy_adoption":
        target_edge = branch_mass(tree, pi)
        counts = np.zeros(tree["n_edges"])
        available = np.ones(tree["n"], dtype=bool)
        weight = 1.0 / k
        started = time.perf_counter()
        for _ in range(k):
            change = np.zeros(tree["n"])
            for depth in range(4):
                edge = tree["paths"][:, depth]
                change += tree["lengths"][edge] * (np.abs(target_edge[edge] - counts[edge] - weight) - np.abs(target_edge[edge] - counts[edge]))
            change[~available] = np.inf
            chosen = int(np.argmin(change))
            available[chosen] = False
            np.add.at(counts, tree["paths"][chosen], weight)
        elapsed = time.perf_counter() - started
    elif operation == "greedy_breadth":
        kernel = np.empty((tree["n"], tree["n"]), dtype=np.float32)
        for i in range(tree["n"]):
            kernel[i] = np.exp(-distances_to(tree, i) / tau).astype(np.float32)
        kernel_seconds = time.perf_counter() - started
        weights = pi.astype(np.float32)
        best = np.zeros(tree["n"], dtype=np.float32)
        available = np.ones(tree["n"], dtype=bool)
        started = time.perf_counter()
        for _ in range(k):
            gains = ((np.maximum(kernel, best[:, None]) - best[:, None]) * weights[:, None]).sum(axis=0)
            gains[~available] = -np.inf
            chosen = int(np.argmax(gains))
            available[chosen] = False
            best = np.maximum(best, kernel[:, chosen])
        elapsed = time.perf_counter() - started + kernel_seconds
    else:
        raise ValueError(operation)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"population_variants": len(table), "edges": tree["n_edges"], "panel_size": k, "operation": operation,
            "seconds": elapsed, "tree_seconds": tree_seconds, "peak_rss_mb": peak / 1024.0, "extra_rss_mb": (peak - baseline_rss) / 1024.0}


def validate_tree() -> float:
    """The tree built here must match the validated geometry on the real population."""
    F = foundation()
    table = F.variant[HIERARCHY].copy()
    tree = build_tree(table, F.PI_A)
    return float(np.abs(np.sort(tree["lengths"]) - np.sort(F.lengths)).max())


def run_one(args: List[str], timeout: int = 900) -> Dict[str, Any]:
    result = subprocess.run([sys.executable, __file__, "--worker", *args], capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-400:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> None:
    OUT = deliverable("P1_6_scalability")
    tree_error = validate_tree()
    assert tree_error < 1e-9, f"tree differs from the validated geometry by {tree_error}"
    rows: List[Dict[str, Any]] = []
    for population in POPULATIONS:
        for operation in OPERATIONS:
            sizes = PANEL_SIZES if operation in {"adoption_gap", "breadth", "structural", "coverage", "greedy_adoption", "greedy_breadth"} else [10]
            for k in sizes:
                if operation == "greedy_breadth" and population > KERNEL_LIMIT:
                    rows.append({"population_variants": population, "panel_size": k, "operation": operation, "status": "skipped: kernel over limit"})
                    continue
                try:
                    row = run_one([str(population), str(k), operation])
                    row["status"] = "measured"
                except Exception as error:
                    row = {"population_variants": population, "panel_size": k, "operation": operation, "status": f"failed: {type(error).__name__}"}
                rows.append(row)
                print(population, k, operation, row.get("seconds"), row.get("peak_rss_mb"), row["status"], flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "scalability_grid.csv", index=False)

    stage_rows = []
    summary_path = common.ROOT / "logs" / f"run_summary_{RELEASE}.json"
    if summary_path.exists():
        for step in json.loads(summary_path.read_text()):
            if step["step"] in {"clean", "lineage", "validation", "appendix_c"}:
                stage_rows.append({"stage": step["step"], "seconds": step["seconds"]})
    stages = pd.DataFrame(stage_rows)
    if len(stages):
        stages.to_csv(OUT / "pipeline_stage_times.csv", index=False)

    write_outputs(grid, stages, OUT, tree_error)
    print((OUT / "REPORT.md").read_text(encoding="utf-8"))


def write_outputs(grid: pd.DataFrame, stages: pd.DataFrame, OUT: Path, tree_error: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measured = grid[grid["status"] == "measured"]
    k20 = measured[(measured["panel_size"] == 20) | ((measured["operation"] == "tree"))]
    table = k20.pivot_table(index="population_variants", columns="operation", values="seconds").reindex(columns=OPERATIONS)
    memory = k20.pivot_table(index="population_variants", columns="operation", values="peak_rss_mb").reindex(columns=OPERATIONS)
    by_panel = measured[measured["population_variants"] == measured["population_variants"].where(measured["population_variants"] <= 8000).max()]
    by_panel_table = by_panel.pivot_table(index="panel_size", columns="operation", values="seconds").reindex(columns=[c for c in OPERATIONS if c != "tree"])
    complexity = pd.DataFrame([{"operation": op, "time": COMPLEXITY[op][0], "memory": COMPLEXITY[op][1]} for op in OPERATIONS])
    table.reset_index().to_csv(OUT / "time_by_population_k20.csv", index=False)
    memory.reset_index().to_csv(OUT / "memory_by_population_k20.csv", index=False)
    by_panel_table.reset_index().to_csv(OUT / "time_by_panel_size.csv", index=False)
    complexity.to_csv(OUT / "complexity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    colors = {"tree": "#1f77b4", "adoption_gap": "#d95f02", "breadth": "#2a9d4b", "structural": "#8c6bb1", "coverage": "#7f7f7f",
              "greedy_adoption": "#e377c2", "greedy_breadth": "#17becf"}
    for operation in OPERATIONS:
        part = table[operation].dropna()
        axes[0].plot(part.index, part.values, marker="o", markersize=4, linewidth=1.6, color=colors[operation], label=operation)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Canonical variants in the population"); axes[0].set_ylabel("Seconds, panel of 20")
    axes[0].grid(True, color="#e6e6e6", linewidth=0.5)
    # Tree and greedy adoption use the same memory, so the tree line is dashed to stay visible.
    for operation in ["greedy_adoption", "tree", "greedy_breadth"]:
        part = memory[operation].dropna()
        axes[1].plot(part.index, part.values, marker="o", markersize=4, linewidth=1.6, color=colors[operation],
                     linestyle="--" if operation == "tree" else "-")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Canonical variants in the population"); axes[1].set_ylabel("Peak process memory (MB)")
    axes[1].grid(True, color="#e6e6e6", linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8)
    fig.suptitle("Scaling of the coverage computations", fontsize=11)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    save_figure(fig, OUT, "scalability")
    plt.close(fig)

    stage_block = markdown_table(stages) if len(stages) else "Recorded once the release runs through `run_all_experiments.py`."
    text = f"""# P1.6 Scalability, release {RELEASE}

## Time by population size, panel of 20 (seconds)

{markdown_table(table.reset_index(), digits=5)}

`adoption_gap`, `structural`, and `coverage` are per panel after one setup. `tree` also reports construction for all
{int(measured['edges'].max()) if 'edges' in measured else 0:,} edges in the largest case.

## Peak process memory, panel of 20 (MB)

Includes loading the foundation cache and pandas, which accounts for the floor in small populations.

{markdown_table(memory.reset_index(), digits=0)}

## Time by panel size, real-scale population (seconds)

{markdown_table(by_panel_table.reset_index(), digits=5)}

## Complexity

V is the number of variants, E the edges (about 1.8 V here), k the panel size, and C the candidate pool or structural cells.

{markdown_table(complexity)}

## Whole-pipeline stages on real data (seconds)

{stage_block}

## Notes

- Measurements are single-threaded (BLAS limited to one thread) on a shared 64-core machine, each in a fresh process.
- Populations above the real size replicate the real table under renamed families. They test computation, not realism.
- The tree builder here matches the validated geometry on the real population, to within {tree_error:.1e}.
- The breadth greedy stores a full V by V kernel, which limits it to about {KERNEL_LIMIT:,} variants here (4 bytes times V squared, about
  {4 * KERNEL_LIMIT ** 2 / 1e9:.0f} GB). Since similarity depends only on the lowest common ancestor, it could use per-edge aggregation instead, but that
  has not been implemented. Larger cells are marked skipped in `scalability_grid.csv`.
"""
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        print(json.dumps(worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])))
    else:
        main()
