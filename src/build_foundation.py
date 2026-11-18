"""Stage 0: extract a picklable *foundation* from the validated source notebook.

The existing ``coverage_metric_validation_source.py`` is the single source of
truth for the population, lineage tree, probabilistic resolver, and per-paper
panels. It is written as a notebook and rebuilds everything (including the slow
null/external blocks) on import. This stage imports it once, then dumps only the
primitive arrays / frames / dicts the downstream diagnostics need, so those
stages run in seconds and never re-touch the heavy pipeline.

Crucially we store *primitives* (numpy arrays, DataFrames, plain dicts), never
instances of classes defined inside the source module. That keeps the pickle
loadable without re-importing (hence re-running) the source. Pure-numpy
re-implementations of the tree/panel math live in ``coverage_geometry.py`` and
are validated against the ground-truth metric table dumped here.
"""

from __future__ import annotations

import builtins
import os
import pickle
import sys
from pathlib import Path

from project_paths import CACHE, WORK_ROOT  # noqa: F401
CACHE.mkdir(parents=True, exist_ok=True)


def build() -> dict:
    os.environ.setdefault("COVERAGE_FAST_MODE", "0")
    import matplotlib
    matplotlib.use("Agg")
    # the source was authored for Jupyter; neutralise display() before import
    builtins.display = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.path.insert(0, str(WORK_ROOT))
    import coverage_metric_validation_source as C  # noqa: E402

    import numpy as np
    import pandas as pd

    # --- adoption proxies not loaded by the source (re-read from parquet) ------
    proxy_cols = ["variant_id", "downloads_all_time", "trending_score", "is_multimodal_leak"]
    proxy_raw = pd.read_parquet(C.LINEAGE_FILE, columns=proxy_cols)
    proxy_raw = proxy_raw[~proxy_raw["is_multimodal_leak"].fillna(False)].copy()
    for col in ["downloads_all_time", "trending_score"]:
        proxy_raw[col] = pd.to_numeric(proxy_raw[col], errors="coerce").fillna(0.0)
    proxy_by_variant = proxy_raw.groupby("variant_id")[
        ["downloads_all_time", "trending_score"]
    ].sum()
    order = C.variant["variant_id"].to_numpy()
    downloads_all_time = proxy_by_variant.reindex(order)["downloads_all_time"].fillna(0.0).to_numpy(float)
    trending = proxy_by_variant.reindex(order)["trending_score"].fillna(0.0).to_numpy(float)

    # --- per-paper panels reduced to plain entry dicts ------------------------
    panels = {}
    for pid, panel in C.paper_panels.items():
        entries = [
            {
                "candidate_idx": np.asarray(e.candidate_idx, dtype=int),
                "posterior": np.asarray(e.posterior, dtype=float),
                "status": e.status,
                "match_type": e.match_type,
                "input_model": e.input_model,
            }
            for e in panel.resolved_entries
        ]
        panels[pid] = {
            "entries": entries,
            "n_input": panel.n_input,
            "n_resolved": panel.n_resolved,
        }

    # --- four geometries reduced to distance primitives -----------------------
    geometries = {}
    for name, g in C.geometry_family.items():
        geometries[name] = {
            "lengths": np.asarray(g.lengths, float),
            "root_distance": np.asarray(g.root_distance, float),
            "cumulative": np.asarray(g.cumulative, float),
            "tau": float(g.tau),
        }

    # --- structural-discrepancy partitions (against pi_b) ---------------------
    discrepancy = {
        "codes": {k: np.asarray(v, int) for k, v in C.discrepancy_b.codes.items()},
        "labels": {k: np.asarray(v, dtype=object) for k, v in C.discrepancy_b.labels.items()},
        "pop": {k: np.asarray(v, float) for k, v in C.discrepancy_b.pop.items()},
    }

    foundation = {
        "variant": C.variant.copy(),
        "PI_A": np.asarray(C.PI_A, float),
        "PI_B": np.asarray(C.PI_B, float),
        "UNIFORM": np.asarray(C.UNIFORM, float),
        "DOWNLOAD_PROB": np.asarray(C.DOWNLOAD_PROB, float),
        "DOWNLOADS": np.asarray(C.DOWNLOADS, float),
        "LIKES": np.asarray(C.LIKES, float),
        "DOWNLOADS_ALL_TIME": downloads_all_time,
        "TRENDING": trending,
        # tree primitives (primary geometry)
        "paths": np.asarray(C.tree.paths, int),
        "n_edges": int(C.tree.n_edges),
        "lengths": np.asarray(C.tree.lengths, float),
        "root_distance": np.asarray(C.tree.root_distance, float),
        "cumulative": np.asarray(C.tree.cumulative, float),
        "tau": float(C.tree.tau),
        "edge_labels": list(C.tree.edge_labels),
        "edge_depth": np.asarray(C.tree.edge_depth, int),
        "geometries": geometries,
        "discrepancy_b": discrepancy,
        "panels": panels,
        "paper_metadata": C.paper_metadata.copy(),
        "paper_posterior": C.paper_posterior.copy(),
        # ground truth for validating the standalone re-implementations
        "gt_paper_metrics": C.paper_metrics.copy(),
        "gt_geometry_scores": C.geometry_scores.copy(),
        # constants
        "ALPHA": float(C.ALPHA),
        "BETA": float(C.BETA),
        "LAMBDA0": float(C.LAMBDA0),
        "Q_GRID": np.asarray(C.Q_GRID, float),
        "SEED": int(C.SEED),
        "HIERARCHY": list(C.HIERARCHY),
    }
    return foundation


def main() -> None:
    foundation = build()
    out = CACHE / "foundation.pkl"
    with out.open("wb") as fh:
        pickle.dump(foundation, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print("foundation written:", out)
    print("variants:", len(foundation["variant"]), "papers:", len(foundation["panels"]))


if __name__ == "__main__":
    main()
