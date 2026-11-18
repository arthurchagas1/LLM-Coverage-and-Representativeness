"""Helpers shared by the research-question analyses (rq1_audit.py, rq2_sensitivity.py, rq3_construction.py).

The adoption gap of many panels against one target is computed from the target's branch masses, touching only
the edges on the panel's paths, which makes thousands of reference panels per paper cheap. `check_against_w1`
verifies the shortcut against the foundation's own W1 before any result is written.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Sequence, Tuple

import common
from common import OUTPUT

import numpy as np
import pandas as pd

import build_appendix_c as baseline  # noqa: E402

C4_SEED = baseline.SEED          # seed of the appendix C reference panels
N_REFERENCES = baseline.DEFAULT_N_NULL


class GapScorer:
    """Adoption gap W1(target, sigma) on the global tree, for a fixed target and edge lengths."""

    def __init__(self, F, target: np.ndarray, lengths: np.ndarray | None = None) -> None:
        self.paths = F.paths
        self.lengths = F.lengths if lengths is None else np.asarray(lengths, float)
        self.T = F.branch_mass(target)
        self.base = float(np.dot(self.lengths, np.abs(self.T)))

    def gap(self, idx: np.ndarray, weights: np.ndarray) -> float:
        edges = self.paths[np.asarray(idx, int)].ravel()
        w = np.repeat(np.asarray(weights, float), self.paths.shape[1])
        unique, inverse = np.unique(edges, return_inverse=True)
        mass = np.bincount(inverse, weights=w)
        t = self.T[unique]
        return self.base + float(np.dot(self.lengths[unique], np.abs(t - mass) - np.abs(t)))

    def gap_uniform(self, idx: np.ndarray) -> float:
        idx = np.asarray(idx, int)
        return self.gap(idx, np.full(len(idx), 1.0 / len(idx)))


def sigma_parts(entries: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Variant indices and masses of sigma_P: one unit per entry, spread by its posterior."""
    idx = np.concatenate([np.asarray(e["candidate_idx"], int) for e in entries])
    w = np.concatenate([np.asarray(e["posterior"], float) / float(np.sum(e["posterior"])) for e in entries]) / len(entries)
    return idx, w


def regime_entries(entries: Sequence[Dict[str, Any]], regime: str) -> List[Dict[str, Any]]:
    """Resolution regimes of src/build_resolver_ablation.py."""
    if regime == "posterior":
        return list(entries)
    if regime == "exact_only":
        return [e for e in entries if len(e["candidate_idx"]) == 1]
    if regime == "descendant_expansion":
        return [{"candidate_idx": np.array([int(c)]), "posterior": np.array([1.0])}
                for e in entries for c in e["candidate_idx"]]
    raise ValueError(regime)


def paper_positions(F) -> Dict[str, int]:
    """Position of each paper in the foundation's panel order, which seeds the appendix C reference panels."""
    return {pid: position for position, pid in enumerate(F.panels)}


def uniform_references(support_idx: np.ndarray, n: int, position: int, draws: int = N_REFERENCES) -> List[np.ndarray]:
    """The uniform reference panels of appendix C: same seed, same order, so the primary percentiles match."""
    rng = np.random.default_rng(C4_SEED + 1009 * position + len("uniform"))
    replace = len(support_idx) < n
    return [rng.choice(support_idx, n, replace=replace) for _ in range(draws)]


def adoption_references(support_idx: np.ndarray, target: np.ndarray, n: int, position: int,
                        draws: int = N_REFERENCES) -> List[np.ndarray]:
    rng = np.random.default_rng(C4_SEED + 1009 * position + len("adoption_weighted"))
    p = target[support_idx] / target[support_idx].sum()
    replace = len(support_idx) < n
    return [rng.choice(support_idx, n, replace=replace, p=p) for _ in range(draws)]


def percentile_no_better(observed_gap: float, reference_gaps: Sequence[float]) -> float:
    """Share of reference panels whose gap is at least the observed one (larger is better for the panel)."""
    return float(np.mean(np.asarray(reference_gaps) >= observed_gap))


def load_contexts() -> Dict[str, Dict[str, Any]]:
    with (OUTPUT / "paper_contexts.pkl").open("rb") as handle:
        return pickle.load(handle)


def scoreable(F) -> List[str]:
    return [pid for pid, panel in F.panels.items() if panel["entries"]]


def check_against_w1(F, contexts: Dict[str, Dict[str, Any]]) -> float:
    """Largest absolute difference between the shortcut and F.w1 over all observed panels."""
    worst = 0.0
    for pid in scoreable(F):
        eligible = np.asarray(contexts[pid]["eligible"], bool)
        target = baseline.target_adoption(F, eligible)
        entries = F.panels[pid]["entries"]
        fast = GapScorer(F, target).gap(*sigma_parts(entries))
        slow = float(F.w1(target, F.sigma(entries)))
        worst = max(worst, abs(fast - slow))
    if worst > 1e-9:
        raise RuntimeError(f"Fast adoption gap disagrees with F.w1 by {worst:.3g}")
    return worst


def rank_table(values: pd.Series) -> pd.Series:
    return values.rank(method="average")
