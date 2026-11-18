"""Standalone, dependency-light re-implementation of the tree / panel math.

Every function here is a faithful port of the corresponding method in
``coverage_metric_validation_source.py`` but operates purely on the primitive
arrays dumped by ``build_foundation.py``. Fidelity is enforced by
``validate_foundation.py``, which asserts these reproduce the source's
per-paper metric table to numerical tolerance. Downstream diagnostics depend
only on this module, never on the heavy source notebook.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from project_paths import FOUNDATION_PKL, WORK_ROOT  # noqa: F401
N_HIER = 4  # family -> series -> base -> variant


def load_foundation(path: Path = FOUNDATION_PKL) -> "Foundation":
    with Path(path).open("rb") as fh:
        return Foundation(pickle.load(fh))


class Foundation:
    """Wraps the primitive foundation and exposes the validated metric math."""

    def __init__(self, data: Dict):
        self.data = data
        self.variant: pd.DataFrame = data["variant"]
        self.n = len(self.variant)
        self.PI_A = data["PI_A"]
        self.PI_B = data["PI_B"]
        self.UNIFORM = data["UNIFORM"]
        self.DOWNLOAD_PROB = data["DOWNLOAD_PROB"]
        self.DOWNLOADS = data["DOWNLOADS"]
        self.LIKES = data["LIKES"]
        self.DOWNLOADS_ALL_TIME = data["DOWNLOADS_ALL_TIME"]
        self.TRENDING = data["TRENDING"]
        self.paths = data["paths"]
        self.n_edges = data["n_edges"]
        self.lengths = data["lengths"]
        self.cumulative = data["cumulative"]
        self.root_distance = data["root_distance"]
        self.tau = data["tau"]
        self.geometries = data["geometries"]
        self.discrepancy = data["discrepancy_b"]
        self.panels = data["panels"]
        self.Q_GRID = data["Q_GRID"]
        self.ALPHA = data["ALPHA"]
        self.BETA = data["BETA"]
        self.LAMBDA0 = data["LAMBDA0"]
        self.SEED = data["SEED"]
        self._dist_cache: Dict[tuple, np.ndarray] = {}

    # ---- tree primitives ----------------------------------------------------
    def branch_mass(self, leaf_mass: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_edges)
        leaf_mass = np.asarray(leaf_mass, float)
        for depth in range(N_HIER):
            np.add.at(out, self.paths[:, depth], leaf_mass)
        return out

    def w1(self, a: np.ndarray, b: np.ndarray, geom: str = "primary") -> float:
        lengths = self.lengths if geom == "primary" else self.geometries[geom]["lengths"]
        return float(np.dot(lengths, np.abs(self.branch_mass(a) - self.branch_mass(b))))

    def distances_to(self, idx: int, geom: str = "primary") -> np.ndarray:
        idx = int(idx)
        key = (geom, idx)
        if key in self._dist_cache:
            return self._dist_cache[key]
        if geom == "primary":
            cumulative, root_distance = self.cumulative, self.root_distance
        else:
            cumulative = self.geometries[geom]["cumulative"]
            root_distance = self.geometries[geom]["root_distance"]
        lca = np.zeros(self.n)
        same = np.ones(self.n, dtype=bool)
        for depth in range(N_HIER):
            same &= self.paths[:, depth] == self.paths[idx, depth]
            lca[same] = cumulative[same, depth]
        dist = np.maximum(root_distance + root_distance[idx] - 2.0 * lca, 0.0)
        self._dist_cache[key] = dist
        return dist

    # ---- panel measures -----------------------------------------------------
    def sigma(self, entries: List[Dict]) -> np.ndarray:
        out = np.zeros(self.n)
        m = len(entries)
        if m == 0:
            return out
        for e in entries:
            out[e["candidate_idx"]] += e["posterior"] / m
        return out

    def inclusion_probability(self, entries: List[Dict]) -> np.ndarray:
        not_seen = np.ones(self.n)
        for e in entries:
            not_seen[e["candidate_idx"]] *= 1.0 - e["posterior"]
        return 1.0 - not_seen

    def hill_spectrum(self, entries: List[Dict], target: Optional[np.ndarray] = None) -> Dict[str, float]:
        target = self.PI_A if target is None else target
        inclusion = self.inclusion_probability(entries)
        out: Dict[str, float] = {}
        for q in self.Q_GRID:
            weights = np.ones(self.n) if q == 0 else target ** q
            out[f"coverage_q{q:g}"] = float(np.dot(inclusion, weights) / weights.sum())
        out["coverage_delta_1_0"] = out["coverage_q1"] - out["coverage_q0"]
        return out

    def breadth(
        self,
        entries: List[Dict],
        target: Optional[np.ndarray] = None,
        support_subset: Optional[np.ndarray] = None,
        geom: str = "primary",
    ) -> float:
        if not entries:
            return float("nan")
        target = self.PI_B if target is None else target
        tau = self.tau if geom == "primary" else self.geometries[geom]["tau"]
        rows = np.arange(self.n) if support_subset is None else np.asarray(support_subset, int)
        target_local = target[rows]
        target_local = target_local / target_local.sum()
        best = np.zeros(len(rows))
        for e in entries:
            expected = np.zeros(len(rows))
            for idx, prob in zip(e["candidate_idx"], e["posterior"]):
                expected += prob * np.exp(-self.distances_to(int(idx), geom)[rows] / tau)
            best = np.maximum(best, expected)
        return float(np.dot(target_local, best))

    def structural(self, sigma: np.ndarray, target: str = "pi_b") -> Dict[str, object]:
        rows = []
        for name, codes in self.discrepancy["codes"].items():
            pop = self.discrepancy["pop"][name]
            labels = self.discrepancy["labels"][name]
            observed = np.bincount(codes, weights=sigma, minlength=len(pop))
            delta = observed - pop
            j = int(np.argmax(np.abs(delta)))
            rows.append({
                "partition": name,
                "linf": float(abs(delta[j])),
                "argmax_cell": str(labels[j]),
                "signed_error": float(delta[j]),
                "tv": float(0.5 * np.abs(delta).sum()),
            })
        worst = max(rows, key=lambda r: r["linf"])
        out: Dict[str, object] = {
            "structural_linf": worst["linf"],
            "structural_argmax_partition": worst["partition"],
            "structural_argmax_cell": worst["argmax_cell"],
            "structural_argmax_signed_error": worst["signed_error"],
        }
        out.update({f"structural_tv_{r['partition']}": r["tv"] for r in rows})
        return out

    def metric_bundle(self, entries: List[Dict]) -> Dict[str, object]:
        sigma = self.sigma(entries)
        if sigma.sum() <= 0:
            return {"adoption_gap": float("nan"), "breadth": float("nan"),
                    "structural_linf": float("nan")}
        values: Dict[str, object] = {
            "posterior_effective_variants": float(1.0 / np.sum(sigma ** 2)),
            "adoption_gap": self.w1(self.PI_A, sigma),
            "breadth": self.breadth(entries, self.PI_B),
        }
        values.update(self.hill_spectrum(entries, self.PI_A))
        values.update(self.structural(sigma))
        return values

    # ---- panel helpers ------------------------------------------------------
    def exact_entries(self, indices: Sequence[int]) -> List[Dict]:
        return [
            {"candidate_idx": np.array([int(i)]), "posterior": np.array([1.0]),
             "status": "exact", "match_type": "exact",
             "input_model": str(self.variant.iloc[int(i)]["variant_id"])}
            for i in indices
        ]

    # ---- greedy references --------------------------------------------------
    def greedy_adoption(self, n: int, target: Optional[np.ndarray] = None):
        """Greedy minimisation of the adoption Tree-W1 gap (port of greedy_w1)."""
        target = self.PI_A if target is None else target
        if n <= 0:
            return [], float("nan")
        target_edge = self.branch_mass(target)
        selected: List[int] = []
        available = np.ones(self.n, dtype=bool)
        counts = np.zeros(self.n_edges)
        final_weight = 1.0 / n
        for _ in range(n):
            change = np.zeros(self.n)
            for depth in range(N_HIER):
                edge = self.paths[:, depth]
                before = np.abs(target_edge[edge] - counts[edge])
                after = np.abs(target_edge[edge] - counts[edge] - final_weight)
                change += self.lengths[edge] * (after - before)
            change[~available] = np.inf
            chosen = int(np.argmin(change))
            selected.append(chosen)
            available[chosen] = False
            counts[self.paths[chosen]] += final_weight
        sigma = np.bincount(selected, minlength=self.n) / n
        return selected, self.w1(target, sigma)

    def kernel(self) -> np.ndarray:
        """Full similarity kernel exp(-d/tau) on the primary geometry (float32).

        Built once and cached; ~260MB for the 8k population. Used by the
        vectorised breadth greedy and any all-pairs coverage computation.
        """
        if getattr(self, "_kernel", None) is None:
            K = np.empty((self.n, self.n), dtype=np.float32)
            for i in range(self.n):
                K[i] = np.exp(-self.distances_to(i) / self.tau).astype(np.float32)
            self._kernel = K
        return self._kernel

    def greedy_breadth(self, n: int, target: Optional[np.ndarray] = None,
                       support_subset: Optional[np.ndarray] = None):
        """Greedy maximisation of the submodular facility-location breadth.

        Vectorised CELF-style lazy greedy: at each step add the variant with the
        largest marginal gain in covered (pi_b-weighted) similarity. Monotone
        submodularity gives the standard (1-1/e) guarantee; column j of the
        kernel is sim(.,j).
        """
        if n <= 0:
            return [], float("nan")
        target = self.PI_B if target is None else target
        rows = np.arange(self.n) if support_subset is None else np.asarray(support_subset, int)
        weight = (target[rows] / target[rows].sum()).astype(np.float32)
        K = self.kernel()
        Ksub = K[rows]                       # (|rows|, n): coverage each pop point gets from candidate j
        best = np.zeros(len(rows), dtype=np.float32)
        selected: List[int] = []
        avail = np.ones(self.n, dtype=bool)
        if support_subset is not None:
            mask = np.zeros(self.n, dtype=bool)
            mask[rows] = True
            avail &= mask
        for _ in range(n):
            gains = (np.maximum(Ksub, best[:, None]) - best[:, None]) * weight[:, None]
            gains = gains.sum(axis=0)
            gains[~avail] = -np.inf
            j = int(np.argmax(gains))
            selected.append(j)
            avail[j] = False
            best = np.maximum(best, Ksub[:, j])
        return selected, float(np.dot(weight.astype(float), best.astype(float)))
