"""Shared primitives for the DMKD deliverables.

Everything here works on the foundation cache and the outcome tables, so analyses
never re-execute the validation source. Run any analysis with DMKD_RELEASE set:

    DMKD_RELEASE=june2026 .venv/bin/python analysis/p1_1_validation.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import release_config  # noqa: E402

RELEASE = release_config.current_release()
for key, value in release_config.release_env(RELEASE).items():
    os.environ.setdefault(key, value)
for thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(thread_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from coverage_geometry import Foundation, load_foundation  # noqa: E402

OUTPUT = release_config.output_dir(RELEASE)
TABLES = OUTPUT / "tables"
SEED = 20261105
CAPABILITY_OUTCOMES = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
SAFETY_OUTCOMES = ["salad_base_safety", "salad_attack_safety"]
OUTCOME_LABELS = {
    "IFEval": "IFEval", "BBH": "BBH", "MATH Lvl 5": "MATH Level 5", "GPQA": "GPQA",
    "MUSR": "MUSR", "MMLU-PRO": "MMLU-PRO", "salad_base_safety": "SALAD base safety",
    "salad_attack_safety": "SALAD attack safety",
}
STRATEGIES = ["uniform", "popularity", "family_stratified", "clustered"]
CONTROLS = ["log_n", "mean_log_popularity", "family_entropy"]
PROFILE = ["adoption_gap", "breadth", "structural_linf", "coverage_q0", "coverage_q1", "coverage_delta_1_0"]
PREDICTOR_SETS = {name: [name] for name in PROFILE}
PREDICTOR_SETS["full_profile"] = PROFILE
PREDICTOR_LABELS = {
    "adoption_gap": "Adoption gap", "breadth": "Lineage breadth", "structural_linf": "Structural imbalance",
    "coverage_q0": "C(0)", "coverage_q1": "C(1)", "coverage_delta_1_0": "C(1) - C(0)", "full_profile": "Full profile",
}


def deliverable(task: str) -> Path:
    return release_config.deliverables_dir(task, RELEASE)


def foundation() -> Foundation:
    return load_foundation(OUTPUT / "cache" / "foundation.pkl")


def load_outcomes() -> Dict[str, Tuple[np.ndarray, pd.DataFrame]]:
    """Per group: variant indices and the complete-case outcomes as fractions, as in the source."""
    capability = pd.read_csv(TABLES / "open_llm_leaderboard_by_variant.csv").dropna(subset=CAPABILITY_OUTCOMES)
    safety = pd.read_csv(TABLES / "salad_safety_by_variant.csv").dropna(subset=SAFETY_OUTCOMES)
    return {
        "capability": (capability["variant_idx"].to_numpy(int), capability[CAPABILITY_OUTCOMES].reset_index(drop=True) / 100.0),
        "safety": (safety["variant_idx"].to_numpy(int), safety[SAFETY_OUTCOMES].reset_index(drop=True) / 100.0),
    }


class ConditionedPopulation:
    """Targets and fast panel coordinates on a support S, with optional family weights.

    `multiplicity` holds how many times each support element's family was drawn in a
    cluster bootstrap. Elements with zero multiplicity leave the population, and the
    tree geometry stays fixed.
    """

    def __init__(self, F: Foundation, support: np.ndarray, kernel: Optional[np.ndarray] = None,
                 pi_a: Optional[np.ndarray] = None, pi_b: Optional[np.ndarray] = None,
                 multiplicity: Optional[np.ndarray] = None) -> None:
        self.F = F
        full_support = np.asarray(support, int)
        multiplicity = np.ones(len(full_support)) if multiplicity is None else np.asarray(multiplicity, float)
        keep = multiplicity > 0
        self.support = full_support[keep]
        self.keep = keep
        self.multiplicity = multiplicity[keep]
        pi_a = F.PI_A if pi_a is None else pi_a
        pi_b = F.PI_B if pi_b is None else pi_b

        self.target_a = np.zeros(F.n)
        self.target_b = np.zeros(F.n)
        self.target_a[self.support] = pi_a[self.support] * self.multiplicity
        self.target_b[self.support] = pi_b[self.support] * self.multiplicity
        self.target_a /= self.target_a.sum()
        self.target_b /= self.target_b.sum()
        self.target_edge = F.branch_mass(self.target_a)
        self.base_sum = float(np.dot(F.lengths, self.target_edge))

        self.local_of = np.full(F.n, -1, dtype=int)
        self.local_of[self.support] = np.arange(len(self.support))
        self.kernel = kernel_on(F, full_support)[np.ix_(keep, keep)] if kernel is None else kernel[np.ix_(keep, keep)]
        self.breadth_weight = self.target_b[self.support]

        self.partitions = []
        for name, codes in F.discrepancy["codes"].items():
            uniques, local_codes = np.unique(codes[self.support], return_inverse=True)
            pop = np.bincount(local_codes, weights=self.breadth_weight, minlength=len(uniques))
            self.partitions.append((local_codes, pop))

        self.families = F.variant["family_id"].to_numpy()[self.support]
        self.family_codes, self.family_names = pd.factorize(self.families, sort=True)
        self.log_downloads = np.log1p(F.DOWNLOADS)
        self.sampling_prob = self.target_a[self.support] / self.target_a[self.support].sum()
        self.uniform_prob = self.multiplicity / self.multiplicity.sum()
        self.family_members = [np.where(self.family_codes == k)[0] for k in range(len(self.family_names))]
        self.family_prob = np.array([self.sampling_prob[m].sum() for m in self.family_members])

    def draw(self, n: int, strategy: str, rng: np.random.Generator, subset: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw a panel of support positions. Same logic as `random_indices` in the source."""
        pool = np.arange(len(self.support)) if subset is None else np.asarray(subset, int)
        prob = self.sampling_prob[pool]
        prob = prob / prob.sum()
        if strategy == "uniform":
            weights = self.uniform_prob[pool]
            weights = weights / weights.sum()
            if np.allclose(weights, weights[0]):
                return rng.choice(pool, n, replace=len(pool) < n)
            return rng.choice(pool, n, replace=len(pool) < n, p=weights)
        if strategy == "popularity":
            return rng.choice(pool, n, replace=len(pool) < n, p=prob)
        codes = self.family_codes[pool]
        available = np.unique(codes)
        family_mass = np.array([prob[codes == k].sum() for k in available])
        family_mass = family_mass / family_mass.sum()
        if strategy == "clustered":
            family = rng.choice(available, p=family_mass)
            local = pool[codes == family]
            local_p = self.sampling_prob[local] / self.sampling_prob[local].sum()
            return rng.choice(local, n, replace=len(local) < n, p=local_p)
        if strategy == "family_stratified":
            chosen = rng.choice(available, min(n, len(available)), replace=False, p=family_mass)
            selected: List[int] = []
            for family in chosen:
                local = pool[codes == family]
                local_p = self.sampling_prob[local] / self.sampling_prob[local].sum()
                selected.append(int(rng.choice(local, p=local_p)))
            while len(selected) < n:
                candidate = int(rng.choice(pool, p=prob))
                if candidate not in selected or len(pool) < n:
                    selected.append(candidate)
            return np.asarray(selected, int)
        raise ValueError(strategy)

    def coordinates(self, positions: np.ndarray) -> np.ndarray:
        """Coordinates of an equal-mass exact panel, in the order of `PROFILE + CONTROLS`."""
        positions = np.asarray(positions, int)
        n = len(positions)
        indices = self.support[positions]
        F = self.F
        edges, counts = np.unique(F.paths[indices].ravel(), return_counts=True)
        target_here = self.target_edge[edges]
        gap = self.base_sum + float(np.dot(F.lengths[edges], np.abs(target_here - counts / n) - target_here))
        breadth = float(np.dot(self.breadth_weight, self.kernel[:, positions].max(axis=1)))
        linf = 0.0
        for local_codes, pop in self.partitions:
            observed = np.bincount(local_codes[positions], minlength=len(pop)) / n
            linf = max(linf, float(np.abs(observed - pop).max()))
        unique = np.unique(positions)
        c0 = len(unique) / len(self.support)
        c1 = float(self.target_a[self.support[unique]].sum())
        share = np.bincount(self.family_codes[positions]).astype(float)
        share = share[share > 0] / n
        entropy = float(-(share * np.log(share)).sum())
        return np.array([gap, breadth, linf, c0, c1, c1 - c0, math.log(n),
                         float(self.log_downloads[indices].mean()), entropy])

    def targets_for(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Local outcome matrix, population means, and population standard deviations."""
        local = values[self.keep]
        weights = self.target_a[self.support]
        mean = weights @ local
        sd = np.maximum(np.sqrt(weights @ (local - mean) ** 2), 1e-8)
        return local, mean, sd


def kernel_on(F: Foundation, support: np.ndarray) -> np.ndarray:
    """Similarity exp(-d/tau) restricted to the support, in support order."""
    support = np.asarray(support, int)
    out = np.empty((len(support), len(support)), dtype=np.float64)
    for j, idx in enumerate(support):
        out[:, j] = np.exp(-F.distances_to(int(idx))[support] / F.tau)
    return out


def distances_on(F: Foundation, support: np.ndarray) -> np.ndarray:
    support = np.asarray(support, int)
    out = np.empty((len(support), len(support)), dtype=np.float64)
    for j, idx in enumerate(support):
        out[:, j] = F.distances_to(int(idx))[support]
    return out


def simulation_sizes(population_size: int) -> List[int]:
    """Panel sizes used by the validation design, as in the source."""
    sizes = sorted({min(v, max(2, population_size // 3)) for v in [3, 5, 10, 20]})
    return [v for v in sizes if 2 <= v < population_size]


COORDINATE_NAMES = PROFILE + CONTROLS


def simulate_panels(pop: ConditionedPopulation, values: np.ndarray, sizes: Sequence[int], per_cell: int,
                    rng_for, subset: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Simulate the size-by-strategy design. `rng_for(n, s)` gives the generator of a cell."""
    local, mean, sd = pop.targets_for(values)
    coordinate_rows, error_rows, size_rows, strategy_rows = [], [], [], []
    for n in sizes:
        for strategy_index, strategy in enumerate(STRATEGIES):
            rng = rng_for(n, strategy_index)
            for _ in range(per_cell):
                positions = pop.draw(n, strategy, rng, subset=subset)
                coordinate_rows.append(pop.coordinates(positions))
                error_rows.append(np.abs(local[positions].mean(axis=0) - mean) / sd)
                size_rows.append(n)
                strategy_rows.append(strategy_index)
    return {
        "X": np.asarray(coordinate_rows, float),
        "E": np.asarray(error_rows, float),
        "n": np.asarray(size_rows, int),
        "strategy": np.asarray(strategy_rows, int),
        "sd": sd,
    }


def ridge_predict(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """StandardScaler plus Ridge(alpha), matching the sklearn pipeline, for several outcomes at once."""
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale == 0] = 1.0
    A = (X_train - mean) / scale
    B = (X_test - mean) / scale
    y_mean = Y_train.mean(axis=0)
    beta = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (Y_train - y_mean))
    return B @ beta + y_mean


def r2_by_column(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    ss_res = ((y - pred) ** 2).sum(axis=0)
    ss_tot = np.maximum(((y - y.mean(axis=0)) ** 2).sum(axis=0), 1e-30)
    return 1.0 - ss_res / ss_tot


def column_positions(names: Sequence[str]) -> List[int]:
    return [COORDINATE_NAMES.index(name) for name in names]


def cross_validate(X: np.ndarray, E: np.ndarray, groups: np.ndarray,
                   predictor_sets: Dict[str, List[str]] = PREDICTOR_SETS) -> Dict[str, Dict[str, np.ndarray]]:
    """Leave-one-group-out comparison of the control model and each augmented model."""
    base_cols = column_positions(CONTROLS)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    base_pred = np.empty_like(E)
    labels = np.unique(groups)
    for label in labels:
        test = groups == label
        base_pred[test] = ridge_predict(X[~test][:, base_cols], E[~test], X[test][:, base_cols])
    for name, members in predictor_sets.items():
        cols = base_cols + column_positions(members)
        pred = np.empty_like(E)
        for label in labels:
            test = groups == label
            pred[test] = ridge_predict(X[~test][:, cols], E[~test], X[test][:, cols])
        out[name] = summarize_fit(E, base_pred, pred)
    return out


def summarize_fit(y: np.ndarray, base_pred: np.ndarray, pred: np.ndarray) -> Dict[str, np.ndarray]:
    r2_base = r2_by_column(y, base_pred)
    r2_aug = r2_by_column(y, pred)
    mae_base = np.abs(y - base_pred).mean(axis=0)
    mae_aug = np.abs(y - pred).mean(axis=0)
    return {"r2_base": r2_base, "r2_aug": r2_aug, "delta_r2": r2_aug - r2_base,
            "mae_base": mae_base, "mae_aug": mae_aug, "delta_mae": mae_base - mae_aug}


def family_split(pop: ConditionedPopulation, rng: np.random.Generator) -> np.ndarray:
    """Assign each family to one of two sides, balancing variant counts. True means side one."""
    order = rng.permutation(len(pop.family_names))
    side_one = np.zeros(len(pop.family_names), dtype=bool)
    totals = [0, 0]
    for family in order:
        side = int(totals[1] < totals[0])
        side_one[family] = side == 1
        totals[side] += len(pop.family_members[family])
    return side_one


def family_disjoint_cv(pop: ConditionedPopulation, values: np.ndarray, sizes: Sequence[int], per_cell: int,
                       splits: int, seed: int,
                       predictor_sets: Dict[str, List[str]] = PREDICTOR_SETS) -> Dict[str, Dict[str, np.ndarray]]:
    """Train and test on panels drawn from disjoint family sets, so no model is shared.

    Families are split in two, with each side training the model that scores the
    other, and this repeats over random splits. Coordinates and errors stay relative
    to the whole population, because the goal remains recovering its mean.
    """
    y_all, base_all, aug_all = [], [], {name: [] for name in predictor_sets}
    base_cols = column_positions(CONTROLS)
    rng_split = np.random.default_rng(seed)
    for split in range(splits):
        side_one = family_split(pop, rng_split)
        sides = [np.where(~side_one[pop.family_codes])[0], np.where(side_one[pop.family_codes])[0]]
        if min(len(s) for s in sides) == 0:
            continue
        panels = []
        for side_index, subset in enumerate(sides):
            def rng_for(n, s, _side=side_index, _split=split):
                return np.random.default_rng([seed, _split, _side, n, s])
            panels.append(simulate_panels(pop, values, sizes, per_cell, rng_for, subset=subset))
        for train, test in [(panels[0], panels[1]), (panels[1], panels[0])]:
            y_all.append(test["E"])
            base_all.append(ridge_predict(train["X"][:, base_cols], train["E"], test["X"][:, base_cols]))
            for name, members in predictor_sets.items():
                cols = base_cols + column_positions(members)
                aug_all[name].append(ridge_predict(train["X"][:, cols], train["E"], test["X"][:, cols]))
    y = np.vstack(y_all)
    base = np.vstack(base_all)
    return {name: summarize_fit(y, base, np.vstack(preds)) for name, preds in aug_all.items()}


def family_multiplicity(pop: ConditionedPopulation, rng: np.random.Generator) -> np.ndarray:
    """Draw families with replacement and return each support element's multiplicity."""
    draws = rng.integers(0, len(pop.family_names), len(pop.family_names))
    counts = np.bincount(draws, minlength=len(pop.family_names))
    return counts[pop.family_codes].astype(float)


def rank_of(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy()


def save_figure(fig, folder: Path, name: str) -> None:
    fig.savefig(folder / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    """Render a table as GitHub Markdown without an extra dependency."""
    def cell(value) -> str:
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return ""
            if value != 0 and abs(value) < 10 ** (-digits):
                return f"{value:.1e}"
            text = f"{value:.{digits}f}"
            return text.rstrip("0").rstrip(".") if "." in text else text
        return str(value)
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])
