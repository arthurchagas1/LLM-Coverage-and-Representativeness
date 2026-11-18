# %% [markdown]
# # Validation of the LLM-ecosystem representation framework
#
# (Notebook-style script, executed by src/build_foundation.py in the `validation` step. Some comments below are in
# Portuguese.)
#
# Pipeline: target -> geometry -> representation coordinate -> target-specific interpretation.
#
# An ambiguous model name is not expanded to all its descendants: each paper entry induces a posterior
# `p(variant | name)` and keeps total mass one. Papers are resolved against the population parquet of the release.
#
# The main coordinates are not aggregated into one score:
#
# * `G_a`: Tree-Wasserstein gap to the adoption target, lower is better;
# * `B_b`: facility-location breadth under the uniform-branching target, higher is better;
# * `D_b`: structural discrepancy against the breadth target, lower is better;
# * `C(q)`: observable coverage spectrum with probabilistic credit for ambiguous names;
# * `Eff_g`: unclipped comparison with a same-size greedy reference.

# %%
from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import sys
import warnings
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linprog
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid", context="notebook")

_LINEAGE_RELATIVE = "data/oome_lineages_20261105/oome_models_with_lineage.parquet"


def _work_root() -> Path:
    """Locate the coverage project root from the working directory."""
    override = os.environ.get("COVERAGE_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "coverage_geometry.py").exists():
            return candidate
    raise RuntimeError(
        "Could not locate the coverage project root. "
        "Set COVERAGE_PROJECT_ROOT to the directory holding src/."
    )


def _data_root(work_root: Path) -> Path:
    """Locate the checkout holding the population parquet and the resolver."""
    override = os.environ.get("COVERAGE_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    candidates = [
        work_root,
    ]
    for candidate in candidates:
        if (candidate / _LINEAGE_RELATIVE).exists():
            return candidate.resolve()
    raise RuntimeError(
        "Could not locate the data root holding "
        f"{_LINEAGE_RELATIVE}. Set COVERAGE_DATA_ROOT."
    )


WORK_ROOT = _work_root()
DATA_ROOT = _data_root(WORK_ROOT)
LINEAGE_FILE = Path(os.environ.get("COVERAGE_LINEAGE_FILE") or DATA_ROOT / _LINEAGE_RELATIVE)
ANNOTATION_FILE = DATA_ROOT / "data/paper_annotations/acl15_new_schema.json"
OUT_DIR = Path(os.environ.get("COVERAGE_OUTPUT_DIR") or WORK_ROOT / "outputs_corrected")
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
EXTERNAL_DIR = WORK_ROOT / "external_data"
for path in (OUT_DIR, TABLE_DIR, FIG_DIR, EXTERNAL_DIR):
    path.mkdir(parents=True, exist_ok=True)

SEED = 20260612
RNG = np.random.default_rng(SEED)
ALPHA = 0.10
BETA = 0.15
LAMBDA0 = 0.25
Q_GRID = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
FAST_MODE = os.environ.get("COVERAGE_FAST_MODE", "0") == "1"
N_NULL = 30 if FAST_MODE else 120
N_SIM_PER_CELL = 12 if FAST_MODE else 50
N_CV_BOOT = 30 if FAST_MODE else 150

print("Population:", LINEAGE_FILE)
print("Annotations:", ANNOTATION_FILE)
print("Corrected outputs:", OUT_DIR)
print("FAST_MODE:", FAST_MODE)

# %% [markdown]
# ## 1. Populacao, unidade de analise e dois targets
#
# A unidade terminal e `variant_id`. O target de adocao mistura downloads e um
# piso uniforme:
#
# \[
# \pi_a(i)=(1-\alpha)\frac{d_i}{\sum_jd_j}+\frac{\alpha}{N}.
# \]
#
# O target de breadth e uniforme por ramificacao: familias recebem massas
# iguais, depois series dentro de cada familia, bases dentro de cada serie e
# variantes dentro de cada base. Assim, breadth nao reutiliza a concentracao de
# downloads que o adoption gap deve auditar.

# %%
LINEAGE_COLUMNS = [
    "repo_id", "variant_id", "base_model_id", "series_id", "family_id",
    "size_bucket", "model_role", "training_stage", "downloads_30d", "likes",
    "created_at", "is_likely_official_model", "is_multimodal_leak",
]
raw = pd.read_parquet(LINEAGE_FILE, columns=LINEAGE_COLUMNS)
raw = raw[~raw["is_multimodal_leak"].fillna(False)].copy()
for col in ["repo_id", "variant_id", "base_model_id", "series_id", "family_id",
            "size_bucket", "model_role", "training_stage"]:
    raw[col] = raw[col].fillna("MISSING").astype(str).replace("", "MISSING")
raw["downloads_30d"] = pd.to_numeric(raw["downloads_30d"], errors="coerce").fillna(0.0)
raw["likes"] = pd.to_numeric(raw["likes"], errors="coerce").fillna(0.0)
raw["official"] = raw["is_likely_official_model"].fillna(False).astype(bool)


def mode_or_first(values: pd.Series, default: str = "MISSING") -> str:
    values = values.fillna("").astype(str)
    values = values[~values.isin({"", "nan", "None"})]
    return str(values.value_counts().index[0]) if len(values) else default


variant = (
    raw.groupby("variant_id", sort=False)
    .agg(
        base_model_id=("base_model_id", mode_or_first),
        series_id=("series_id", mode_or_first),
        family_id=("family_id", mode_or_first),
        size_bucket=("size_bucket", mode_or_first),
        model_role=("model_role", mode_or_first),
        training_stage=("training_stage", mode_or_first),
        downloads_30d=("downloads_30d", "sum"),
        likes=("likes", "sum"),
        repo_count=("repo_id", "size"),
    )
    .reset_index()
)
for col in ["variant_id", "base_model_id", "series_id", "family_id",
            "size_bucket", "model_role", "training_stage"]:
    variant[col] = variant[col].fillna("MISSING").astype(str).replace("", "MISSING")
variant["variant_idx"] = np.arange(len(variant), dtype=int)
variant_index = pd.Series(variant["variant_idx"].to_numpy(), index=variant["variant_id"]).to_dict()

DOWNLOADS = variant["downloads_30d"].to_numpy(float)
LIKES = variant["likes"].to_numpy(float)
UNIFORM = np.full(len(variant), 1.0 / len(variant))
DOWNLOAD_PROB = DOWNLOADS / max(DOWNLOADS.sum(), 1.0)
PI_A = (1.0 - ALPHA) * DOWNLOAD_PROB + ALPHA * UNIFORM


def uniform_branching_target(table: pd.DataFrame) -> np.ndarray:
    keys = ["family_id", "series_id", "base_model_id", "variant_id"]
    frame = table[keys].copy()
    n_family = frame["family_id"].nunique()
    n_series = frame.groupby("family_id")["series_id"].transform("nunique")
    n_base = frame.groupby(["family_id", "series_id"])["base_model_id"].transform("nunique")
    n_variant = frame.groupby(
        ["family_id", "series_id", "base_model_id"]
    )["variant_id"].transform("nunique")
    mass = 1.0 / n_family / n_series / n_base / n_variant
    return mass.to_numpy(float) / mass.sum()


PI_B = uniform_branching_target(variant)

population_summary = pd.Series({
    "repositories": len(raw),
    "variants": len(variant),
    "families": variant["family_id"].nunique(),
    "series": variant[["family_id", "series_id"]].drop_duplicates().shape[0],
    "bases": variant[["family_id", "series_id", "base_model_id"]].drop_duplicates().shape[0],
    "pi_a_sum": PI_A.sum(),
    "pi_b_sum": PI_B.sum(),
    "pi_a_effective_n": 1.0 / np.sum(PI_A ** 2),
    "pi_b_effective_n": 1.0 / np.sum(PI_B ** 2),
})
display(population_summary.to_frame("value"))
population_summary.to_csv(TABLE_DIR / "population_summary.csv", header=["value"])

# %% [markdown]
# ## 2. Geometria estrutural de linhagem
#
# Para cada aresta:
#
# \[
# \lambda_e\propto(1-\beta)\left[-\log
# \frac{c(L_e)}{c(L_{p(e)})}\right]+\beta\lambda_0.
# \]
#
# `c` e estrutural, nao downloads. A implementacao abaixo usa caminhos
# qualificados pelo pai, evitando colapsar duas series homonimas em familias
# diferentes. Todas as arestas recebem comprimento estritamente positivo, e a
# escala e normalizada para profundidade esperada um sob `pi_a`.

# %%
HIERARCHY = ["family_id", "series_id", "base_model_id", "variant_id"]


@dataclass
class TreeGeometry:
    table: pd.DataFrame
    normalization_pi: np.ndarray
    count_mode: str = "variant_count"
    beta: float = BETA
    lambda0: float = LAMBDA0

    def __post_init__(self):
        self.n = len(self.table)
        self.paths = np.empty((self.n, len(HIERARCHY)), dtype=int)
        self.edge_labels: List[str] = []
        self.edge_depth: List[int] = []
        offset = 0
        prefixes: List[np.ndarray] = []
        for depth in range(len(HIERARCHY)):
            labels = self.table[HIERARCHY[:depth + 1]].astype(str).agg(" / ".join, axis=1)
            codes, uniques = pd.factorize(labels, sort=True)
            self.paths[:, depth] = offset + codes
            self.edge_labels.extend(uniques.tolist())
            self.edge_depth.extend([depth] * len(uniques))
            prefixes.append(codes)
            offset += len(uniques)
        self.n_edges = offset
        self.edge_depth = np.asarray(self.edge_depth, dtype=int)

        if self.count_mode == "repo_count":
            leaf_count = self.table["repo_count"].to_numpy(float)
        elif self.count_mode == "adoption_weighted":
            leaf_count = np.asarray(self.normalization_pi, float)
        elif self.count_mode == "topological":
            leaf_count = np.ones(self.n)
        else:
            leaf_count = np.ones(self.n)

        edge_count = self.branch_mass(leaf_count)
        raw_length = np.zeros(self.n_edges)
        root_total = max(float(leaf_count.sum()), 1e-15)
        for depth in range(len(HIERARCHY)):
            edges = self.paths[:, depth]
            child = edge_count[edges]
            if depth == 0:
                parent = np.full(self.n, root_total)
            else:
                parent = edge_count[self.paths[:, depth - 1]]
            ratio = np.clip(child / np.maximum(parent, 1e-15), 1e-15, 1.0)
            if self.count_mode == "topological":
                per_leaf = np.ones(self.n)
            else:
                per_leaf = (1.0 - self.beta) * (-np.log(ratio)) + self.beta * self.lambda0
            raw_length[edges] = per_leaf

        raw_depth = raw_length[self.paths].sum(axis=1)
        scale = float(np.dot(self.normalization_pi, raw_depth))
        self.lengths = raw_length / max(scale, 1e-15)
        self.path_lengths = self.lengths[self.paths]
        self.cumulative = np.cumsum(self.path_lengths, axis=1)
        self.root_distance = self.cumulative[:, -1]
        self._distance_cache: Dict[int, np.ndarray] = {}
        self.tau = self.estimate_tau()

    def branch_mass(self, leaf_mass: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_edges)
        leaf_mass = np.asarray(leaf_mass, float)
        for depth in range(len(HIERARCHY)):
            np.add.at(out, self.paths[:, depth], leaf_mass)
        return out

    def w1(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(self.lengths, np.abs(self.branch_mass(a) - self.branch_mass(b))))

    def distances_to(self, selected_idx: int, cache: bool = True) -> np.ndarray:
        selected_idx = int(selected_idx)
        if cache and selected_idx in self._distance_cache:
            return self._distance_cache[selected_idx]
        lca = np.zeros(self.n)
        same = np.ones(self.n, dtype=bool)
        for depth in range(len(HIERARCHY)):
            same &= self.paths[:, depth] == self.paths[selected_idx, depth]
            lca[same] = self.cumulative[same, depth]
        dist = np.maximum(
            self.root_distance + self.root_distance[selected_idx] - 2.0 * lca, 0.0
        )
        if cache:
            self._distance_cache[selected_idx] = dist
        return dist

    def pair_distance(self, i: int, j: int) -> float:
        return float(self.distances_to(j)[i])

    def estimate_tau(self, n_anchors: int = 1200) -> float:
        rng = np.random.default_rng(SEED + 19)
        anchors = rng.choice(self.n, min(self.n, n_anchors), replace=False)
        nearest = []
        for idx in anchors:
            distances = self.distances_to(int(idx), cache=False)
            positive = distances[distances > 1e-12]
            if len(positive):
                nearest.append(float(positive.min()))
        return float(np.median(nearest)) if nearest else 1.0

    def greedy_w1(self, n: int, target: np.ndarray = PI_A) -> Tuple[List[int], float]:
        if n <= 0:
            return [], np.nan
        target_edge = self.branch_mass(target)
        selected: List[int] = []
        available = np.ones(self.n, dtype=bool)
        counts = np.zeros(self.n_edges)
        final_weight = 1.0 / n
        for _ in range(n):
            change = np.zeros(self.n)
            for depth in range(len(HIERARCHY)):
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


tree = TreeGeometry(variant, PI_A, count_mode="variant_count")
print("Edges:", tree.n_edges)
print("Minimum edge length:", tree.lengths.min())
print("E_pi_a depth:", np.dot(PI_A, tree.root_distance))
print("Kernel tau:", tree.tau)

# %% [markdown]
# ## 3. Resolver probabilistico sem expansao
#
# O resolver preserva conjuntos de candidatos. Um basename duplicado nunca e
# convertido no primeiro repo encontrado. Matches exatos geram posterior
# degenerada; nomes de base, serie, familia ou aliases ambiguos geram posterior
# proporcional a `pi_a` restrita aos candidatos. O log registra entropia,
# probabilidade maxima e alternativas principais.

# %%
sys.path.insert(0, str(DATA_ROOT))
from model_resolver import (  # noqa: E402
    MANUAL_ALIAS_MAP_RAW,
    extract_roles,
    extract_sizes,
    extract_versions,
    find_family,
    generate_aliases_for_repo,
    informative_tokens,
    normalize_key,
    normalize_text,
    provider_of,
)


def add_to_set_map(mapping: Dict[str, Set[int]], key: str, values: Iterable[int]) -> None:
    key = normalize_key(key)
    if key:
        mapping.setdefault(key, set()).update(int(v) for v in values)


@dataclass
class SoftResolution:
    input_model: str
    status: str
    match_type: str
    confidence: float
    candidate_idx: np.ndarray
    posterior: np.ndarray
    reason: str

    @property
    def resolved(self) -> bool:
        return len(self.candidate_idx) > 0

    def audit_row(self) -> Dict[str, object]:
        if self.resolved:
            order = np.argsort(-self.posterior)[:8]
            alternatives = "; ".join(
                f"{variant.iloc[self.candidate_idx[i]]['variant_id']} ({self.posterior[i]:.3f})"
                for i in order
            )
            entropy = float(-np.sum(self.posterior * np.log(np.maximum(self.posterior, 1e-15))))
            max_p = float(self.posterior.max())
        else:
            alternatives, entropy, max_p = "", np.nan, np.nan
        return {
            "input_model": self.input_model,
            "status": self.status,
            "match_type": self.match_type,
            "confidence": self.confidence,
            "n_candidate_variants": len(self.candidate_idx),
            "posterior_entropy": entropy,
            "posterior_max": max_p,
            "top_candidates": alternatives,
            "reason": self.reason,
        }


class ProbabilisticVariantResolver:
    def __init__(self, repo_table: pd.DataFrame, variant_table: pd.DataFrame, prior: np.ndarray):
        self.variant_table = variant_table
        self.prior = np.asarray(prior, float)
        self.repo_exact: Dict[str, Set[int]] = {}
        self.basename_map: Dict[str, Set[int]] = {}
        self.alias_map: Dict[str, Set[int]] = {}
        self.taxonomy_map: Dict[str, Tuple[str, Set[int]]] = {}
        self.token_to_variants: Dict[str, Set[int]] = {}
        self.representative: Dict[int, str] = {}
        self.manual = {normalize_key(k): normalize_text(v) for k, v in MANUAL_ALIAS_MAP_RAW.items()}

        repo_small = repo_table[
            ["repo_id", "variant_id", "downloads_30d", "official"]
        ].drop_duplicates("repo_id").copy()
        repo_small["variant_idx"] = repo_small["variant_id"].map(variant_index)
        repo_small = repo_small.dropna(subset=["variant_idx"])
        repo_small["variant_idx"] = repo_small["variant_idx"].astype(int)
        for row in repo_small.itertuples(index=False):
            idx = int(row.variant_idx)
            add_to_set_map(self.repo_exact, row.repo_id, [idx])
            add_to_set_map(self.basename_map, normalize_key(row.repo_id).split("/")[-1], [idx])

        # Generated aliases are built from one canonical representative per variant,
        # which keeps the index small while retaining suffix/provider normalization.
        representatives = (
            repo_small.sort_values(["official", "downloads_30d"], ascending=False)
            .drop_duplicates("variant_idx")
        )
        for row in representatives.itertuples(index=False):
            idx = int(row.variant_idx)
            self.representative[idx] = str(row.repo_id)
            extra = self.variant_table.iloc[idx][
                ["variant_id", "base_model_id", "series_id", "family_id"]
            ].astype(str).tolist()
            for alias in generate_aliases_for_repo(str(row.repo_id), []):
                add_to_set_map(self.alias_map, alias, [idx])
            for token in informative_tokens(" ".join([str(row.repo_id), *extra])):
                self.token_to_variants.setdefault(token, set()).add(idx)

        for level in ["variant_id", "base_model_id", "series_id", "family_id"]:
            grouped = self.variant_table.groupby(level).indices
            for label, idxs in grouped.items():
                key = normalize_key(label)
                if not key:
                    continue
                old = self.taxonomy_map.get(key)
                values = set(map(int, idxs))
                if old is None or len(values) < len(old[1]):
                    self.taxonomy_map[key] = (level, values)

    def posterior(self, candidates: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.array(sorted(set(map(int, candidates))), dtype=int)
        if len(idx) == 0:
            return idx, np.array([], dtype=float)
        mass = self.prior[idx]
        if mass.sum() <= 0:
            mass = np.ones(len(idx))
        return idx, mass / mass.sum()

    def result(
        self, original: str, status: str, match_type: str, confidence: float,
        candidates: Iterable[int], reason: str,
    ) -> SoftResolution:
        idx, prob = self.posterior(candidates)
        return SoftResolution(original, status, match_type, confidence, idx, prob, reason)

    def _features(self, text: str) -> Dict[str, object]:
        return {
            "key": normalize_key(text),
            "basename": normalize_key(text).split("/")[-1],
            "provider": provider_of(text),
            "tokens": informative_tokens(text),
            "family": find_family(text),
            "sizes": extract_sizes(text),
            "versions": extract_versions(text),
            "roles": extract_roles(text),
        }

    def _score(self, original: str, candidate_idx: int) -> float:
        candidate = self.representative[candidate_idx]
        a, b = self._features(original), self._features(candidate)
        at, bt = a["tokens"], b["tokens"]
        overlap = len(at & bt) / max(1, len(at))
        jaccard = len(at & bt) / max(1, len(at | bt))
        seq = difflib.SequenceMatcher(None, a["basename"], b["basename"]).ratio()
        score = 0.45 * overlap + 0.20 * jaccard + 0.15 * seq
        if a["family"] and b["family"]:
            score += 0.12 if a["family"] == b["family"] else -0.25
        if a["provider"]:
            score += 0.06 if a["provider"] == b["provider"] else -0.04
        if a["sizes"]:
            score += 0.12 if a["sizes"] & b["sizes"] else -0.18
        if a["roles"]:
            score += 0.06 if a["roles"] & b["roles"] else -0.04
        if a["versions"] and b["versions"] and a["versions"] & b["versions"]:
            score += 0.04
        return float(np.clip(score, 0.0, 1.0))

    def resolve(self, model: object, allow_fuzzy: bool = True) -> SoftResolution:
        original = normalize_text(model)
        if not original:
            return self.result("", "empty", "empty", 0.0, [], "empty input")
        key = normalize_key(original)

        manual_target = self.manual.get(key)
        if manual_target:
            target_key = normalize_key(manual_target)
            candidates = self.repo_exact.get(target_key, set())
            if candidates:
                return self.result(
                    original, "matched_manual_alias", "manual_alias", 1.0,
                    candidates, f"manual alias -> {manual_target}",
                )

        candidates = self.repo_exact.get(key, set())
        if candidates:
            return self.result(
                original, "matched_repo_exact", "repo_exact", 1.0,
                candidates, "exact full repository id",
            )

        taxonomy = self.taxonomy_map.get(key)
        if taxonomy:
            level, candidates = taxonomy
            status = "matched_variant_exact" if level == "variant_id" and len(candidates) == 1 else "matched_taxonomy_soft"
            return self.result(
                original, status, f"taxonomy_{level}", 1.0 if len(candidates) == 1 else 0.90,
                candidates, f"exact {level}; retained as posterior over descendants",
            )

        base_key = key.split("/")[-1]
        candidates = self.basename_map.get(base_key, set())
        if candidates:
            status = "matched_basename_unique" if len(candidates) == 1 else "matched_basename_soft"
            return self.result(
                original, status, "repo_basename", 0.98 if len(candidates) == 1 else 0.85,
                candidates, "basename candidates retained; no first-match collapse",
            )

        alias_hits = []
        for alias in generate_aliases_for_repo(original):
            normalized_alias = normalize_key(alias)
            candidates = self.alias_map.get(normalized_alias, set())
            if candidates:
                alias_hits.append((len(candidates), -len(normalized_alias), normalized_alias, candidates))
        if alias_hits:
            _, _, best_alias, aliases = min(alias_hits, key=lambda item: (item[0], item[1]))
            aliases = set(aliases)
            explicit_sizes = extract_sizes(original)
            if explicit_sizes:
                size_compatible = {
                    idx for idx in aliases
                    if explicit_sizes & extract_sizes(
                        self.representative[idx] + " " + str(self.variant_table.iloc[idx]["variant_id"])
                    )
                }
                aliases = size_compatible
            explicit_roles = extract_roles(original)
            role_map = {"it": "instruct", "chat": "chat", "base": "base", "instruct": "instruct", "code": "code", "coder": "code"}
            desired_roles = {role_map[role] for role in explicit_roles if role in role_map}
            if aliases and desired_roles:
                role_compatible = {
                    idx for idx in aliases
                    if str(self.variant_table.iloc[idx]["model_role"]).lower() in desired_roles
                }
                if role_compatible:
                    aliases = role_compatible
            if not aliases and explicit_sizes:
                fallback = set().union(*(item[3] for item in alias_hits))
                fallback = {
                    idx for idx in fallback
                    if explicit_sizes & extract_sizes(
                        self.representative[idx] + " " + str(self.variant_table.iloc[idx]["variant_id"])
                    )
                }
                if fallback and desired_roles:
                    role_compatible = {
                        idx for idx in fallback
                        if str(self.variant_table.iloc[idx]["model_role"]).lower() in desired_roles
                    }
                    if role_compatible:
                        fallback = role_compatible
                aliases = fallback
                best_alias = best_alias + " [compatibility fallback]"
            if aliases:
                status = "matched_generated_alias_unique" if len(aliases) == 1 else "matched_generated_alias_soft"
                return self.result(
                    original, status, "generated_alias", 0.96 if len(aliases) == 1 else 0.82,
                    aliases, f"most specific compatible generated alias: {best_alias}",
                )

        if allow_fuzzy:
            candidate_pool: Dict[int, int] = {}
            for token in informative_tokens(original):
                for idx in self.token_to_variants.get(token, set()):
                    candidate_pool[idx] = candidate_pool.get(idx, 0) + 1
            shortlist = [
                idx for idx, _ in sorted(
                    candidate_pool.items(), key=lambda item: item[1], reverse=True
                )[:400]
            ]
            if shortlist:
                scored = sorted(
                    [(self._score(original, idx), idx) for idx in shortlist], reverse=True
                )
                top_score = scored[0][0]
                second = scored[1][0] if len(scored) > 1 else -1.0
                if top_score >= 0.86 and top_score - second >= 0.06:
                    return self.result(
                        original, "matched_fuzzy_high_confidence", "fuzzy_structural",
                        top_score, [scored[0][1]], f"top margin={top_score-second:.3f}",
                    )
                review = [idx for score, idx in scored if score >= max(0.78, top_score - 0.04)]
                if top_score >= 0.78 and review:
                    result = self.result(
                        original, "matched_fuzzy_soft_review", "fuzzy_structural_soft",
                        top_score, review, f"{len(review)} candidates within review band",
                    )
                    # Fuzzy scores refine the adoption prior without creating mass.
                    scores = np.array([dict((idx, score) for score, idx in scored)[i] for i in result.candidate_idx])
                    weights = result.posterior * np.exp(8.0 * (scores - scores.max()))
                    result.posterior = weights / weights.sum()
                    return result

        return self.result(
            original, "unresolved", "none", 0.0, [],
            "no exact, taxonomy, basename, alias, or accepted fuzzy match",
        )


resolver = ProbabilisticVariantResolver(raw, variant, PI_A)
print("Resolver exact repos:", len(resolver.repo_exact))
print("Resolver basenames:", len(resolver.basename_map))
print("Resolver generated aliases:", len(resolver.alias_map))

# %% [markdown]
# ## 4. Medidas soft: uma entrada continua valendo uma entrada
#
# Para um painel com entradas resolvidas `s=1,...,m`:
#
# \[
# \sigma_P(i)=\frac1m\sum_s p(i\mid s).
# \]
#
# A cobertura observavel usa a probabilidade de inclusao
# `c_i = 1 - product_s(1-p(i|s))`, e nao o indicador de que `i` apareceu em
# qualquer conjunto candidato. Para breadth, cada entrada ambigua fornece a
# similaridade esperada `E_{i~p(.|s)} k(x,i)`, seguida do maximo entre entradas.
# Portanto, cem candidatos ambiguos nao recebem cem creditos integrais.

# %%
@dataclass
class SoftPanel:
    entries: List[SoftResolution]

    @property
    def resolved_entries(self) -> List[SoftResolution]:
        return [entry for entry in self.entries if entry.resolved]

    @property
    def n_input(self) -> int:
        return len(self.entries)

    @property
    def n_resolved(self) -> int:
        return len(self.resolved_entries)

    def sigma(self, n_variants: int = len(variant)) -> np.ndarray:
        out = np.zeros(n_variants)
        if not self.n_resolved:
            return out
        for entry in self.resolved_entries:
            out[entry.candidate_idx] += entry.posterior / self.n_resolved
        return out

    def inclusion_probability(self, n_variants: int = len(variant)) -> np.ndarray:
        not_seen = np.ones(n_variants)
        for entry in self.resolved_entries:
            not_seen[entry.candidate_idx] *= 1.0 - entry.posterior
        return 1.0 - not_seen


class StructuralDiscrepancy:
    def __init__(self, table: pd.DataFrame, target: np.ndarray):
        self.target = np.asarray(target, float)
        self.views = {
            "family": table["family_id"].astype(str),
            "series": table["family_id"].astype(str) + " / " + table["series_id"].astype(str),
            "base": (
                table["family_id"].astype(str) + " / " + table["series_id"].astype(str)
                + " / " + table["base_model_id"].astype(str)
            ),
            "size": table["size_bucket"].astype(str),
            "role": table["model_role"].astype(str),
            "stage": table["training_stage"].astype(str),
            "size_x_role": table["size_bucket"].astype(str) + " || " + table["model_role"].astype(str),
            "role_x_stage": table["model_role"].astype(str) + " || " + table["training_stage"].astype(str),
        }
        self.codes: Dict[str, np.ndarray] = {}
        self.labels: Dict[str, np.ndarray] = {}
        self.pop: Dict[str, np.ndarray] = {}
        for name, values in self.views.items():
            codes, labels = pd.factorize(values, sort=True)
            self.codes[name] = codes
            self.labels[name] = np.asarray(labels, dtype=str)
            self.pop[name] = np.bincount(codes, weights=self.target)

    def compute(self, sigma: np.ndarray) -> Dict[str, object]:
        rows = []
        for name, codes in self.codes.items():
            observed = np.bincount(codes, weights=sigma, minlength=len(self.pop[name]))
            delta = observed - self.pop[name]
            j = int(np.argmax(np.abs(delta)))
            rows.append({
                "partition": name,
                "linf": float(abs(delta[j])),
                "argmax_cell": str(self.labels[name][j]),
                "signed_error": float(delta[j]),
                "tv": float(0.5 * np.abs(delta).sum()),
            })
        worst = max(rows, key=lambda row: row["linf"])
        out: Dict[str, object] = {
            "structural_linf": worst["linf"],
            "structural_argmax_partition": worst["partition"],
            "structural_argmax_cell": worst["argmax_cell"],
            "structural_argmax_signed_error": worst["signed_error"],
        }
        out.update({f"structural_tv_{row['partition']}": row["tv"] for row in rows})
        return out


discrepancy_b = StructuralDiscrepancy(variant, PI_B)


def soft_hill_spectrum(panel: SoftPanel, target: np.ndarray = PI_A) -> Dict[str, float]:
    inclusion = panel.inclusion_probability()
    out: Dict[str, float] = {}
    for q in Q_GRID:
        weights = np.ones(len(target)) if q == 0 else target ** q
        out[f"coverage_q{q:g}"] = float(np.dot(inclusion, weights) / weights.sum())
    out["coverage_delta_1_0"] = out["coverage_q1"] - out["coverage_q0"]
    return out


def soft_breadth(
    panel: SoftPanel,
    geom: TreeGeometry = tree,
    target: np.ndarray = PI_B,
    support_subset: Optional[np.ndarray] = None,
) -> float:
    entries = panel.resolved_entries
    if not entries:
        return np.nan
    rows = np.arange(geom.n) if support_subset is None else np.asarray(support_subset, int)
    target_local = target[rows]
    target_local = target_local / target_local.sum()
    best = np.zeros(len(rows))
    for entry in entries:
        expected = np.zeros(len(rows))
        for idx, probability in zip(entry.candidate_idx, entry.posterior):
            expected += probability * np.exp(-geom.distances_to(int(idx))[rows] / geom.tau)
        best = np.maximum(best, expected)
    return float(np.dot(target_local, best))


def metric_bundle(panel: SoftPanel, geom: TreeGeometry = tree) -> Dict[str, object]:
    sigma = panel.sigma()
    if sigma.sum() <= 0:
        return {
            "n_input": panel.n_input, "n_resolved": 0, "resolution_rate": 0.0,
            "adoption_gap": np.nan, "breadth": np.nan,
            "structural_linf": np.nan,
        }
    values: Dict[str, object] = {
        "n_input": panel.n_input,
        "n_resolved": panel.n_resolved,
        "resolution_rate": panel.n_resolved / max(panel.n_input, 1),
        "posterior_effective_variants": float(1.0 / np.sum(sigma ** 2)),
        "adoption_gap": geom.w1(PI_A, sigma),
        "breadth": soft_breadth(panel, geom, PI_B),
    }
    values.update(soft_hill_spectrum(panel, PI_A))
    values.update(discrepancy_b.compute(sigma))
    return values

# %% [markdown]
# ## 5. Regeneracao integral dos 49 papers
#
# O notebook ignora os `matched_models.csv` antigos e parte da anotacao
# canonica. Cada uma das 526 entradas e resolvida contra
# `oome_models_with_lineage.parquet`. Os tres papers sem modelos open-weight
# anotados permanecem no arquivo final com `n_input=0`.

# %%
annotations = json.loads(ANNOTATION_FILE.read_text(encoding="utf-8"))
included = [
    record for record in annotations
    if str(record.get("include_final_sample", "")).startswith("yes")
]
assert len(included) == 49, f"Expected 49 included papers, got {len(included)}"


def split_model_list(value: object) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [item.strip() for item in text.split(";") if item.strip()]


paper_panels: Dict[str, SoftPanel] = {}
paper_metadata_rows: List[Dict[str, object]] = []
resolution_audit_rows: List[Dict[str, object]] = []
posterior_rows: List[Dict[str, object]] = []

for record in included:
    paper_id = str(record["paper_id"])
    names = split_model_list(record.get("coverage_model_list_open_only", ""))
    entries = [resolver.resolve(name) for name in names]
    panel = SoftPanel(entries)
    paper_panels[paper_id] = panel
    paper_metadata_rows.append({
        "paper_id": paper_id,
        "title": record.get("title", ""),
        "year": record.get("year"),
        "venue": record.get("venue", ""),
        "sample_role": record.get("include_final_sample", ""),
        "annotation_n_models": record.get("n_coverage_models_open_only", 0),
        "n_input": panel.n_input,
        "n_resolved": panel.n_resolved,
        "n_unresolved": panel.n_input - panel.n_resolved,
        "resolution_rate": panel.n_resolved / max(panel.n_input, 1) if panel.n_input else np.nan,
    })
    for input_position, entry in enumerate(entries):
        audit = entry.audit_row()
        audit.update({"paper_id": paper_id, "input_position": input_position})
        resolution_audit_rows.append(audit)
        for idx, probability in zip(entry.candidate_idx, entry.posterior):
            posterior_rows.append({
                "paper_id": paper_id,
                "input_position": input_position,
                "input_model": entry.input_model,
                "variant_idx": int(idx),
                "variant_id": variant.iloc[int(idx)]["variant_id"],
                "posterior": float(probability),
                "sigma_mass": float(probability / max(panel.n_resolved, 1)),
                "status": entry.status,
                "match_type": entry.match_type,
            })

paper_metadata = pd.DataFrame(paper_metadata_rows)
resolution_audit = pd.DataFrame(resolution_audit_rows)
paper_posterior = pd.DataFrame(posterior_rows)
paper_metadata.to_csv(TABLE_DIR / "paper_resolution_summary.csv", index=False)
resolution_audit.to_csv(TABLE_DIR / "corrected_resolution_audit.csv", index=False)
paper_posterior.to_parquet(TABLE_DIR / "paper_variant_posterior.parquet", index=False)

display(paper_metadata.sort_values(["resolution_rate", "n_input"]).head(12))
display(resolution_audit["status"].value_counts(dropna=False).to_frame("entries"))
print("Papers rerun:", len(paper_panels))
print("Input names:", len(resolution_audit))
print("Resolved input names:", int((resolution_audit["n_candidate_variants"] > 0).sum()))

# %% [markdown]
# ## 6. Testes matematicos e teste especifico contra inflacao ambigua
#
# Alem da equivalencia com transporte linear e das propriedades de
# submodularidade, construimos um ataque direto: uma entrada ambigua com `k`
# candidatos. A implementacao incorreta credita `k/N` em `C(0)`; a correta
# credita apenas `sum_i p_i / N = 1/N`. Repetir a mesma entrada nao altera
# breadth, pois o maximo recebe a mesma funcao esperada.

# %%
def ot_lp_distance(distance: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    constraints, rhs = [], []
    for i in range(n):
        row = np.zeros((n, n))
        row[i, :] = 1
        constraints.append(row.ravel())
        rhs.append(a[i])
    for j in range(n):
        col = np.zeros((n, n))
        col[:, j] = 1
        constraints.append(col.ravel())
        rhs.append(b[j])
    result = linprog(
        distance.ravel(), A_eq=np.vstack(constraints), b_eq=np.asarray(rhs),
        bounds=(0, None), method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.fun)


tiny_idx = RNG.choice(len(variant), 7, replace=False)
tiny = variant.iloc[tiny_idx].reset_index(drop=True)
tiny_a = RNG.dirichlet(np.ones(7))
tiny_b = RNG.dirichlet(np.ones(7))
tiny_tree = TreeGeometry(tiny, tiny_a)
tiny_distance = np.array([
    [tiny_tree.pair_distance(i, j) for j in range(7)] for i in range(7)
])
closed = tiny_tree.w1(tiny_a, tiny_b)
linear = ot_lp_distance(tiny_distance, tiny_a, tiny_b)

ambiguous_candidates = np.array(RNG.choice(len(variant), 25, replace=False), dtype=int)
ambiguous_entry = SoftResolution(
    "synthetic ambiguous family", "synthetic", "synthetic", 1.0,
    ambiguous_candidates, np.full(25, 1.0 / 25), "ambiguity inflation test",
)
ambiguous_panel = SoftPanel([ambiguous_entry])
soft_c0 = soft_hill_spectrum(ambiguous_panel)["coverage_q0"]
expanded_c0 = len(ambiguous_candidates) / len(variant)
expected_soft_c0 = 1.0 / len(variant)
ambiguous_breadth = soft_breadth(ambiguous_panel)
clone_ambiguous_breadth = soft_breadth(SoftPanel([ambiguous_entry] * 10))

submodular_checks = []
for _ in range(25):
    pool = RNG.choice(len(variant), 7, replace=False)
    singleton = [
        SoftResolution(str(idx), "exact", "exact", 1.0, np.array([idx]), np.array([1.0]), "")
        for idx in pool
    ]
    a = SoftPanel(singleton[:3])
    b = SoftPanel(singleton[:6])
    x = singleton[6]
    gain_a = soft_breadth(SoftPanel(a.entries + [x])) - soft_breadth(a)
    gain_b = soft_breadth(SoftPanel(b.entries + [x])) - soft_breadth(b)
    submodular_checks.append(gain_a + 1e-12 >= gain_b)

bound_ratios = []
for _ in range(40):
    selected = RNG.choice(len(variant), 10, replace=False, p=PI_A)
    sigma = np.bincount(selected, minlength=len(variant)) / len(selected)
    w1 = tree.w1(PI_A, sigma)
    probe = tree.distances_to(int(RNG.integers(len(variant))))
    error = abs(np.dot(PI_A, probe) - np.dot(sigma, probe))
    bound_ratios.append(error / max(w1, 1e-15))

sanity = pd.DataFrame([
    {
        "test": "Tree-W1 closed form equals LP",
        "value": abs(closed - linear), "threshold": "<1e-8",
        "pass": abs(closed - linear) < 1e-8,
    },
    {
        "test": "all branch lengths strictly positive",
        "value": tree.lengths.min(), "threshold": ">0",
        "pass": tree.lengths.min() > 0,
    },
    {
        "test": "ambiguous C0 conserves one-entry credit",
        "value": abs(soft_c0 - expected_soft_c0), "threshold": "<1e-12",
        "pass": abs(soft_c0 - expected_soft_c0) < 1e-12,
    },
    {
        "test": "ambiguous soft C0 below expansion",
        "value": soft_c0 / expanded_c0, "threshold": "=1/k",
        "pass": soft_c0 < expanded_c0,
    },
    {
        "test": "soft breadth clone invariant",
        "value": abs(ambiguous_breadth - clone_ambiguous_breadth), "threshold": "<1e-12",
        "pass": abs(ambiguous_breadth - clone_ambiguous_breadth) < 1e-12,
    },
    {
        "test": "breadth diminishing returns",
        "value": np.mean(submodular_checks), "threshold": "=1",
        "pass": all(submodular_checks),
    },
    {
        "test": "1-Lipschitz probe obeys W1 bound",
        "value": max(bound_ratios), "threshold": "<=1",
        "pass": max(bound_ratios) <= 1.0 + 1e-8,
    },
])
display(sanity)
sanity.to_csv(TABLE_DIR / "axiomatic_and_ambiguity_tests.csv", index=False)
assert sanity["pass"].all(), sanity.loc[~sanity["pass"]]

# %% [markdown]
# ## 7. Coordenadas corrigidas para os papers
#
# `Eff_g = G_a(P_g)/G_a(P)` e salvo sem clipping. Valores maiores que um
# significam que o paper supera a heuristica; valores menores que um significam
# que fica atras dela. Isto nao e apresentado como razao para o otimo.

# %%
paper_metric_rows: List[Dict[str, object]] = []
for paper_id, panel in paper_panels.items():
    paper_metric_rows.append({"paper_id": paper_id, **metric_bundle(panel)})
paper_metrics = paper_metadata.merge(pd.DataFrame(paper_metric_rows), on="paper_id", how="left")

greedy_reference: Dict[int, float] = {}
for n in sorted(paper_metrics["n_resolved_x"].dropna().astype(int).unique()):
    if n > 0:
        _, greedy_reference[n] = tree.greedy_w1(n)
paper_metrics["greedy_adoption_gap"] = paper_metrics["n_resolved_x"].map(greedy_reference)
paper_metrics["efficiency_vs_greedy"] = (
    paper_metrics["greedy_adoption_gap"] / paper_metrics["adoption_gap"]
)
paper_metrics = paper_metrics.rename(columns={
    "n_input_x": "annotation_input_count",
    "n_resolved_x": "annotation_resolved_count",
    "resolution_rate_x": "annotation_resolution_rate",
    "n_input_y": "metric_input_count",
    "n_resolved_y": "metric_resolved_count",
    "resolution_rate_y": "metric_resolution_rate",
})
paper_metrics.to_csv(TABLE_DIR / "paper_metrics_corrected.csv", index=False)

display(
    paper_metrics[[
        "paper_id", "annotation_input_count", "annotation_resolved_count",
        "adoption_gap", "breadth", "structural_linf",
        "coverage_q0", "coverage_q1", "coverage_delta_1_0",
        "efficiency_vs_greedy",
    ]].sort_values("adoption_gap").head(15)
)

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
sns.histplot(paper_metrics["adoption_gap"], bins=16, ax=axes[0], color="#4472C4")
sns.histplot(paper_metrics["breadth"], bins=16, ax=axes[1], color="#70AD47")
sns.histplot(paper_metrics["structural_linf"], bins=16, ax=axes[2], color="#ED7D31")
axes[0].set_title("Adoption gap (lower)")
axes[1].set_title("Lineage breadth (higher)")
axes[2].set_title("Structural imbalance (lower)")
plt.tight_layout()
plt.savefig(FIG_DIR / "paper_profile_distributions.png", dpi=220, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Geometria como parte do estimand
#
# Recalculamos `G_a` sob quatro geometrias plausiveis: surpresa estrutural por
# variantes, topologica, surpresa ponderada por numero de repositorios e a
# geometria acoplada a adocao como ablacao. O resultado principal inclui o
# intervalo de rank de cada paper, nao apenas uma ordenacao pontual.

# %%
geometry_family = {
    "structural_variant_count": tree,
    "topological": TreeGeometry(variant, PI_A, count_mode="topological"),
    "structural_repo_count": TreeGeometry(variant, PI_A, count_mode="repo_count"),
    "adoption_weighted_ablation": TreeGeometry(variant, PI_A, count_mode="adoption_weighted"),
}
geometry_rows: List[Dict[str, object]] = []
for geometry_name, geometry in geometry_family.items():
    for paper_id, panel in paper_panels.items():
        sigma = panel.sigma()
        geometry_rows.append({
            "paper_id": paper_id,
            "geometry": geometry_name,
            "adoption_gap": geometry.w1(PI_A, sigma) if sigma.sum() else np.nan,
        })
geometry_scores = pd.DataFrame(geometry_rows)
geometry_scores["rank"] = geometry_scores.groupby("geometry")["adoption_gap"].rank(
    method="average", ascending=True
)
rank_intervals = (
    geometry_scores.groupby("paper_id")
    .agg(
        rank_best=("rank", "min"),
        rank_worst=("rank", "max"),
        rank_width=("rank", lambda x: x.max() - x.min()),
    )
    .reset_index()
)
geometry_pivot = geometry_scores.pivot(index="paper_id", columns="geometry", values="adoption_gap")
geometry_tau_rows = []
for left in geometry_pivot.columns:
    for right in geometry_pivot.columns:
        if left < right:
            common = geometry_pivot[[left, right]].dropna()
            geometry_tau_rows.append({
                "geometry_a": left,
                "geometry_b": right,
                "kendall_tau": kendalltau(common[left], common[right]).statistic,
                "n_papers": len(common),
            })
geometry_tau = pd.DataFrame(geometry_tau_rows)
rank_intervals.to_csv(TABLE_DIR / "paper_geometry_rank_intervals.csv", index=False)
geometry_scores.to_csv(TABLE_DIR / "paper_geometry_scores.csv", index=False)
geometry_tau.to_csv(TABLE_DIR / "geometry_rank_correlations.csv", index=False)
display(geometry_tau.sort_values("kendall_tau"))
display(rank_intervals.sort_values("rank_width", ascending=False).head(12))

# %% [markdown]
# ## 9. Validade de construto e calibracao nula
#
# Paineis sinteticos testam sinais esperados: clones, concentracao em uma
# familia, amostragem por popularidade e estratificacao por familia. A
# calibracao nula sempre compara paineis do mesmo tamanho.

# %%
family_groups = {
    family: np.asarray(indices, int)
    for family, indices in variant.groupby("family_id").indices.items()
}
families = np.array(list(family_groups), dtype=object)
family_mass_a = np.array([PI_A[family_groups[family]].sum() for family in families])


def exact_soft_panel(indices: Sequence[int]) -> SoftPanel:
    return SoftPanel([
        SoftResolution(
            str(variant.iloc[int(idx)]["variant_id"]), "exact", "exact", 1.0,
            np.array([int(idx)]), np.array([1.0]), "",
        )
        for idx in indices
    ])


def random_indices(
    n: int, strategy: str, rng: np.random.Generator,
    support: Optional[np.ndarray] = None, probability: Optional[np.ndarray] = None,
) -> List[int]:
    support = np.arange(len(variant)) if support is None else np.asarray(support, int)
    probability = PI_A[support] if probability is None else np.asarray(probability, float)
    probability = probability / probability.sum()
    if strategy == "uniform":
        return rng.choice(support, n, replace=len(support) < n).astype(int).tolist()
    if strategy == "popularity":
        return rng.choice(
            support, n, replace=len(support) < n, p=probability
        ).astype(int).tolist()
    support_families = variant.iloc[support]["family_id"].to_numpy()
    available_families = np.unique(support_families)
    if strategy == "clustered":
        family_weights = np.array([
            probability[support_families == family].sum() for family in available_families
        ])
        chosen_family = rng.choice(available_families, p=family_weights / family_weights.sum())
        local = support[support_families == chosen_family]
        local_p = PI_A[local] / PI_A[local].sum()
        return rng.choice(local, n, replace=len(local) < n, p=local_p).astype(int).tolist()
    if strategy == "family_stratified":
        family_weights = np.array([
            probability[support_families == family].sum() for family in available_families
        ])
        chosen_families = rng.choice(
            available_families, min(n, len(available_families)), replace=False,
            p=family_weights / family_weights.sum(),
        )
        selected: List[int] = []
        for family in chosen_families:
            local = support[support_families == family]
            local_p = PI_A[local] / PI_A[local].sum()
            selected.append(int(rng.choice(local, p=local_p)))
        while len(selected) < n:
            candidate = int(rng.choice(support, p=probability))
            if candidate not in selected or len(support) < n:
                selected.append(candidate)
        return selected
    raise ValueError(strategy)


n_construct = 10
greedy_indices, _ = tree.greedy_w1(n_construct)
largest_family = families[np.argmax(family_mass_a)]
largest_indices = family_groups[largest_family]
one_family = largest_indices[np.argsort(-PI_A[largest_indices])[:n_construct]].tolist()
top_idx = int(np.argmax(PI_A))
construct_panels = {
    "greedy_adoption": greedy_indices,
    "top_popularity": np.argsort(-PI_A)[:n_construct].tolist(),
    "family_stratified": random_indices(n_construct, "family_stratified", RNG),
    "uniform": random_indices(n_construct, "uniform", RNG),
    "one_family": one_family,
    "clone": [top_idx] * n_construct,
}
construct = pd.DataFrame([
    {"panel": name, **metric_bundle(exact_soft_panel(indices))}
    for name, indices in construct_panels.items()
])
construct.to_csv(TABLE_DIR / "construct_validity_corrected.csv", index=False)
display(construct[[
    "panel", "adoption_gap", "breadth", "structural_linf",
    "coverage_q0", "coverage_q1",
]].sort_values("adoption_gap"))

null_rows: List[Dict[str, object]] = []
for n in sorted(set(paper_metrics["metric_resolved_count"].fillna(0).astype(int)) - {0}):
    for strategy in ["uniform", "popularity", "family_stratified"]:
        rng = np.random.default_rng(SEED + 101 * n + len(strategy))
        for draw in range(N_NULL):
            values = metric_bundle(exact_soft_panel(random_indices(n, strategy, rng)))
            null_rows.append({"n": n, "strategy": strategy, "draw": draw, **values})
null_scores = pd.DataFrame(null_rows)
calibration_rows: List[Dict[str, object]] = []
for row in paper_metrics.itertuples(index=False):
    n = int(getattr(row, "metric_resolved_count") or 0)
    if n <= 0 or not np.isfinite(getattr(row, "adoption_gap")):
        continue
    for strategy in ["uniform", "popularity", "family_stratified"]:
        null = null_scores[(null_scores["n"] == n) & (null_scores["strategy"] == strategy)]
        calibration_rows.append({
            "paper_id": row.paper_id,
            "n": n,
            "strategy": strategy,
            "adoption_quality_percentile": float(np.mean(null["adoption_gap"] >= row.adoption_gap)),
            "breadth_quality_percentile": float(np.mean(null["breadth"] <= row.breadth)),
            "structural_quality_percentile": float(np.mean(null["structural_linf"] >= row.structural_linf)),
        })
null_calibration = pd.DataFrame(calibration_rows)
null_calibration.to_csv(TABLE_DIR / "paper_null_calibration_corrected.csv", index=False)
display(
    null_calibration.groupby("strategy")[
        ["adoption_quality_percentile", "breadth_quality_percentile", "structural_quality_percentile"]
    ].median()
)

# %% [markdown]
# ## 10. Snapshots comportamentais externos
#
# Usamos dois snapshots oficiais e congelados:
#
# * Open LLM Leaderboard `contents`, commit
#   `9c09a7cae43334062a82cb164f2ef255013dafa2`;
# * SALAD-Bench Leaderboard, commit
#   `187161d5453ea80512b6625d02aec8b4ecfdb546`.
#
# O notebook nao depende do estado corrente dos leaderboards. Checksums locais,
# URLs e commits sao salvos no manifesto. O arquivo XLSX do SALAD e lido
# diretamente via XML para nao acrescentar uma dependencia de `openpyxl`.

# %%
OLLB_FILE = EXTERNAL_DIR / "open_llm_leaderboard.parquet"
SALAD_FILE = EXTERNAL_DIR / "salad_bench_leaderboard.xlsx"
OLLB_URL = (
    "https://huggingface.co/datasets/open-llm-leaderboard/contents/resolve/"
    "9c09a7cae43334062a82cb164f2ef255013dafa2/data/train-00000-of-00001.parquet"
)
SALAD_URL = (
    "https://huggingface.co/spaces/OpenSafetyLab/Salad-Bench-Leaderboard/resolve/"
    "187161d5453ea80512b6625d02aec8b4ecfdb546/file/leaderboard.xlsx"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


assert OLLB_FILE.exists(), f"Missing frozen external snapshot: {OLLB_FILE}"
assert SALAD_FILE.exists(), f"Missing frozen external snapshot: {SALAD_FILE}"


def xlsx_sheet(path: Path, wanted_name: str) -> pd.DataFrame:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    office_rel = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    with zipfile.ZipFile(path) as archive:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append("".join(node.text or "" for node in item.iterfind(".//m:t", ns)))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target_by_id = {
            item.attrib["Id"]: item.attrib["Target"] for item in relationships.findall("r:Relationship", rel_ns)
        }
        sheet_target = None
        for item in workbook.findall(".//m:sheet", ns):
            if item.attrib.get("name") == wanted_name:
                sheet_target = target_by_id[item.attrib[office_rel]]
                break
        if sheet_target is None:
            raise KeyError(f"Sheet {wanted_name!r} not found")
        sheet_target = sheet_target.lstrip("/")
        if not sheet_target.startswith("xl/"):
            sheet_target = "xl/" + sheet_target
        root = ET.fromstring(archive.read(sheet_target))

        rows: List[Dict[int, object]] = []
        for row in root.findall(".//m:sheetData/m:row", ns):
            values: Dict[int, object] = {}
            for cell in row.findall("m:c", ns):
                ref = cell.attrib.get("r", "A1")
                letters = re.match(r"[A-Z]+", ref).group(0)
                column = 0
                for letter in letters:
                    column = column * 26 + ord(letter) - 64
                column -= 1
                cell_type = cell.attrib.get("t")
                value_node = cell.find("m:v", ns)
                inline = cell.find("m:is", ns)
                value: object = ""
                if cell_type == "inlineStr" and inline is not None:
                    value = "".join(node.text or "" for node in inline.iterfind(".//m:t", ns))
                elif value_node is not None:
                    text = value_node.text or ""
                    if cell_type == "s":
                        value = shared[int(text)]
                    elif cell_type == "b":
                        value = text == "1"
                    else:
                        try:
                            value = float(text)
                        except ValueError:
                            value = text
                values[column] = value
            rows.append(values)
    if not rows:
        return pd.DataFrame()
    width = max(max(row.keys(), default=-1) for row in rows) + 1
    matrix = [[row.get(column, "") for column in range(width)] for row in rows]
    header = [str(value).strip() for value in matrix[0]]
    return pd.DataFrame(matrix[1:], columns=header)


ollb_raw = pd.read_parquet(OLLB_FILE)
salad_base_raw = xlsx_sheet(SALAD_FILE, "base_order")
salad_attack_raw = xlsx_sheet(SALAD_FILE, "attack_order")
print("Open LLM Leaderboard rows:", len(ollb_raw))
print("SALAD base rows:", len(salad_base_raw))
print("SALAD attack rows:", len(salad_attack_raw))
display(salad_base_raw.head())

# %% [markdown]
# ## 11. Resolucao dos outcomes e populacoes condicionadas
#
# Outcomes externos so entram quando o repo resolve para uma unica variante.
# Nao distribuimos a nota de um modelo do leaderboard por uma familia inteira.
# Para capacidade, exigimos observacao conjunta das seis tarefas; para
# seguranca, exigimos os dois scores SALAD. Cada grupo define sua propria
# populacao condicionada e seus targets `pi_a` e `pi_b` renormalizados.

# %%
CAPABILITY_OUTCOMES = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]


def resolve_external_unique(name: object) -> Tuple[Optional[int], SoftResolution]:
    result = resolver.resolve(name, allow_fuzzy=False)
    if result.resolved and len(result.candidate_idx) == 1:
        return int(result.candidate_idx[0]), result
    return None, result


ollb_rows: List[Dict[str, object]] = []
ollb_audit_rows: List[Dict[str, object]] = []
for row in ollb_raw.to_dict("records"):
    fullname = row.get("fullname", "")
    idx, resolution = resolve_external_unique(fullname)
    audit = resolution.audit_row()
    audit.update({"source": "open_llm_leaderboard", "external_name": fullname})
    ollb_audit_rows.append(audit)
    if idx is None:
        continue
    values = {"variant_idx": idx, "external_name": fullname}
    for outcome in ["Average \u2b06\ufe0f", *CAPABILITY_OUTCOMES]:
        values[outcome] = pd.to_numeric(row.get(outcome), errors="coerce")
    ollb_rows.append(values)

ollb_matched = pd.DataFrame(ollb_rows)
ollb_variant = ollb_matched.groupby("variant_idx")[
    ["Average \u2b06\ufe0f", *CAPABILITY_OUTCOMES]
].median()
capability_complete = ollb_variant.dropna(subset=CAPABILITY_OUTCOMES).copy()


def find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    normalized = {normalize_key(column): column for column in frame.columns}
    for candidate in candidates:
        if normalize_key(candidate) in normalized:
            return normalized[normalize_key(candidate)]
    return None


def clean_hf_reference(value: object) -> str:
    text = normalize_text(value)
    text = re.sub(r"^https?://huggingface\.co/", "", text, flags=re.I)
    return text.strip("/")


def salad_outcome(frame: pd.DataFrame, outcome_name: str) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    model_col = find_column(frame, ["Model", "model name"])
    link_col = find_column(frame, ["Link", "URL"])
    score_col = find_column(frame, ["total", "score", "Total Score"])
    open_col = find_column(frame, ["Open or Not", "Open", "open source"])
    if model_col is None or score_col is None:
        raise ValueError(f"Could not identify SALAD columns: {frame.columns.tolist()}")
    rows, audit_rows = [], []
    for record in frame.to_dict("records"):
        if open_col is not None:
            raw_open = record.get(open_col, "")
            try:
                is_open = float(raw_open) > 0.5
            except (TypeError, ValueError):
                openness = str(raw_open).lower()
                is_open = "open" in openness and "close" not in openness
            if not is_open:
                continue
        link = clean_hf_reference(record.get(link_col, "")) if link_col else ""
        name = link if "/" in link else clean_hf_reference(record.get(model_col, ""))
        idx, resolution = resolve_external_unique(name)
        audit = resolution.audit_row()
        audit.update({"source": f"salad_{outcome_name}", "external_name": name})
        audit_rows.append(audit)
        if idx is None:
            continue
        score = pd.to_numeric(record.get(score_col), errors="coerce")
        if np.isfinite(score):
            rows.append({"variant_idx": idx, outcome_name: float(score)})
    return pd.DataFrame(rows), audit_rows


salad_base, salad_base_audit = salad_outcome(salad_base_raw, "salad_base_safety")
salad_attack, salad_attack_audit = salad_outcome(salad_attack_raw, "salad_attack_safety")
if len(salad_base) and len(salad_attack):
    safety_complete = (
        salad_base.groupby("variant_idx")["salad_base_safety"].median().to_frame()
        .join(salad_attack.groupby("variant_idx")["salad_attack_safety"].median(), how="inner")
        .dropna()
    )
else:
    safety_complete = pd.DataFrame(columns=["salad_base_safety", "salad_attack_safety"])

external_audit = pd.DataFrame(ollb_audit_rows + salad_base_audit + salad_attack_audit)
external_audit.to_csv(TABLE_DIR / "external_outcome_resolution_audit.csv", index=False)
ollb_variant.reset_index().to_csv(TABLE_DIR / "open_llm_leaderboard_by_variant.csv", index=False)
safety_complete.reset_index().to_csv(TABLE_DIR / "salad_safety_by_variant.csv", index=False)

external_coverage = pd.DataFrame([
    {
        "outcome_group": "capability_complete_case",
        "external_rows": len(ollb_raw),
        "matched_unique_variants": len(ollb_variant),
        "conditioned_population": len(capability_complete),
        "outcomes": "; ".join(CAPABILITY_OUTCOMES),
    },
    {
        "outcome_group": "safety_complete_case",
        "external_rows": len(salad_base_raw) + len(salad_attack_raw),
        "matched_unique_variants": len(set(salad_base.get("variant_idx", [])) | set(salad_attack.get("variant_idx", []))),
        "conditioned_population": len(safety_complete),
        "outcomes": "salad_base_safety; salad_attack_safety",
    },
])
external_coverage.to_csv(TABLE_DIR / "external_outcome_coverage.csv", index=False)
display(external_coverage)

assert len(capability_complete) >= 100, "Capability conditioned population is too small"
assert len(safety_complete) >= 12, "Safety conditioned population is too small"

# %% [markdown]
# ## 12. Validade externa out-of-sample
#
# O estimand e o erro padronizado de recuperacao da media do outcome sob
# `pi_a` condicionada. Para cada tamanho e estrategia geramos paineis dentro da
# populacao observada. O baseline possui `log(n)`, popularidade media e entropia
# de familias. Cada coordenada e acrescentada separadamente, e um modelo conjunto
# testa o perfil inteiro. `GroupKFold` deixa um tamanho de painel completamente
# fora em cada fold. Intervalos sao bootstrap estratificado por tamanho.

# %%
def family_entropy(indices: Sequence[int]) -> float:
    families_local = variant.iloc[np.asarray(indices, int)]["family_id"].value_counts(normalize=True)
    return float(-np.sum(families_local * np.log(np.maximum(families_local, 1e-15))))


def conditioned_targets(support: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    support = np.asarray(support, int)
    target_a = np.zeros(len(variant))
    target_b = np.zeros(len(variant))
    target_a[support] = PI_A[support] / PI_A[support].sum()
    target_b[support] = PI_B[support] / PI_B[support].sum()
    return target_a, target_b


def conditioned_profile(
    indices: Sequence[int],
    support: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    discrepancy: StructuralDiscrepancy,
) -> Dict[str, float]:
    indices = np.asarray(indices, int)
    sigma = np.bincount(indices, minlength=len(variant)) / len(indices)
    unique = np.unique(indices)
    c0 = len(unique) / len(support)
    c1 = float(target_a[unique].sum())
    panel = exact_soft_panel(indices)
    return {
        "adoption_gap": tree.w1(target_a, sigma),
        "breadth": soft_breadth(panel, tree, target_b, support_subset=support),
        "structural_linf": float(discrepancy.compute(sigma)["structural_linf"]),
        "coverage_q0": c0,
        "coverage_q1": c1,
        "coverage_delta_1_0": c1 - c0,
        "log_n": math.log(len(indices)),
        "mean_log_popularity": float(np.mean(np.log1p(DOWNLOADS[indices]))),
        "family_entropy": family_entropy(indices),
    }


def simulation_sizes(population_size: int) -> List[int]:
    candidates = [3, 5, 10, 20]
    sizes = sorted(set(min(value, max(2, population_size // 3)) for value in candidates))
    return [value for value in sizes if value < population_size and value >= 2]


def simulate_outcome_group(
    group_name: str, outcome_frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    support = outcome_frame.index.to_numpy(int)
    target_a, target_b = conditioned_targets(support)
    local_discrepancy = StructuralDiscrepancy(variant, target_b)
    sizes = simulation_sizes(len(support))
    strategies = ["uniform", "popularity", "family_stratified", "clustered"]
    rows: List[Dict[str, object]] = []
    panel_id = 0
    for n in sizes:
        for strategy_index, strategy in enumerate(strategies):
            rng = np.random.default_rng(SEED + 1009 * n + 37 * strategy_index + len(group_name))
            for repetition in range(N_SIM_PER_CELL):
                indices = random_indices(
                    n, strategy, rng, support=support, probability=target_a[support]
                )
                profile = conditioned_profile(
                    indices, support, target_a, target_b, local_discrepancy
                )
                row: Dict[str, object] = {
                    "outcome_group": group_name,
                    "panel_id": panel_id,
                    "n": n,
                    "strategy": strategy,
                    "repetition": repetition,
                    **profile,
                }
                for outcome in outcome_frame.columns:
                    values = outcome_frame[outcome].to_numpy(float)
                    indexed = pd.Series(values, index=support)
                    population_mean = float(np.dot(target_a[support], values))
                    population_sd = max(
                        float(np.sqrt(np.dot(target_a[support], (values - population_mean) ** 2))),
                        1e-8,
                    )
                    panel_mean = float(indexed.loc[indices].mean())
                    row[f"error_{outcome}"] = abs(panel_mean - population_mean) / population_sd
                    row[f"raw_error_{outcome}"] = abs(panel_mean - population_mean)
                rows.append(row)
                panel_id += 1
    meta = {
        "support": support,
        "target_a": target_a,
        "target_b": target_b,
        "sizes": sizes,
    }
    return pd.DataFrame(rows), meta


capability_sim, capability_meta = simulate_outcome_group(
    "capability", capability_complete[CAPABILITY_OUTCOMES] / 100.0
)
safety_sim, safety_meta = simulate_outcome_group(
    "safety", safety_complete[["salad_base_safety", "salad_attack_safety"]] / 100.0
)
external_simulations = pd.concat([capability_sim, safety_sim], ignore_index=True)
external_simulations.to_parquet(TABLE_DIR / "external_validation_simulations.parquet", index=False)

CONTROL_COLUMNS = ["log_n", "mean_log_popularity", "family_entropy"]
PROFILE_COLUMNS = [
    "adoption_gap", "breadth", "structural_linf",
    "coverage_q0", "coverage_q1", "coverage_delta_1_0",
]


def grouped_cv_predictions(frame: pd.DataFrame, y_column: str, predictors: Sequence[str]) -> np.ndarray:
    data = frame[[*predictors, y_column, "n"]].dropna().copy()
    output = np.full(len(frame), np.nan)
    splitter = GroupKFold(n_splits=data["n"].nunique())
    x = data[list(predictors)].to_numpy(float)
    y = data[y_column].to_numpy(float)
    for train, test in splitter.split(x, y, groups=data["n"].to_numpy()):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x[train], y[train])
        predicted = model.predict(x[test])
        output[data.index.to_numpy()[test]] = predicted
    return output


def incremental_r2(frame: pd.DataFrame, outcome: str, metric_columns: Sequence[str]) -> Dict[str, float]:
    y_column = f"error_{outcome}"
    valid = frame[[y_column, *CONTROL_COLUMNS, *metric_columns, "n"]].dropna().copy()
    base_pred = grouped_cv_predictions(valid, y_column, CONTROL_COLUMNS)
    full_pred = grouped_cv_predictions(valid, y_column, [*CONTROL_COLUMNS, *metric_columns])
    y = valid[y_column].to_numpy(float)
    base_r2 = r2_score(y, base_pred)
    full_r2 = r2_score(y, full_pred)
    return {
        "baseline_oos_r2": float(base_r2),
        "augmented_oos_r2": float(full_r2),
        "incremental_oos_r2": float(full_r2 - base_r2),
    }


def bootstrap_incremental_r2(
    frame: pd.DataFrame, outcome: str, metric_columns: Sequence[str], repetitions: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(SEED + len(outcome) + len(metric_columns))
    values = []
    for _ in range(repetitions):
        sampled_parts = []
        for _, group in frame.groupby("n"):
            chosen = rng.choice(group.index.to_numpy(), len(group), replace=True)
            sampled_parts.append(group.loc[chosen].reset_index(drop=True))
        sampled = pd.concat(sampled_parts, ignore_index=True)
        try:
            values.append(incremental_r2(sampled, outcome, metric_columns)["incremental_oos_r2"])
        except Exception:
            continue
    if not values:
        return np.nan, np.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


validation_rows: List[Dict[str, object]] = []
for group_name, frame, outcomes in [
    ("capability", capability_sim, CAPABILITY_OUTCOMES),
    ("safety", safety_sim, ["salad_base_safety", "salad_attack_safety"]),
]:
    for outcome in outcomes:
        for metric in PROFILE_COLUMNS:
            result = incremental_r2(frame, outcome, [metric])
            low, high = bootstrap_incremental_r2(frame, outcome, [metric], N_CV_BOOT)
            validation_rows.append({
                "outcome_group": group_name,
                "outcome": outcome,
                "predictor": metric,
                **result,
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "n_simulated_panels": len(frame),
                "conditioned_population": len(
                    capability_complete if group_name == "capability" else safety_complete
                ),
            })
        result = incremental_r2(frame, outcome, PROFILE_COLUMNS)
        low, high = bootstrap_incremental_r2(frame, outcome, PROFILE_COLUMNS, N_CV_BOOT)
        validation_rows.append({
            "outcome_group": group_name,
            "outcome": outcome,
            "predictor": "joint_profile",
            **result,
            "bootstrap_95_low": low,
            "bootstrap_95_high": high,
            "n_simulated_panels": len(frame),
            "conditioned_population": len(
                capability_complete if group_name == "capability" else safety_complete
            ),
        })

external_validation = pd.DataFrame(validation_rows)
external_validation.to_csv(TABLE_DIR / "external_grouped_cv_incremental_r2.csv", index=False)
display(
    external_validation.sort_values(
        ["outcome", "incremental_oos_r2"], ascending=[True, False]
    )
)

plt.figure(figsize=(11, 7))
plot_data = external_validation[external_validation["predictor"] != "joint_profile"].copy()
sns.barplot(
    data=plot_data, x="incremental_oos_r2", y="predictor",
    hue="outcome_group", errorbar=None,
)
plt.axvline(0, color="black", linewidth=1)
plt.title("Incremental out-of-sample R2 beyond budget and composition controls")
plt.tight_layout()
plt.savefig(FIG_DIR / "external_incremental_r2.png", dpi=220, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 13. Constante de Lipschitz, bound realizado e papers condicionados
#
# Para cada outcome reportamos a constante empirica
#
# \[
# \ell_g=\max_{i\ne j}\frac{|g_i-g_j|}{d_T(i,j)}
# \]
#
# e o produto `ell_g * G_a`. Isto torna explicito quando o bound de media e
# informativo ou apenas verdadeiro e frouxo. Para papers reais, a distribuicao
# do painel e condicionada ao subconjunto com outcome observado; a massa de
# overlap e sempre registrada.

# %%
def empirical_lipschitz(support: np.ndarray, values: np.ndarray) -> float:
    support = np.asarray(support, int)
    values = np.asarray(values, float)
    maximum = 0.0
    for position, idx in enumerate(support):
        distance = tree.distances_to(int(idx))[support]
        valid = distance > 0
        if np.any(valid):
            maximum = max(
                maximum,
                float(np.max(np.abs(values - values[position])[valid] / distance[valid])),
            )
    return maximum


lipschitz_rows: List[Dict[str, object]] = []
paper_bound_rows: List[Dict[str, object]] = []
for group_name, outcome_frame, meta in [
    ("capability", capability_complete[CAPABILITY_OUTCOMES] / 100.0, capability_meta),
    ("safety", safety_complete[["salad_base_safety", "salad_attack_safety"]] / 100.0, safety_meta),
]:
    support = np.asarray(meta["support"], int)
    target_a = np.asarray(meta["target_a"], float)
    for outcome in outcome_frame.columns:
        values = outcome_frame[outcome].to_numpy(float)
        value_by_idx = pd.Series(values, index=support)
        ell = empirical_lipschitz(support, values)
        target_mean = float(np.dot(target_a[support], values))

        simulation_frame = capability_sim if group_name == "capability" else safety_sim
        bound_ratio = (
            simulation_frame[f"raw_error_{outcome}"]
            / np.maximum(ell * simulation_frame["adoption_gap"], 1e-15)
        )
        lipschitz_rows.append({
            "outcome_group": group_name,
            "outcome": outcome,
            "conditioned_population": len(support),
            "empirical_lipschitz": ell,
            "max_simulated_error_to_bound_ratio": float(bound_ratio.max()),
            "median_simulated_error_to_bound_ratio": float(bound_ratio.median()),
        })

        for paper_id, panel in paper_panels.items():
            sigma = panel.sigma()
            overlap = float(sigma[support].sum())
            if overlap <= 0:
                paper_bound_rows.append({
                    "paper_id": paper_id,
                    "outcome_group": group_name,
                    "outcome": outcome,
                    "outcome_overlap_mass": 0.0,
                    "conditioned_adoption_gap": np.nan,
                    "realized_mean_error": np.nan,
                    "lipschitz_bound": np.nan,
                    "error_to_bound_ratio": np.nan,
                })
                continue
            conditioned_sigma = np.zeros(len(variant))
            conditioned_sigma[support] = sigma[support] / overlap
            gap = tree.w1(target_a, conditioned_sigma)
            panel_mean = float(np.dot(conditioned_sigma[support], values))
            error = abs(panel_mean - target_mean)
            bound = ell * gap
            paper_bound_rows.append({
                "paper_id": paper_id,
                "outcome_group": group_name,
                "outcome": outcome,
                "outcome_overlap_mass": overlap,
                "conditioned_adoption_gap": gap,
                "realized_mean_error": error,
                "lipschitz_bound": bound,
                "error_to_bound_ratio": error / max(bound, 1e-15),
            })

lipschitz_summary = pd.DataFrame(lipschitz_rows)
paper_outcome_bounds = pd.DataFrame(paper_bound_rows)
lipschitz_summary.to_csv(TABLE_DIR / "behavioral_lipschitz_summary.csv", index=False)
paper_outcome_bounds.to_csv(TABLE_DIR / "paper_behavioral_bounds_conditioned.csv", index=False)
display(lipschitz_summary)
assert (lipschitz_summary["max_simulated_error_to_bound_ratio"] <= 1.0 + 1e-8).all()

# %% [markdown]
# ## 14. Veredito empirico e recomendacao para o paper
#
# A escolha nao e feita por elegancia isolada. O notebook resume:
#
# 1. validade axiomatica e resistencia a ambiguidade;
# 2. validade de construto nos ataques sinteticos;
# 3. incremental out-of-sample `R2` por outcome comportamental;
# 4. estabilidade de rank entre geometrias;
# 5. interpretabilidade formal de cada coordenada.
#
# O candidato headline e escolhido entre as coordenadas individuais pela media
# do incremental `R2`, mas a recomendacao final continua sendo um perfil:
# `G_a` para fidelidade de adocao, `B_b` para diversidade de linhagem e `D_b`
# para diagnostico estrutural.

# %%
individual_validation = external_validation[
    external_validation["predictor"].isin(PROFILE_COLUMNS)
]
predictor_summary = (
    individual_validation.groupby("predictor")
    .agg(
        mean_incremental_oos_r2=("incremental_oos_r2", "mean"),
        median_incremental_oos_r2=("incremental_oos_r2", "median"),
        positive_outcomes=("incremental_oos_r2", lambda x: int((x > 0).sum())),
        outcomes=("incremental_oos_r2", "size"),
        mean_ci_low=("bootstrap_95_low", "mean"),
    )
    .sort_values(["mean_incremental_oos_r2", "positive_outcomes"], ascending=False)
    .reset_index()
)
predictor_summary.to_csv(TABLE_DIR / "external_predictor_summary.csv", index=False)
display(predictor_summary)

headline_predictor = str(predictor_summary.iloc[0]["predictor"])
recommendation = pd.DataFrame([
    {
        "role": "headline adoption fidelity",
        "coordinate": "adoption_gap",
        "target": "pi_a",
        "direction": "lower",
        "formal_property": "Tree-W1 IPM; controls mean error for lineage-Lipschitz outcomes",
    },
    {
        "role": "lineage breadth and panel selection",
        "coordinate": "breadth",
        "target": "pi_b",
        "direction": "higher",
        "formal_property": "monotone, submodular, replication invariant",
    },
    {
        "role": "structural diagnosis",
        "coordinate": "structural_linf plus per-partition TV",
        "target": "pi_b",
        "direction": "lower",
        "formal_property": "auditable worst cell and partition-level total variation",
    },
    {
        "role": "observable support description",
        "coordinate": "C(q), especially C(0), C(1), and delta",
        "target": "pi_a",
        "direction": "profile",
        "formal_property": "soft inclusion probability prevents ambiguous-match inflation",
    },
    {
        "role": "same-budget reference",
        "coordinate": "efficiency_vs_greedy",
        "target": "pi_a",
        "direction": "greater than 1 beats greedy",
        "formal_property": "unclipped baseline ratio; no optimality claim",
    },
])
recommendation.to_csv(TABLE_DIR / "metric_recommendation_corrected.csv", index=False)
display(recommendation)

manifest = {
    "created_date": "2026-06-12",
    "seed": SEED,
    "fast_mode": FAST_MODE,
    "paths": {
        "lineage": os.path.relpath(LINEAGE_FILE, WORK_ROOT),
        "annotations": os.path.relpath(ANNOTATION_FILE, WORK_ROOT),
        "work_root": ".",
        "output_root": os.path.relpath(OUT_DIR, WORK_ROOT),
    },
    "targets": {
        "adoption_uniform_mix_alpha": ALPHA,
        "breadth": "uniform branching family -> series -> base -> variant",
    },
    "geometry": {
        "primary": "structural variant-count surprise with positive floor",
        "beta": BETA,
        "lambda0": LAMBDA0,
        "normalization": "expected pi_a root depth equals one",
        "breadth_tau": tree.tau,
        "breadth_tau_rule": "median nearest-neighbor lineage distance",
        "family": list(geometry_family),
    },
    "resolver": {
        "papers": len(paper_panels),
        "input_names": len(resolution_audit),
        "resolved_names": int((resolution_audit["n_candidate_variants"] > 0).sum()),
        "principle": "one input contributes one posterior distribution; no descendant expansion",
    },
    "external_snapshots": {
        "open_llm_leaderboard": {
            "url": OLLB_URL,
            "commit": "9c09a7cae43334062a82cb164f2ef255013dafa2",
            "sha256": sha256_file(OLLB_FILE),
            "conditioned_variants": len(capability_complete),
        },
        "salad_bench": {
            "url": SALAD_URL,
            "commit": "187161d5453ea80512b6625d02aec8b4ecfdb546",
            "sha256": sha256_file(SALAD_FILE),
            "conditioned_variants": len(safety_complete),
        },
    },
    "external_validation": {
        "folding": "GroupKFold by panel size",
        "controls": CONTROL_COLUMNS,
        "bootstrap_repetitions": N_CV_BOOT,
        "empirically_best_individual_predictor": headline_predictor,
        "caution": "headline formal coordinate remains adoption_gap; no scalar replaces the profile",
    },
    "tests_all_passed": bool(sanity["pass"].all()),
}
(OUT_DIR / "validation_manifest_corrected.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)

print("All mathematical tests passed:", sanity["pass"].all())
print("Empirically strongest individual predictor:", headline_predictor)
print("Tables:", TABLE_DIR)
print("Figures:", FIG_DIR)
print("Manifest:", OUT_DIR / "validation_manifest_corrected.json")

