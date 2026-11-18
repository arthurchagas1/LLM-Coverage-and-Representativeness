#!/usr/bin/env python3
"""Build all CSV evidence required for Technical Supplement C.2.1, C.4, and C.5.

Run from the coverage project root:

    python build_appendix_c.py

The script expects the current project layout:

    coverage_metric_validation_source.py
    src/coverage_geometry.py
    outputs_corrected/cache/foundation.pkl

The lineage parquet and full paper-annotation JSON are auto-discovered from the
paths declared in ``coverage_metric_validation_source.py``. They can also be
provided explicitly:

    python build_appendix_c.py \
        --lineage-file /path/to/oome_models_with_lineage.parquet \
        --annotations-file /path/to/annotations.json \
        --cutoffs-file paper_cutoffs.csv

or through the ``COVERAGE_LINEAGE_FILE`` and ``COVERAGE_ANNOTATIONS_FILE``
environment variables.

Outputs are written to ``outputs_corrected/appendix_c`` by default. The pipeline
uses only information supported by the current code and metadata:

* historical availability from the cutoff and eligible set that
  ``analysis/f_feasibility.py`` writes to ``paper_contexts.pkl`` (the newest
  evaluated model, dated by the repository the paper names), so every analysis
  shares one cutoff; variant creation dates only when that file is absent;
* access compatibility from Hugging Face gating metadata;
* task/functional compatibility from ``model_role``;
* a coarse cost envelope from ``size_bucket``.

It does not invent unavailable context-length, language, domain, precision,
throughput, hardware, or active-parameter constraints. Those missing dimensions
are recorded explicitly in the C.2.1 CSVs.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import importlib
import pickle
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


SEED = 20260612
DEFAULT_N_NULL = 120
SIZE_ORDER = [
    "<1B",
    "1B-3B",
    "3B-7B",
    "7B-13B",
    "13B-34B",
    "34B-70B",
    "70B-180B",
    "180B+",
]
SIZE_RANK = {name: i for i, name in enumerate(SIZE_ORDER)}
GATE_LABEL = {0: "open_or_not_gated", 1: "auto_gated", 2: "manual_or_restricted"}
TEXT_MISSING = {"", "nan", "none", "null", "missing", "unknown", "n/a", "na"}

# Data-directory layouts this script knows how to find, relative to the project
# root or to its parent. No machine- or account-specific path is hard-coded:
# pass --lineage-file / --annotations-file, or set COVERAGE_LINEAGE_FILE /
# COVERAGE_ANNOTATIONS_FILE, for any other layout.
SIBLING_DATA_ROOTS: List[str] = []  # DMKD27 is self-contained: never fall back to older checkouts.
LINEAGE_RELATIVE = Path("data/oome_lineages_20261105/oome_models_with_lineage.parquet")
ANNOTATIONS_RELATIVE = Path("data/paper_annotations/acl15_new_schema.json")


# ---------------------------------------------------------------------------
# Project discovery
# ---------------------------------------------------------------------------

def resolve_project_root(raw: Path) -> Path:
    root = raw.expanduser().resolve()
    if (root / "src" / "coverage_geometry.py").exists():
        return root
    if (root / "coverage" / "src" / "coverage_geometry.py").exists():
        return root / "coverage"
    raise FileNotFoundError(
        "Could not find src/coverage_geometry.py. Run this script from the "
        "coverage project root or pass --root."
    )


def declared_path(source_file: Path, variable: str) -> Optional[Path]:
    """Read a simple pathlib assignment such as DATA_ROOT / 'data/file'."""
    if not source_file.exists():
        return None
    text = source_file.read_text(encoding="utf-8", errors="replace")

    # Direct quoted path: NAME = Path("/...") or NAME = "/..."
    patterns = [
        rf"^{re.escape(variable)}\s*=\s*Path\((['\"])(.*?)\1\)",
        rf"^{re.escape(variable)}\s*=\s*(['\"])(.*?)\1",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return Path(match.group(2)).expanduser()

    # Resolve the exact assignments used by this project.
    roots: Dict[str, Path] = {}
    for name in ["DATA_ROOT", "WORK_ROOT"]:
        match = re.search(
            rf"^{name}\s*=\s*Path\((['\"])(.*?)\1\)", text, flags=re.MULTILINE
        )
        if match:
            roots[name] = Path(match.group(2)).expanduser()
    match = re.search(
        rf"^{re.escape(variable)}\s*=\s*([A-Z_]+)\s*/\s*(['\"])(.*?)\2",
        text,
        flags=re.MULTILINE,
    )
    if match and match.group(1) in roots:
        return roots[match.group(1)] / match.group(3)
    return None


def first_existing(candidates: Iterable[Optional[Path]], label: str) -> Path:
    seen: List[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser()
        seen.append(str(path))
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not locate {label}. Tried: {seen}")


def sibling_candidates(root: Path, relative: Path) -> List[Path]:
    """Portable fallbacks: the project, its parent, then sibling checkouts.

    The parent case covers the supplementary package, where this project is a
    subdirectory of the repository that carries ``data/``.
    """
    return [root / relative, root.parent / relative] + [
        root.parent / name / relative for name in SIBLING_DATA_ROOTS
    ]


def discover_lineage_file(root: Path, explicit: Optional[str]) -> Path:
    source = root / "coverage_metric_validation_source.py"
    return first_existing(
        [
            Path(explicit) if explicit else None,
            Path(os.environ["COVERAGE_LINEAGE_FILE"]) if os.environ.get("COVERAGE_LINEAGE_FILE") else None,
            declared_path(source, "LINEAGE_FILE"),
            *sibling_candidates(root, LINEAGE_RELATIVE),
        ],
        "lineage parquet",
    )


def discover_annotations_file(root: Path, explicit: Optional[str]) -> Optional[Path]:
    source = root / "coverage_metric_validation_source.py"
    candidates = [
        Path(explicit) if explicit else None,
        Path(os.environ["COVERAGE_ANNOTATIONS_FILE"]) if os.environ.get("COVERAGE_ANNOTATIONS_FILE") else None,
        declared_path(source, "ANNOTATION_FILE"),
        *sibling_candidates(root, ANNOTATIONS_RELATIVE),
    ]
    for candidate in candidates:
        if candidate is not None and Path(candidate).expanduser().exists():
            return Path(candidate).expanduser().resolve()
    return None


def portable_path(path: Optional[Path], root: Path) -> Optional[str]:
    """Record a path without leaking the machine layout into shipped outputs.

    Anything inside the project root is written relative to it; anything outside
    is reduced to its file name. Keeps the manifest anonymous for submission.
    """
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return resolved.name
    return str(relative) if str(relative) != "." else "."


# ---------------------------------------------------------------------------
# Inputs and normalization
# ---------------------------------------------------------------------------

def _install_numpy_pickle_compat() -> None:
    """Make NumPy-2 pickles readable in environments that still use NumPy 1.x.

    ``foundation.pkl`` may have been created under NumPy 2, whose private
    modules are serialized as ``numpy._core.*``. NumPy 1.x exposes the same
    objects under ``numpy.core.*``. Registering these aliases before unpickling
    avoids changing the project environment or rebuilding the foundation.
    """
    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
    }
    for serialized_name, local_name in aliases.items():
        if serialized_name in sys.modules:
            continue
        try:
            sys.modules[serialized_name] = importlib.import_module(serialized_name)
        except ModuleNotFoundError:
            sys.modules[serialized_name] = importlib.import_module(local_name)


def _pandas_string_dtype_compat(storage=None, na_value=pd.NA):
    """Reconstruct pandas StringDtype pickled by pandas >=2.3 on pandas 2.2.

    pandas 2.3 serializes ``StringDtype`` with two arguments
    ``(storage, na_value)``. pandas 2.2 accepts only ``storage`` and represents
    NumPy/NaN string semantics through the legacy ``pyarrow_numpy`` storage
    label. This factory preserves those semantics when possible.
    """
    # First try the native constructor. This is the correct path on pandas 2.3+.
    try:
        return pd.StringDtype(storage=storage, na_value=na_value)
    except TypeError:
        pass

    # pandas 2.2 uses pyarrow_numpy to encode pyarrow storage with np.nan
    # missing-value semantics.
    is_nan_semantics = False
    try:
        is_nan_semantics = bool(pd.isna(na_value)) and na_value is not pd.NA
    except Exception:
        is_nan_semantics = False

    legacy_storage = storage
    if storage == "pyarrow" and is_nan_semantics:
        legacy_storage = "pyarrow_numpy"

    try:
        return pd.StringDtype(storage=legacy_storage)
    except (TypeError, ValueError):
        # Last-resort fallback for older pandas releases that do not recognize
        # pyarrow_numpy. The user's environment still needs pyarrow to rebuild
        # the ArrowStringArray stored in the pickle.
        return pd.StringDtype(storage=storage)


class _FoundationCompatUnpickler(pickle.Unpickler):
    """Compatibility loader for NumPy-2 and pandas-2.3 foundation pickles."""

    def find_class(self, module: str, name: str):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = module.replace("numpy._core", "numpy.core", 1)
        if module == "pandas" and name == "StringDtype":
            return _pandas_string_dtype_compat
        return super().find_class(module, name)


def load_coverage_geometry(root: Path):
    sys.path.insert(0, str(root / "src"))
    from coverage_geometry import Foundation  # type: ignore

    output_root = Path(os.environ.get("COVERAGE_OUTPUT_DIR") or root / "outputs_corrected")
    foundation_path = output_root / "cache" / "foundation.pkl"
    if not foundation_path.exists():
        raise FileNotFoundError(
            f"Missing {foundation_path}. Run src/build_foundation.py first."
        )

    _install_numpy_pickle_compat()
    try:
        with foundation_path.open("rb") as fh:
            data = _FoundationCompatUnpickler(fh).load()
    except (TypeError, ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "Could not read outputs_corrected/cache/foundation.pkl because it "
            "was created with incompatible NumPy/pandas extension dtypes. "
            "The compatibility loader handled NumPy private-module aliases and "
            "pandas StringDtype(storage, na_value), but another dependency "
            "mismatch remains. Run with pandas>=2.3,<2.4 or rebuild the "
            "foundation in the current environment. Original error: " + repr(exc)
        ) from exc

    return Foundation(data)


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalize_gate(value: Any) -> int:
    text = text_value(value).lower()
    if text in {"true", "manual", "restricted", "2"}:
        return 2
    if text in {"auto", "automatic", "1"}:
        return 1
    return 0


def split_list_field(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [text_value(x) for x in value if text_value(x)]
    text = text_value(value)
    if not text:
        return []
    return [part.strip() for part in re.split(r"[;|]", text) if part.strip()]


def load_annotations(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "papers" in data and isinstance(data["papers"], list):
            data = data["papers"]
        else:
            data = [data]
    result: Dict[str, Dict[str, Any]] = {}
    for record in data:
        if not isinstance(record, dict) or not record.get("paper_id"):
            continue
        result[str(record["paper_id"])] = record
    return result


def load_cutoff_overrides(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    frame = pd.read_csv(path)
    required = {"paper_id", "cutoff_date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cutoff file is missing columns: {sorted(missing)}")
    out: Dict[str, Dict[str, str]] = {}
    for row in frame.to_dict("records"):
        pid = text_value(row.get("paper_id"))
        date = text_value(row.get("cutoff_date"))
        if not pid or not date:
            continue
        out[pid] = {
            "cutoff_date": date,
            "cutoff_source": text_value(row.get("cutoff_source")) or "manual_override",
            "cutoff_note": text_value(row.get("cutoff_note")),
        }
    return out


@dataclass
class VariantMetadata:
    creation_date: np.ndarray
    gate_rank: np.ndarray
    size_rank: np.ndarray
    size_known: np.ndarray
    role: np.ndarray
    family: np.ndarray


def load_variant_metadata(F, lineage_file: Path) -> VariantMetadata:
    columns = [
        "variant_id",
        "created_at",
        "gated",
        "is_multimodal_leak",
    ]
    raw = pd.read_parquet(lineage_file, columns=columns)
    if "is_multimodal_leak" in raw:
        raw = raw[~raw["is_multimodal_leak"].fillna(False)].copy()
    raw["created_at_parsed"] = pd.to_datetime(raw["created_at"], errors="coerce", utc=True)
    raw["gate_rank"] = raw["gated"].map(normalize_gate).astype(int)

    order = F.variant["variant_id"].astype(str).to_numpy()
    created = raw.groupby("variant_id")["created_at_parsed"].min().reindex(order)
    # A variant is considered accessible if any observable repository carrying it
    # has the less restrictive gate level. This matches the current feasibility code.
    gate = raw.groupby("variant_id")["gate_rank"].min().reindex(order).fillna(0)

    size_text = F.variant["size_bucket"].astype(str)
    size_rank = size_text.map(SIZE_RANK).to_numpy(float)
    size_known = np.isfinite(size_rank)
    role = F.variant["model_role"].fillna("MISSING").astype(str).to_numpy()
    family = F.variant["family_id"].fillna("MISSING").astype(str).to_numpy()
    return VariantMetadata(
        creation_date=created.to_numpy(dtype="datetime64[ns]"),
        gate_rank=gate.to_numpy(int),
        size_rank=size_rank,
        size_known=size_known,
        role=role,
        family=family,
    )


# ---------------------------------------------------------------------------
# Shared metric functions
# ---------------------------------------------------------------------------

def hardened_indices(panel: Mapping[str, Any]) -> List[int]:
    result: List[int] = []
    for entry in panel["entries"]:
        candidate = np.asarray(entry["candidate_idx"], dtype=int)
        posterior = np.asarray(entry["posterior"], dtype=float)
        if len(candidate):
            result.append(int(candidate[int(np.argmax(posterior))]))
    return result


def target_adoption(F, support_mask: np.ndarray) -> np.ndarray:
    support = support_mask.astype(float)
    n_support = int(support.sum())
    if n_support == 0:
        return np.zeros(F.n)
    adoption = F.DOWNLOAD_PROB * support
    if adoption.sum() > 0:
        adoption = adoption / adoption.sum()
    else:
        adoption = support / n_support
    uniform = support / n_support
    target = (1.0 - F.ALPHA) * adoption + F.ALPHA * uniform
    return target / target.sum()


def target_breadth(F, support_mask: np.ndarray) -> np.ndarray:
    target = F.PI_B * support_mask.astype(float)
    if target.sum() <= 0:
        return np.zeros(F.n)
    return target / target.sum()


def population_views(F, pi_b: np.ndarray) -> Dict[str, np.ndarray]:
    views: Dict[str, np.ndarray] = {}
    for name, codes in F.discrepancy["codes"].items():
        views[name] = np.bincount(
            codes, weights=pi_b, minlength=int(np.max(codes)) + 1
        )
    return views


def structural_linf(F, sigma: np.ndarray, views: Mapping[str, np.ndarray]) -> Tuple[float, str, str, float]:
    worst = (-1.0, "", "", 0.0)
    for name, codes in F.discrepancy["codes"].items():
        pop = np.asarray(views[name], float)
        observed = np.bincount(codes, weights=sigma, minlength=len(pop))
        delta = observed - pop
        j = int(np.argmax(np.abs(delta)))
        value = float(abs(delta[j]))
        labels = F.discrepancy["labels"][name]
        label = str(labels[j]) if j < len(labels) else str(j)
        if value > worst[0]:
            worst = (value, name, label, float(delta[j]))
    return worst


def breadth_soft(F, entries: List[Dict[str, Any]], pi_b: np.ndarray, support_idx: np.ndarray) -> float:
    if not entries or len(support_idx) == 0:
        return float("nan")
    # Foundation.breadth evaluates all panel entries while integrating only over
    # the declared target support, so out-of-scope models do not receive direct
    # target mass but may still provide nearby lineage representation.
    return float(F.breadth(entries, target=pi_b, support_subset=support_idx))


def conditioned_coverage(F, entries: List[Dict[str, Any]], pi_a: np.ndarray, support_mask: np.ndarray) -> Dict[str, float]:
    inclusion = F.inclusion_probability(entries)
    support = support_mask.astype(bool)
    support_n = int(support.sum())
    c0 = float(inclusion[support].sum() / support_n) if support_n else float("nan")
    c1 = float(np.dot(inclusion, pi_a)) if pi_a.sum() else float("nan")
    return {
        "coverage_q0": c0,
        "coverage_q1": c1,
        "coverage_delta_1_0": c1 - c0 if np.isfinite(c0) and np.isfinite(c1) else float("nan"),
    }


def score_entries(F, entries: List[Dict[str, Any]], support_mask: np.ndarray) -> Dict[str, Any]:
    support_idx = np.where(support_mask)[0]
    if not entries or len(support_idx) == 0:
        return {
            "adoption_gap": float("nan"),
            "breadth": float("nan"),
            "structural_linf": float("nan"),
            "structural_partition": "",
            "structural_cell": "",
            "structural_signed_error": float("nan"),
            "coverage_q0": float("nan"),
            "coverage_q1": float("nan"),
            "coverage_delta_1_0": float("nan"),
            "panel_mass_in_population": float("nan"),
        }
    pi_a = target_adoption(F, support_mask)
    pi_b = target_breadth(F, support_mask)
    sigma = F.sigma(entries)
    linf, part, cell, signed = structural_linf(F, sigma, population_views(F, pi_b))
    values: Dict[str, Any] = {
        "adoption_gap": float(F.w1(pi_a, sigma)),
        "breadth": breadth_soft(F, entries, pi_b, support_idx),
        "structural_linf": linf,
        "structural_partition": part,
        "structural_cell": cell,
        "structural_signed_error": signed,
        "panel_mass_in_population": float(sigma[support_mask].sum()),
    }
    values.update(conditioned_coverage(F, entries, pi_a, support_mask))
    return values


def exact_entries(F, indices: Sequence[int]) -> List[Dict[str, Any]]:
    return F.exact_entries([int(i) for i in indices])


def greedy_adoption_support(F, n: int, target: np.ndarray, candidates: np.ndarray) -> List[int]:
    if n <= 0 or len(candidates) == 0:
        return []
    n_select = min(n, len(candidates))
    target_edge = F.branch_mass(target)
    counts = np.zeros(F.n_edges)
    available = np.zeros(F.n, dtype=bool)
    available[candidates] = True
    selected: List[int] = []
    final_weight = 1.0 / n
    for _ in range(n_select):
        change = np.zeros(F.n)
        for depth in range(4):
            edge = F.paths[:, depth]
            before = np.abs(target_edge[edge] - counts[edge])
            after = np.abs(target_edge[edge] - counts[edge] - final_weight)
            change += F.lengths[edge] * (after - before)
        change[~available] = np.inf
        chosen = int(np.argmin(change))
        selected.append(chosen)
        available[chosen] = False
        counts[F.paths[chosen]] += final_weight
    return selected


def greedy_breadth_support(
    F,
    n: int,
    K: np.ndarray,
    target_pi_b: np.ndarray,
    population_support: np.ndarray,
    candidate_support: np.ndarray,
) -> List[int]:
    if n <= 0 or len(candidate_support) == 0 or len(population_support) == 0:
        return []
    n_select = min(n, len(candidate_support))
    rows = np.asarray(population_support, int)
    candidates = np.asarray(candidate_support, int)
    weights = target_pi_b[rows].astype(np.float32)
    weights = weights / weights.sum()
    Kc = K[np.ix_(rows, candidates)]
    best = np.zeros(len(rows), dtype=np.float32)
    available = np.ones(len(candidates), dtype=bool)
    selected_local: List[int] = []
    for _ in range(n_select):
        gains = ((np.maximum(Kc, best[:, None]) - best[:, None]) * weights[:, None]).sum(axis=0)
        gains[~available] = -np.inf
        local = int(np.argmax(gains))
        selected_local.append(local)
        available[local] = False
        best = np.maximum(best, Kc[:, local])
    return [int(candidates[j]) for j in selected_local]


def sample_reference(
    F,
    n: int,
    sampler: str,
    support: np.ndarray,
    pi_a: np.ndarray,
    family_codes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    replace = len(support) < n
    if sampler == "uniform":
        return rng.choice(support, n, replace=replace)
    pop = pi_a[support].copy()
    pop = pop / pop.sum()
    if sampler == "adoption_weighted":
        return rng.choice(support, n, replace=replace, p=pop)
    if sampler != "family_stratified":
        raise ValueError(f"Unknown sampler: {sampler}")

    families = family_codes[support]
    unique = np.unique(families)
    family_mass = np.array([pop[families == fam].sum() for fam in unique], float)
    family_mass = family_mass / family_mass.sum()
    k = min(n, len(unique))
    chosen_families = rng.choice(len(unique), k, replace=False, p=family_mass)
    selected = np.empty(n, dtype=int)
    for position, family_position in enumerate(chosen_families):
        local = support[families == unique[family_position]]
        local_prob = pi_a[local]
        local_prob = local_prob / local_prob.sum()
        selected[position] = int(rng.choice(local, p=local_prob))
    if n > k:
        selected[k:] = rng.choice(support, n - k, replace=True, p=pop)
    return selected


# ---------------------------------------------------------------------------
# C.2.1: eligibility and feasibility
# ---------------------------------------------------------------------------

def resolve_cutoff(
    pid: str,
    panel: Mapping[str, Any],
    metadata_row: Mapping[str, Any],
    variant_meta: VariantMetadata,
    overrides: Mapping[str, Mapping[str, str]],
    F,
) -> Dict[str, Any]:
    own = hardened_indices(panel)
    known_pairs = [
        (idx, variant_meta.creation_date[idx])
        for idx in own
        if not np.isnat(variant_meta.creation_date[idx])
    ]
    if pid in overrides:
        cutoff = pd.to_datetime(overrides[pid]["cutoff_date"], errors="raise", utc=True)
        source = overrides[pid].get("cutoff_source", "manual_override")
        note = overrides[pid].get("cutoff_note", "")
        anchor_idx = None
    elif known_pairs:
        anchor_idx, raw_date = max(known_pairs, key=lambda pair: pair[1])
        cutoff = pd.Timestamp(raw_date).tz_localize("UTC")
        source = "newest_evaluated_model_release"
        note = ""
    else:
        year = pd.to_numeric(metadata_row.get("year"), errors="coerce")
        if np.isfinite(year):
            cutoff = pd.Timestamp(year=int(year), month=12, day=31, tz="UTC")
            source = "publication_year_upper_bound"
            note = "No evaluated model had a known creation date."
            anchor_idx = None
        else:
            raise ValueError(f"Cannot determine cutoff for {pid}: no dates and no year")

    return {
        "cutoff": cutoff,
        "cutoff_date": cutoff.date().isoformat(),
        "cutoff_source": source,
        "cutoff_note": note,
        "anchor_idx": anchor_idx,
        "own": own,
        "own_known_date_count": len(known_pairs),
        "own_missing_date_count": len(own) - len(known_pairs),
    }


def build_c21(
    F,
    variant_meta: VariantMetadata,
    annotations: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Mapping[str, str]],
    output_dir: Path,
    shared: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    metadata = F.data["paper_metadata"].set_index("paper_id", drop=False)
    rows: List[Dict[str, Any]] = []
    contexts: Dict[str, Dict[str, Any]] = {}

    for pid, panel in F.panels.items():
        if not panel["entries"]:
            continue
        meta = metadata.loc[pid].to_dict() if pid in metadata.index else {"paper_id": pid}
        cutoff_info = resolve_cutoff(pid, panel, meta, variant_meta, overrides, F)
        cutoff = cutoff_info["cutoff"]
        dates_pd = pd.to_datetime(variant_meta.creation_date, utc=True)
        eligible = (~pd.isna(dates_pd)) & (dates_pd <= cutoff)
        eligible = np.asarray(eligible, dtype=bool)
        if shared and pid in shared and pid not in overrides:
            # One cutoff and one eligible set for every analysis: the ones F_P uses.
            cutoff = pd.Timestamp(shared[pid]["cutoff"])
            cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
            eligible = np.asarray(shared[pid]["eligible"], dtype=bool).copy()
            cutoff_info = {**cutoff_info, "cutoff": cutoff, "cutoff_date": cutoff.date().isoformat(),
                           "cutoff_note": "newest evaluated model, dated by the repository the paper names (F_P facts)"}
        own = cutoff_info["own"]

        own_gate = variant_meta.gate_rank[own] if own else np.array([], int)
        gate_max = int(own_gate.max()) if len(own_gate) else 0

        own_roles_all = [text_value(x) for x in variant_meta.role[own]]
        own_roles = sorted({r for r in own_roles_all if r.lower() not in TEXT_MISSING})
        role_constraint_applied = bool(own_roles)

        own_size_values = variant_meta.size_rank[own] if own else np.array([], float)
        known_own_sizes = own_size_values[np.isfinite(own_size_values)]
        size_max = float(known_own_sizes.max()) if len(known_own_sizes) else np.nan
        size_constraint_applied = bool(np.isfinite(size_max))

        feasible = eligible & (variant_meta.gate_rank <= gate_max)
        if role_constraint_applied:
            feasible &= np.isin(variant_meta.role, own_roles)
        if size_constraint_applied:
            feasible &= variant_meta.size_known & (variant_meta.size_rank <= size_max)

        eligible_idx = np.where(eligible)[0]
        feasible_idx = np.where(feasible)[0]
        own_in_feasible = sum(bool(feasible[idx]) for idx in own)
        anchor_idx = cutoff_info["anchor_idx"]
        anchor_variant = (
            str(F.variant.iloc[int(anchor_idx)]["variant_id"])
            if anchor_idx is not None
            else ""
        )

        annotation = annotations.get(pid, {})
        own_size_labels = sorted(
            {
                SIZE_ORDER[int(rank)]
                for rank in known_own_sizes
                if 0 <= int(rank) < len(SIZE_ORDER)
            },
            key=lambda x: SIZE_RANK[x],
        )
        unavailable = [
            "context_length",
            "language_requirements",
            "domain_requirements",
            "numerical_precision",
            "measured_throughput",
            "hardware_requirements",
            "active_parameter_count",
            "total_inference_workload",
        ]
        row = {
            "paper_id": pid,
            "title": text_value(meta.get("title") or annotation.get("title")),
            "venue": text_value(meta.get("venue") or annotation.get("venue")),
            "year": meta.get("year", annotation.get("year")),
            "n_input": panel.get("n_input", np.nan),
            "n_resolved": panel.get("n_resolved", len(panel["entries"])),
            "cutoff_date": cutoff_info["cutoff_date"],
            "cutoff_source": cutoff_info["cutoff_source"],
            "cutoff_note": cutoff_info["cutoff_note"],
            "cutoff_anchor_variant": anchor_variant,
            "own_known_date_count": cutoff_info["own_known_date_count"],
            "own_missing_date_count": cutoff_info["own_missing_date_count"],
            "eligible_set_size": len(eligible_idx),
            "eligible_fraction": len(eligible_idx) / F.n,
            "variants_missing_creation_date": int(pd.isna(dates_pd).sum()),
            "gate_max_rank": gate_max,
            "gate_max_label": GATE_LABEL[gate_max],
            "own_roles": ";".join(own_roles),
            "role_constraint_applied": role_constraint_applied,
            "own_size_buckets": ";".join(own_size_labels),
            "max_size_bucket": SIZE_ORDER[int(size_max)] if np.isfinite(size_max) else "",
            "size_constraint_applied": size_constraint_applied,
            "own_unknown_size_count": int((~np.isfinite(own_size_values)).sum()),
            "feasible_set_size": len(feasible_idx),
            "feasible_fraction_of_eligible": len(feasible_idx) / max(len(eligible_idx), 1),
            "budget_feasible": len(feasible_idx) >= int(panel["n_resolved"]),
            "own_models_in_feasible_count": own_in_feasible,
            "own_models_outside_feasible_count": len(own) - own_in_feasible,
            "constraints_applied": "availability;access_gate;functional_role;coarse_size_bucket",
            "constraints_unavailable": ";".join(unavailable),
            "geometry_scope": "global_frozen_lineage_tree",
            "target_scope": "paper_specific_eligible_support",
        }
        rows.append(row)
        contexts[pid] = {
            "eligible_mask": eligible,
            "feasible_mask": feasible,
            "eligible_idx": eligible_idx,
            "feasible_idx": feasible_idx,
            "cutoff": cutoff,
            "row": row,
        }

    per_paper = pd.DataFrame(rows).sort_values(["year", "paper_id"]).reset_index(drop=True)
    per_paper.to_csv(output_dir / "appendix_c21_eligibility_feasibility_per_paper.csv", index=False)

    summary_rows = [
        {"statistic": "scoreable_papers", "value": len(per_paper)},
        {"statistic": "median_panel_size", "value": per_paper["n_resolved"].median()},
        {"statistic": "median_eligible_set_size", "value": per_paper["eligible_set_size"].median()},
        {"statistic": "eligible_set_size_min", "value": per_paper["eligible_set_size"].min()},
        {"statistic": "eligible_set_size_max", "value": per_paper["eligible_set_size"].max()},
        {"statistic": "median_feasible_set_size", "value": per_paper["feasible_set_size"].median()},
        {"statistic": "feasible_set_size_min", "value": per_paper["feasible_set_size"].min()},
        {"statistic": "feasible_set_size_max", "value": per_paper["feasible_set_size"].max()},
        {"statistic": "papers_with_same_size_feasible_pool", "value": int(per_paper["budget_feasible"].sum())},
        {"statistic": "papers_using_manual_cutoff", "value": int((per_paper["cutoff_source"] == "manual_override").sum())},
        {"statistic": "papers_using_newest_model_cutoff", "value": int((per_paper["cutoff_source"] == "newest_evaluated_model_release").sum())},
        {"statistic": "papers_using_year_fallback", "value": int((per_paper["cutoff_source"] == "publication_year_upper_bound").sum())},
        {"statistic": "papers_with_role_constraint", "value": int(per_paper["role_constraint_applied"].sum())},
        {"statistic": "papers_with_size_constraint", "value": int(per_paper["size_constraint_applied"].sum())},
        {"statistic": "papers_whose_own_hardened_models_all_pass_feasibility", "value": int((per_paper["own_models_outside_feasible_count"] == 0).sum())},
    ]
    pd.DataFrame(summary_rows).to_csv(output_dir / "appendix_c21_summary.csv", index=False)

    constraint_rows = [
        {
            "constraint": "historical_availability",
            "implemented": True,
            "source_field": "created_at",
            "rule": "earliest repository creation date for variant <= paper cutoff",
            "missing_data_rule": "exclude candidate with unknown creation date",
        },
        {
            "constraint": "access_gate",
            "implemented": True,
            "source_field": "gated",
            "rule": "candidate gate rank <= maximum gate rank accepted by observed panel",
            "missing_data_rule": "missing gating metadata treated as not gated",
        },
        {
            "constraint": "functional_role",
            "implemented": True,
            "source_field": "model_role",
            "rule": "candidate role must be among roles evaluated by the paper",
            "missing_data_rule": "constraint omitted if observed roles are all unknown",
        },
        {
            "constraint": "coarse_size_envelope",
            "implemented": True,
            "source_field": "size_bucket",
            "rule": "known candidate size bucket <= largest known observed size bucket",
            "missing_data_rule": "constraint omitted if every observed size is unknown; unknown candidate size excluded when applied",
        },
        {
            "constraint": "modality",
            "implemented": True,
            "source_field": "primary population filter",
            "rule": "all candidates already belong to the text-generative population",
            "missing_data_rule": "not applicable",
        },
    ]
    for constraint in [
        "context_length",
        "language_requirements",
        "domain_requirements",
        "numerical_precision",
        "measured_throughput",
        "hardware_requirements",
        "active_parameter_count",
        "total_inference_workload",
    ]:
        constraint_rows.append(
            {
                "constraint": constraint,
                "implemented": False,
                "source_field": "unavailable_or_insufficiently_complete",
                "rule": "not enforced",
                "missing_data_rule": "never imputed; not used to support practical-attainability claims",
            }
        )
    pd.DataFrame(constraint_rows).to_csv(output_dir / "appendix_c21_constraint_definitions.csv", index=False)

    # A template makes it easy to replace the conservative newest-model cutoff
    # with an explicitly annotated experiment/model-selection date.
    template = per_paper[["paper_id", "cutoff_date", "cutoff_source"]].copy()
    template["cutoff_note"] = ""
    template.to_csv(output_dir / "paper_cutoffs_template.csv", index=False)
    return per_paper, contexts


# ---------------------------------------------------------------------------
# C.4: complete audit results and matched alternatives
# ---------------------------------------------------------------------------

def build_c4(
    F,
    c21: pd.DataFrame,
    contexts: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    n_null: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metadata = F.data["paper_metadata"].set_index("paper_id", drop=False)
    family_codes = pd.factorize(F.variant["family_id"])[0]
    K = F.kernel()
    samplers = ["adoption_weighted", "family_stratified", "uniform"]

    rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []

    for paper_position, (pid, panel) in enumerate(F.panels.items()):
        if not panel["entries"]:
            continue
        context = contexts[pid]
        eligible_mask = np.asarray(context["eligible_mask"], bool)
        feasible_mask = np.asarray(context["feasible_mask"], bool)
        eligible_idx = np.where(eligible_mask)[0]
        feasible_idx = np.where(feasible_mask)[0]
        n = int(panel["n_resolved"])
        pi_a = target_adoption(F, eligible_mask)
        pi_b = target_breadth(F, eligible_mask)
        observed = score_entries(F, panel["entries"], eligible_mask)

        hist_adopt_idx = greedy_adoption_support(F, n, pi_a, eligible_idx)
        hist_breadth_idx = greedy_breadth_support(F, n, K, pi_b, eligible_idx, eligible_idx)
        hist_adopt = score_entries(F, exact_entries(F, hist_adopt_idx), eligible_mask)
        hist_breadth = score_entries(F, exact_entries(F, hist_breadth_idx), eligible_mask)

        feasible_ok = len(feasible_idx) >= n
        if feasible_ok:
            feas_adopt_idx = greedy_adoption_support(F, n, pi_a, feasible_idx)
            feas_breadth_idx = greedy_breadth_support(F, n, K, pi_b, eligible_idx, feasible_idx)
            feas_adopt = score_entries(F, exact_entries(F, feas_adopt_idx), eligible_mask)
            feas_breadth = score_entries(F, exact_entries(F, feas_breadth_idx), eligible_mask)
        else:
            feas_adopt_idx, feas_breadth_idx = [], []
            feas_adopt = {key: np.nan for key in observed}
            feas_breadth = {key: np.nan for key in observed}

        row: Dict[str, Any] = {
            "paper_id": pid,
            "title": text_value(metadata.loc[pid, "title"]) if pid in metadata.index else "",
            "venue": text_value(metadata.loc[pid, "venue"]) if pid in metadata.index else "",
            "year": metadata.loc[pid, "year"] if pid in metadata.index else np.nan,
            "n": n,
            "cutoff_date": context["row"]["cutoff_date"],
            "cutoff_source": context["row"]["cutoff_source"],
            "eligible_set_size": len(eligible_idx),
            "feasible_set_size": len(feasible_idx),
            "budget_feasible": feasible_ok,
        }
        for prefix, scores in [
            ("observed", observed),
            ("historical_adoption_greedy", hist_adopt),
            ("historical_breadth_greedy", hist_breadth),
            ("feasible_adoption_greedy", feas_adopt),
            ("feasible_breadth_greedy", feas_breadth),
        ]:
            for key, value in scores.items():
                row[f"{prefix}_{key}"] = value

        obs_ga = float(observed["adoption_gap"])
        hist_ga = float(hist_adopt["adoption_gap"])
        row["historical_eff_ratio"] = hist_ga / obs_ga if obs_ga > 0 else np.nan
        row["historical_reduction_pct"] = 100.0 * (1.0 - row["historical_eff_ratio"])
        row["historical_adoption_improved"] = hist_ga < obs_ga - 1e-12
        row["historical_breadth_improved"] = hist_breadth["breadth"] > observed["breadth"] + 1e-12
        row["historical_structural_best"] = min(
            hist_adopt["structural_linf"], hist_breadth["structural_linf"]
        )
        row["historical_structural_improved"] = row["historical_structural_best"] < observed["structural_linf"] - 1e-12

        if feasible_ok:
            feas_ga = float(feas_adopt["adoption_gap"])
            row["feasible_eff_ratio"] = feas_ga / obs_ga if obs_ga > 0 else np.nan
            row["feasible_reduction_pct"] = 100.0 * (1.0 - row["feasible_eff_ratio"])
            row["feasible_adoption_improved"] = feas_ga < obs_ga - 1e-12
            row["feasible_breadth_improved"] = feas_breadth["breadth"] > observed["breadth"] + 1e-12
            row["feasible_structural_best"] = min(
                feas_adopt["structural_linf"], feas_breadth["structural_linf"]
            )
            row["feasible_structural_improved"] = row["feasible_structural_best"] < observed["structural_linf"] - 1e-12
        else:
            for key in [
                "feasible_eff_ratio",
                "feasible_reduction_pct",
                "feasible_adoption_improved",
                "feasible_breadth_improved",
                "feasible_structural_best",
                "feasible_structural_improved",
            ]:
                row[key] = np.nan

        # Same-size null calibration on the historical eligible support.
        null_values: Dict[str, Dict[str, List[float]]] = {
            sampler: {"adoption": [], "breadth": [], "structural": []}
            for sampler in samplers
        }
        for sampler in samplers:
            rng = np.random.default_rng(seed + 1009 * paper_position + len(sampler))
            for _ in range(n_null):
                selection = sample_reference(
                    F, n, sampler, eligible_idx, pi_a, family_codes, rng
                )
                scores = score_entries(F, exact_entries(F, selection), eligible_mask)
                null_values[sampler]["adoption"].append(scores["adoption_gap"])
                null_values[sampler]["breadth"].append(scores["breadth"])
                null_values[sampler]["structural"].append(scores["structural_linf"])
            row[f"{sampler}_adoption_percentile"] = float(
                np.mean(np.asarray(null_values[sampler]["adoption"]) >= observed["adoption_gap"])
            )
            row[f"{sampler}_breadth_percentile"] = float(
                np.mean(np.asarray(null_values[sampler]["breadth"]) <= observed["breadth"])
            )
            row[f"{sampler}_structural_percentile"] = float(
                np.mean(np.asarray(null_values[sampler]["structural"]) >= observed["structural_linf"])
            )

        rows.append(row)

        for panel_type, indices in [
            ("historical_adoption_greedy", hist_adopt_idx),
            ("historical_breadth_greedy", hist_breadth_idx),
            ("feasible_adoption_greedy", feas_adopt_idx),
            ("feasible_breadth_greedy", feas_breadth_idx),
        ]:
            for rank, idx in enumerate(indices, start=1):
                selected_rows.append(
                    {
                        "paper_id": pid,
                        "panel_type": panel_type,
                        "selection_rank": rank,
                        "variant_idx": idx,
                        "variant_id": str(F.variant.iloc[idx]["variant_id"]),
                        "family_id": str(F.variant.iloc[idx]["family_id"]),
                        "series_id": str(F.variant.iloc[idx]["series_id"]),
                        "base_model_id": str(F.variant.iloc[idx]["base_model_id"]),
                        "model_role": str(F.variant.iloc[idx]["model_role"]),
                        "size_bucket": str(F.variant.iloc[idx]["size_bucket"]),
                    }
                )

    per_paper = pd.DataFrame(rows).sort_values(["year", "paper_id"]).reset_index(drop=True)
    per_paper.to_csv(output_dir / "appendix_c4_complete_audit_per_paper.csv", index=False)
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(output_dir / "appendix_c4_greedy_panel_members.csv", index=False)

    feasible = per_paper[per_paper["budget_feasible"]].copy()
    summary: List[Dict[str, Any]] = [
        {"statistic": "scoreable_papers", "value": len(per_paper)},
        {"statistic": "median_panel_size", "value": per_paper["n"].median()},
        {"statistic": "median_eligible_set_size", "value": per_paper["eligible_set_size"].median()},
        {"statistic": "median_feasible_set_size", "value": per_paper["feasible_set_size"].median()},
        {"statistic": "feasible_set_size_min", "value": per_paper["feasible_set_size"].min()},
        {"statistic": "feasible_set_size_max", "value": per_paper["feasible_set_size"].max()},
        {"statistic": "papers_with_feasible_same_size_pool", "value": len(feasible)},
        {"statistic": "historical_adoption_improved_count", "value": int(per_paper["historical_adoption_improved"].sum())},
        {"statistic": "historical_median_eff_ratio", "value": per_paper["historical_eff_ratio"].median()},
        {"statistic": "historical_median_reduction_pct", "value": per_paper["historical_reduction_pct"].median()},
        {"statistic": "historical_breadth_improved_count", "value": int(per_paper["historical_breadth_improved"].sum())},
        {"statistic": "historical_structural_improved_count", "value": int(per_paper["historical_structural_improved"].sum())},
        {"statistic": "feasible_adoption_improved_count", "value": int(feasible["feasible_adoption_improved"].sum())},
        {"statistic": "feasible_median_eff_ratio", "value": feasible["feasible_eff_ratio"].median()},
        {"statistic": "feasible_median_reduction_pct", "value": feasible["feasible_reduction_pct"].median()},
        {"statistic": "feasible_breadth_improved_count", "value": int(feasible["feasible_breadth_improved"].sum())},
        {"statistic": "feasible_structural_improved_count", "value": int(feasible["feasible_structural_improved"].sum())},
        {"statistic": "n_null_panels_per_sampler_per_paper", "value": n_null},
    ]
    for sampler in samplers:
        for coordinate in ["adoption", "breadth", "structural"]:
            summary.append(
                {
                    "statistic": f"median_{sampler}_{coordinate}_percentile",
                    "value": per_paper[f"{sampler}_{coordinate}_percentile"].median(),
                }
            )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(output_dir / "appendix_c4_summary.csv", index=False)
    return per_paper, summary_frame


# ---------------------------------------------------------------------------
# C.5: custom and claim-conditioned populations
# ---------------------------------------------------------------------------

def default_population_masks(F) -> Dict[str, Dict[str, Any]]:
    role = F.variant["model_role"].fillna("MISSING").astype(str).str.lower()
    size = F.variant["size_bucket"].map(SIZE_RANK)
    definitions: Dict[str, Dict[str, Any]] = {
        "all_textual": {
            "description": "All variants in the primary text-generative ecosystem",
            "mask": np.ones(F.n, dtype=bool),
            "dimension": "none",
        },
        "assistant_models": {
            "description": "Instruction-, chat-, or reasoning-oriented variants",
            "mask": role.isin({"instruct", "instruction", "chat", "reasoning"}).to_numpy(),
            "dimension": "model_role",
        },
        "reasoning_models": {
            "description": "Variants classified with the reasoning functional role",
            "mask": role.eq("reasoning").to_numpy(),
            "dimension": "model_role",
        },
        "code_models": {
            "description": "Variants classified with the code functional role",
            "mask": role.eq("code").to_numpy(),
            "dimension": "model_role",
        },
        "base_models": {
            "description": "Untuned base-model variants",
            "mask": role.eq("base").to_numpy(),
            "dimension": "model_role",
        },
        "small_models_le_13b": {
            "description": "Variants in known size buckets up to and including 13B",
            "mask": size.le(SIZE_RANK["7B-13B"]).fillna(False).to_numpy(),
            "dimension": "size_bucket",
        },
        "large_models_ge_34b": {
            "description": "Variants in known size buckets of 34B or larger",
            "mask": size.ge(SIZE_RANK["34B-70B"]).fillna(False).to_numpy(),
            "dimension": "size_bucket",
        },
    }
    return definitions


def paper_scope_text(annotation: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    fields = [
        annotation.get("benchmark_type"),
        annotation.get("claim_scope"),
        annotation.get("title"),
        metadata.get("title"),
        annotation.get("notes"),
    ]
    return " ".join(text_value(x).lower() for x in fields if text_value(x))


def choose_claim_population(
    annotation: Mapping[str, Any], metadata: Mapping[str, Any]
) -> Tuple[str, str, str]:
    """Return population, transparent rule, and unsupported claim dimensions."""
    text = paper_scope_text(annotation, metadata)
    if re.search(r"\b(code|coding|programming|software|code generation)\b", text):
        return "code_models", "coding_or_programming_scope", ""
    if re.search(r"\b(reasoning|mathematical reasoning|math benchmark|proof)\b", text):
        return "reasoning_models", "reasoning_scope", ""
    if re.search(r"\b(safety|jailbreak|alignment|toxicity|harmful|refusal|judge|preference|chatbot|agent|tool use|rag|retrieval)\b", text):
        return "assistant_models", "assistant_deployment_scope", ""

    unsupported: List[str] = []
    if re.search(r"\b(multilingual|translation|language|cross-lingual|african|arabic|chinese|financial|medical|legal|domain)\b", text):
        unsupported.append("language_or_domain_not_encoded_in_lineage_taxonomy")
    if re.search(r"\b(multimodal|vision|image|audio|video)\b", text):
        unsupported.append("multimodal_scope_outside_primary_textual_population")
    return "all_textual", "no_supported_taxonomy_restriction", ";".join(unsupported)


def build_c5(
    F,
    contexts: Mapping[str, Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    definitions = default_population_masks(F)
    metadata = F.data["paper_metadata"].set_index("paper_id", drop=False)

    definition_rows = []
    for pop_id, info in definitions.items():
        mask = np.asarray(info["mask"], bool)
        definition_rows.append(
            {
                "population_id": pop_id,
                "description": info["description"],
                "conditioning_dimension": info["dimension"],
                "global_support_size": int(mask.sum()),
                "global_support_fraction": float(mask.mean()),
            }
        )
    definitions_frame = pd.DataFrame(definition_rows)
    definitions_frame.to_csv(output_dir / "appendix_c5_population_definitions.csv", index=False)

    score_rows: List[Dict[str, Any]] = []
    chosen_population: Dict[str, Tuple[str, str, str]] = {}
    for pid, panel in F.panels.items():
        if not panel["entries"]:
            continue
        meta = metadata.loc[pid].to_dict() if pid in metadata.index else {"paper_id": pid}
        annotation = annotations.get(pid, {})
        chosen_population[pid] = choose_claim_population(annotation, meta)
        eligible = np.asarray(contexts[pid]["eligible_mask"], bool)
        for pop_id, info in definitions.items():
            support = eligible & np.asarray(info["mask"], bool)
            scores = score_entries(F, panel["entries"], support)
            score_rows.append(
                {
                    "paper_id": pid,
                    "title": text_value(meta.get("title") or annotation.get("title")),
                    "venue": text_value(meta.get("venue") or annotation.get("venue")),
                    "year": meta.get("year", annotation.get("year")),
                    "population_id": pop_id,
                    "population_description": info["description"],
                    "cutoff_date": contexts[pid]["row"]["cutoff_date"],
                    "historically_eligible_population_size": int(support.sum()),
                    **scores,
                }
            )

    scores = pd.DataFrame(score_rows)
    scores["adoption_rank_within_population"] = scores.groupby("population_id")["adoption_gap"].rank(method="average", ascending=True)
    scores.to_csv(output_dir / "appendix_c5_scores_per_paper_population.csv", index=False)

    full = scores[scores["population_id"] == "all_textual"].set_index("paper_id")
    claim_rows: List[Dict[str, Any]] = []
    for pid, (population_id, rule, unsupported) in chosen_population.items():
        selected = scores[(scores["paper_id"] == pid) & (scores["population_id"] == population_id)]
        if selected.empty or pid not in full.index:
            continue
        conditioned = selected.iloc[0]
        baseline = full.loc[pid]
        claim_rows.append(
            {
                "paper_id": pid,
                "title": conditioned["title"],
                "venue": conditioned["venue"],
                "year": conditioned["year"],
                "selected_population_id": population_id,
                "conditioning_rule": rule,
                "unsupported_claim_dimensions": unsupported,
                "conditioned_population_size": conditioned["historically_eligible_population_size"],
                "panel_mass_in_conditioned_population": conditioned["panel_mass_in_population"],
                "full_adoption_gap": baseline["adoption_gap"],
                "conditioned_adoption_gap": conditioned["adoption_gap"],
                "delta_adoption_gap": conditioned["adoption_gap"] - baseline["adoption_gap"],
                "full_breadth": baseline["breadth"],
                "conditioned_breadth": conditioned["breadth"],
                "delta_breadth": conditioned["breadth"] - baseline["breadth"],
                "full_structural_linf": baseline["structural_linf"],
                "conditioned_structural_linf": conditioned["structural_linf"],
                "delta_structural_linf": conditioned["structural_linf"] - baseline["structural_linf"],
                "full_coverage_q0": baseline["coverage_q0"],
                "conditioned_coverage_q0": conditioned["coverage_q0"],
                "full_coverage_q1": baseline["coverage_q1"],
                "conditioned_coverage_q1": conditioned["coverage_q1"],
                "full_adoption_rank": baseline["adoption_rank_within_population"],
                "conditioned_adoption_rank": conditioned["adoption_rank_within_population"],
                "absolute_rank_shift": abs(conditioned["adoption_rank_within_population"] - baseline["adoption_rank_within_population"]),
            }
        )
    claim = pd.DataFrame(claim_rows).sort_values(["selected_population_id", "paper_id"])
    claim.to_csv(output_dir / "appendix_c5_claim_conditioned_per_paper.csv", index=False)

    summary_rows: List[Dict[str, Any]] = []
    full_scores = scores[scores["population_id"] == "all_textual"][["paper_id", "adoption_gap", "adoption_rank_within_population"]].rename(
        columns={"adoption_gap": "full_gap", "adoption_rank_within_population": "full_rank"}
    )
    for pop_id, group in scores.groupby("population_id"):
        merged = group.merge(full_scores, on="paper_id", how="inner")
        valid = merged[["adoption_gap", "full_gap"]].dropna()
        tau = (
            kendalltau(valid["full_gap"], valid["adoption_gap"]).statistic
            if len(valid) >= 3
            else np.nan
        )
        shifts = (merged["adoption_rank_within_population"] - merged["full_rank"]).abs()
        definition = definitions_frame.set_index("population_id").loc[pop_id]
        summary_rows.append(
            {
                "population_id": pop_id,
                "description": definition["description"],
                "global_support_size": definition["global_support_size"],
                "papers_scored": int(group["adoption_gap"].notna().sum()),
                "median_historical_support_size": group["historically_eligible_population_size"].median(),
                "median_panel_mass_in_population": group["panel_mass_in_population"].median(),
                "median_adoption_gap": group["adoption_gap"].median(),
                "median_breadth": group["breadth"].median(),
                "median_structural_linf": group["structural_linf"].median(),
                "median_coverage_q0": group["coverage_q0"].median(),
                "median_coverage_q1": group["coverage_q1"].median(),
                "kendall_tau_adoption_gap_vs_all_textual": tau,
                "median_absolute_adoption_rank_shift": shifts.median(),
                "maximum_absolute_adoption_rank_shift": shifts.max(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "appendix_c5_population_summary.csv", index=False)

    if len(claim):
        claim_summary = pd.DataFrame(
            [
                {"statistic": "papers", "value": len(claim)},
                {"statistic": "papers_with_nondefault_conditioning", "value": int((claim["selected_population_id"] != "all_textual").sum())},
                {"statistic": "papers_with_unsupported_claim_dimensions", "value": int(claim["unsupported_claim_dimensions"].astype(str).ne("").sum())},
                {"statistic": "median_panel_mass_in_conditioned_population", "value": claim["panel_mass_in_conditioned_population"].median()},
                {"statistic": "median_delta_adoption_gap", "value": claim["delta_adoption_gap"].median()},
                {"statistic": "median_delta_breadth", "value": claim["delta_breadth"].median()},
                {"statistic": "median_delta_structural_linf", "value": claim["delta_structural_linf"].median()},
                {"statistic": "median_absolute_rank_shift", "value": claim["absolute_rank_shift"].median()},
                {"statistic": "maximum_absolute_rank_shift", "value": claim["absolute_rank_shift"].max()},
            ]
        )
    else:
        claim_summary = pd.DataFrame(columns=["statistic", "value"])
    claim_summary.to_csv(output_dir / "appendix_c5_claim_conditioned_summary.csv", index=False)
    return scores, claim, summary


# ---------------------------------------------------------------------------
# Consolidated output and CLI
# ---------------------------------------------------------------------------

def write_master_csv(
    c21: pd.DataFrame,
    c4: pd.DataFrame,
    c5_claim: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    master = c21.merge(c4, on="paper_id", how="outer", suffixes=("_c21", "_c4"))
    if len(c5_claim):
        keep = [
            "paper_id",
            "selected_population_id",
            "conditioning_rule",
            "unsupported_claim_dimensions",
            "conditioned_population_size",
            "panel_mass_in_conditioned_population",
            "full_adoption_gap",
            "conditioned_adoption_gap",
            "delta_adoption_gap",
            "full_breadth",
            "conditioned_breadth",
            "delta_breadth",
            "full_structural_linf",
            "conditioned_structural_linf",
            "delta_structural_linf",
            "full_coverage_q0",
            "conditioned_coverage_q0",
            "full_coverage_q1",
            "conditioned_coverage_q1",
            "absolute_rank_shift",
        ]
        master = master.merge(c5_claim[keep], on="paper_id", how="left")
    master.to_csv(output_dir / "appendix_c_master_per_paper.csv", index=False)
    return master


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Coverage project root; default: current directory")
    parser.add_argument("--lineage-file", default=None, help="Override lineage parquet path")
    parser.add_argument("--annotations-file", default=None, help="Override paper annotation JSON path")
    parser.add_argument("--cutoffs-file", default=None, help="Optional CSV with paper_id,cutoff_date,cutoff_source,cutoff_note")
    parser.add_argument("--output-dir", default=os.environ.get("COVERAGE_OUTPUT_DIR", "outputs_corrected") + "/appendix_c", help="Output directory relative to root")
    parser.add_argument("--n-null", type=int, default=DEFAULT_N_NULL, help="Null panels per sampler per paper")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_project_root(Path(args.root))
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    lineage_file = discover_lineage_file(root, args.lineage_file)
    annotations_file = discover_annotations_file(root, args.annotations_file)
    annotations = load_annotations(annotations_file)
    overrides = load_cutoff_overrides(args.cutoffs_file)

    print("coverage root:", root)
    print("lineage file:", lineage_file)
    print("annotations file:", annotations_file or "not found; C.5 uses title/venue metadata only")
    print("output directory:", output_dir)
    print("null panels per sampler per paper:", args.n_null)

    F = load_coverage_geometry(root)
    variant_meta = load_variant_metadata(F, lineage_file)

    shared_path = output_dir.parent / "paper_contexts.pkl"
    shared = None
    if shared_path.exists():
        with shared_path.open("rb") as handle:
            shared = pickle.load(handle)
        print("cutoffs and eligible sets from:", shared_path)
    else:
        print("paper_contexts.pkl not found; cutoffs from variant creation dates")
    c21, contexts = build_c21(F, variant_meta, annotations, overrides, output_dir, shared)
    print("C.2.1 complete:", output_dir / "appendix_c21_eligibility_feasibility_per_paper.csv")

    c4, _ = build_c4(F, c21, contexts, output_dir, args.n_null, args.seed)
    print("C.4 complete:", output_dir / "appendix_c4_complete_audit_per_paper.csv")

    _, c5_claim, _ = build_c5(F, contexts, annotations, output_dir)
    print("C.5 complete:", output_dir / "appendix_c5_claim_conditioned_per_paper.csv")

    master = write_master_csv(c21, c4, c5_claim, output_dir)
    print("master CSV:", output_dir / "appendix_c_master_per_paper.csv")
    print("rows:", len(master), "columns:", len(master.columns))

    # Paths are recorded relative to the project root so the shipped manifest
    # carries no machine, account, or author information.
    manifest = {
        "coverage_root": ".",
        "lineage_file": portable_path(lineage_file, root),
        "annotations_file": portable_path(annotations_file, root),
        "cutoffs_file": portable_path(Path(args.cutoffs_file), root) if args.cutoffs_file else None,
        "shared_cutoffs": portable_path(shared_path, root) if shared is not None else None,
        "output_dir": portable_path(output_dir, root),
        "seed": args.seed,
        "n_null": args.n_null,
        "scoreable_papers": len(c4),
        "outputs": sorted(path.name for path in output_dir.glob("*.csv")),
        "methodological_scope": {
            "geometry": "global frozen lineage tree",
            "historical_targets": "current adoption weights restricted to historically eligible variants",
            "feasibility_constraints": [
                "availability",
                "access gate",
                "functional role",
                "coarse size bucket",
            ],
            "not_enforced_due_to_missing_data": [
                "context length",
                "language/domain requirements",
                "precision",
                "throughput",
                "hardware",
                "active parameter count",
                "total inference workload",
            ],
        },
    }
    (output_dir / "appendix_c_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
