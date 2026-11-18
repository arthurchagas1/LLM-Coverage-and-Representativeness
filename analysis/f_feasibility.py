#!/usr/bin/env python3
"""Rebuild the feasible panel sets F_P and report coverage gains, paper by paper.

Each paper's panel is first checked with the original rule, which marks a paper as
problematic when F_P excludes any of its own models. Metadata are then repaired from
the mention's own repository record and from the sourced corrections file, and F_P is
rebuilt so that one rule applies to both sides of the comparison.

Constraints accumulate along a ladder, and each rung reports its own gains:

  historical     available by the paper's cutoff
  core           + access gate, functional role, and maximum size bucket
  cost           + the alternative keeps the panel's size-by-role composition, slot by slot
  license        + license no more restrictive than the panel's most restrictive model
  context        + context length no shorter than the panel's shortest known context
  adoption_floor + at least min(1000, the panel's least-downloaded model) downloads in 30 days

A rung is applied only where it is verifiable. Otherwise the row says why and the
earlier constraints stand. Cost is measured through parameters, as a proxy: no
monetary or measured inference cost exists for these papers.

Outputs in deliverables/<release>/F_feasibility/ and outputs/<release>/paper_contexts.pkl.
"""

from __future__ import annotations

import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import common
from common import OUTPUT, RELEASE, deliverable, foundation, save_figure

import numpy as np
import pandas as pd

import build_appendix_c as baseline  # noqa: E402

ROOT = common.ROOT
CORRECTIONS = ROOT / "data" / "feasibility_model_corrections.csv"
CONFIG_CACHE = ROOT / "data" / "repo_config_cache.csv"
OUT = deliverable("F_feasibility")
RUNGS = ["historical", "core", "cost", "license", "context", "adoption_floor"]
ADOPTION_FLOOR = 1000.0

PERMISSIVE = {"apache-2.0", "mit", "bsd", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear", "cc0-1.0",
              "cc-by-4.0", "cc-by-3.0", "cc-by-2.0", "isc", "unlicense", "artistic-2.0", "pddl", "wtfpl", "zlib",
              "ms-pl", "ecl-2.0", "postgresql", "bsl-1.0", "afl-3.0", "ncsa"}
COPYLEFT = {"gpl", "gpl-2.0", "gpl-3.0", "lgpl", "lgpl-2.1", "lgpl-3.0", "lgpl-lr", "agpl-3.0", "mpl-2.0", "epl-2.0",
            "cc-by-sa-3.0", "cc-by-sa-4.0", "gfdl", "odbl", "osl-3.0", "cecill"}
NONCOMMERCIAL = ("cc-by-nc", "cc-by-nd", "ntu-nc", "research", "non-commercial", "noncommercial")
LICENSE_LABEL = {0: "permissive", 1: "copyleft", 2: "use_restricted_or_custom", 3: "noncommercial"}


def license_rank(value: Any) -> float:
    """Ordered restrictiveness of a license string. NaN when no license is declared."""
    text = str(value or "").strip().lower()
    if text in baseline.TEXT_MISSING or text == "<na>":
        return float("nan")
    ranks = []
    for part in re.split(r"[,|;]", text):
        part = part.strip()
        if not part:
            continue
        if part in PERMISSIVE:
            ranks.append(0.0)
        elif part in COPYLEFT:
            ranks.append(1.0)
        elif any(part.startswith(p) or p in part for p in NONCOMMERCIAL):
            ranks.append(3.0)
        else:
            ranks.append(2.0)
    return max(ranks) if ranks else float("nan")


def leaf_params_b(variant_id: str) -> float:
    """Billions of parameters named by a variant id, such as ::8x7b:: or ::125m::."""
    match = re.search(r"::(?:(\d+)x)?([\d.]+)(b|m)::", str(variant_id))
    if not match:
        return float("nan")
    value = float(match.group(2)) * (float(match.group(1)) if match.group(1) else 1.0)
    return value / 1000.0 if match.group(3) == "m" else value


def as_float(series: pd.Series) -> np.ndarray:
    """Numeric column as plain floats, with missing values as NaN."""
    return pd.to_numeric(series, errors="coerce").astype("Float64").to_numpy(dtype=float, na_value=np.nan)


def clean_number(value: Any) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) and float(number) > 0 else float("nan")


@dataclass
class Facts:
    input_model: str
    variant_id: str
    release_date: Any
    gate_rank: int
    role: Optional[str]
    size_rank: float
    size_bucket: str
    params_b: float
    license: str
    license_rank: float
    context: float
    downloads: float
    source: str
    evidence_url: str
    note: str
    correction_target_missing: bool = False


def load_repo_tables(F) -> Tuple[Dict[str, Mapping[str, Any]], pd.DataFrame]:
    """Repository records and one representative repository per variant."""
    columns = ["repo_id", "variant_id", "created_at", "gated", "license", "downloads_30d", "parameter_count",
               "context_length", "is_likely_official_model", "is_multimodal_leak"]
    raw = pd.read_parquet(common.os.environ["COVERAGE_LINEAGE_FILE"], columns=columns)
    raw = raw[~raw["is_multimodal_leak"].fillna(False).astype(bool)].copy()
    raw["downloads_30d"] = pd.to_numeric(raw["downloads_30d"], errors="coerce").fillna(0.0)
    raw["official"] = raw["is_likely_official_model"].fillna(False).astype(bool)
    raw["created"] = pd.to_datetime(raw["created_at"], errors="coerce", utc=True)
    raw["params_b"] = as_float(raw["parameter_count"]) / 1e9
    raw["context"] = as_float(raw["context_length"])
    if CONFIG_CACHE.exists():
        cache = pd.read_csv(CONFIG_CACHE).drop_duplicates("repo_id").set_index("repo_id")["context_length"]
        cached = as_float(raw["repo_id"].map(cache))
        known = raw["context"].to_numpy(float)
        raw["context"] = np.where(np.isfinite(known), known, cached)
    raw["repo_key"] = raw["repo_id"].astype(str).str.lower()
    records = raw.drop_duplicates("repo_key").set_index("repo_key").to_dict("index")

    ranked = raw.sort_values(["official", "downloads_30d"], ascending=[False, False])
    rep = ranked.groupby("variant_id", sort=False).first()
    rep = rep.reindex(F.variant["variant_id"].astype(str).to_numpy())
    rep["license_rank"] = rep["license"].map(license_rank)
    leaf = F.variant["variant_id"].astype(str).map(leaf_params_b).to_numpy()
    rep["params_b"] = np.where(np.isfinite(rep["params_b"].to_numpy(float)), rep["params_b"].to_numpy(float), leaf)
    return records, rep


def facts_for(F, meta, rep: pd.DataFrame, records, corrections, entry, idx: int) -> Tuple[Facts, Facts]:
    """Facts as the old rule saw them (variant level) and as repaired."""
    variant_id = str(F.variant.iloc[idx]["variant_id"])
    date = meta.creation_date[idx]
    date_value = pd.NaT if np.isnat(date) else pd.Timestamp(date).tz_localize("UTC")
    size_rank = float(meta.size_rank[idx])
    bucket = baseline.SIZE_ORDER[int(size_rank)] if np.isfinite(size_rank) else "unknown"
    role = str(meta.role[idx]).strip()
    role = None if role.lower() in baseline.TEXT_MISSING else role
    row = rep.iloc[idx]
    input_model = str(entry["input_model"])
    old = Facts(input_model, variant_id, date_value, int(meta.gate_rank[idx]), role, size_rank, bucket,
                float(row["params_b"]), str(row["license"]), float(row["license_rank"]), float(row["context"]),
                float(row["downloads_30d"]), "variant_level", "", "")
    new = Facts(**old.__dict__)

    record = records.get(input_model.strip().lower())
    if record is not None:
        new.source = "own_repository_record"
        if pd.notna(record["created"]):
            new.release_date = record["created"]
        new.gate_rank = int(baseline.normalize_gate(record["gated"]))
        new.license = str(record["license"])
        new.license_rank = license_rank(record["license"])
        new.downloads = float(record["downloads_30d"])
        if np.isfinite(record["params_b"]):
            new.params_b = float(record["params_b"])
        if np.isfinite(record["context"]):
            new.context = float(record["context"])

    correction = corrections.get(input_model.strip().lower())
    if correction is not None:
        new.source = "sourced_correction"
        new.evidence_url = str(correction.get("evidence_url", ""))
        new.note = str(correction.get("correction_note", ""))
        release = pd.to_datetime(correction.get("release_date"), errors="coerce", utc=True)
        if pd.notna(release):
            new.release_date = release
        new.gate_rank = int(correction["gate_rank"])
        role_text = str(correction.get("model_role", "")).strip()
        new.role = role_text or new.role
        bucket_text = str(correction.get("size_bucket", "")).strip()
        if bucket_text in baseline.SIZE_RANK:
            new.size_bucket, new.size_rank = bucket_text, float(baseline.SIZE_RANK[bucket_text])
        params = clean_number(correction.get("total_params_b"))
        new.params_b = params if np.isfinite(params) else new.params_b
        context = clean_number(correction.get("context_length"))
        new.context = context if np.isfinite(context) else new.context
        license_text = str(correction.get("license", "")).strip()
        if license_text:
            new.license, new.license_rank = license_text, license_rank(license_text)
        # The sourced leaf may not exist in a newer taxonomy. Fall back to where the canonical
        # repository sits now, and otherwise keep the resolver's leaf for the geometry.
        target = str(correction.get("corrected_variant_id", "")).strip()
        if target not in STATE["lookup"]:
            new.correction_target_missing = bool(target)
            canonical = records.get(str(correction.get("canonical_model", "")).strip().lower())
            target = str(canonical["variant_id"]) if canonical is not None else ""
        new.variant_id = target if target in STATE["lookup"] else new.variant_id
    return old, new


def old_rule_exclusions(facts: Sequence[Facts]) -> List[str]:
    """Why the original rule left each of the panel's own models outside F_P."""
    known = [m.size_rank for m in facts if np.isfinite(m.size_rank)]
    size_max = max(known) if known else float("nan")
    reasons = []
    for model in facts:
        if np.isfinite(size_max) and not np.isfinite(model.size_rank):
            reasons.append("missing_size_bucket")
        else:
            reasons.append("")
    return reasons


def mention_reasons(model: Facts, cutoff, gate_max: int, roles, size_max: float) -> List[str]:
    """Why a mention fails the panel's own envelope, judged on its own facts."""
    reasons = []
    if pd.isna(model.release_date):
        reasons.append("missing_release_date")
    elif pd.Timestamp(model.release_date).normalize() > pd.Timestamp(cutoff).normalize():
        reasons.append("released_after_cutoff")
    if model.gate_rank > gate_max:
        reasons.append("gate_above_envelope")
    if model.role is None:
        reasons.append("missing_role")
    elif roles and model.role not in roles:
        reasons.append("role_outside_envelope")
    if not np.isfinite(model.size_rank):
        reasons.append("missing_size")
    elif np.isfinite(size_max) and model.size_rank > size_max:
        reasons.append("size_above_envelope")
    return reasons


class SlotQuota:
    """Keep the panel's size-by-role composition. A slot with unknown size accepts any known size."""

    def __init__(self, meta, facts: Sequence[Facts]) -> None:
        self.size = np.where(np.isfinite(meta.size_rank), meta.size_rank, -1).astype(int)
        self.role = np.asarray(meta.role, dtype=object)
        self.slots = [(int(m.size_rank) if np.isfinite(m.size_rank) else None, m.role) for m in facts]
        self.relaxed = 0

    def allowed(self, slots) -> np.ndarray:
        mask = np.zeros(len(self.size), dtype=bool)
        for size, role in slots:
            here = self.role == role if role is not None else np.ones(len(self.role), dtype=bool)
            mask |= here & (self.size >= 0 if size is None else self.size == size)
        return mask

    def relaxed_allowed(self, slots) -> np.ndarray:
        """When a cell has no candidate left: same size with any role, then the next smaller size."""
        self.relaxed += 1
        mask = np.zeros(len(self.size), dtype=bool)
        for size, _ in slots:
            mask |= (self.size >= 0) if size is None else (self.size == size)
        return mask, slots

    @staticmethod
    def take(slots, size: int, role: str):
        for option in [(size, role), (None, role)]:
            if option in slots:
                slots.remove(option)
                return slots
        for index, (slot_size, _) in enumerate(slots):
            if slot_size == size or slot_size is None:
                del slots[index]
                return slots
        del slots[int(np.argmin([abs((s if s is not None else size) - size) for s, _ in slots]))]
        return slots


def greedy(F, n: int, pool: np.ndarray, gain_step, quota: Optional[SlotQuota]) -> Tuple[List[int], int]:
    """Generic greedy over a candidate mask. `gain_step(selected)` returns a score per variant, higher is better."""
    available = pool.copy()
    selected: List[int] = []
    slots = list(quota.slots) if quota is not None else None
    relaxed = 0
    for _ in range(n):
        scores = gain_step(selected)
        allowed = available
        if slots is not None:
            allowed = available & quota.allowed(slots)
            if not allowed.any():
                relaxed += 1
                sizes = [s for s, _ in slots]
                allowed = available & quota.relaxed_allowed(slots)[0]
                if not allowed.any():
                    upper = max((s for s in sizes if s is not None), default=None)
                    allowed = available & (quota.size >= 0) & (quota.size <= upper if upper is not None else True)
        if not allowed.any():
            break
        chosen = int(np.argmax(np.where(allowed, scores, -np.inf)))
        selected.append(chosen)
        available[chosen] = False
        if slots is not None:
            slots = SlotQuota.take(slots, int(quota.size[chosen]), str(quota.role[chosen]))
    return selected, relaxed


def adoption_step(F, target: np.ndarray, n: int):
    target_edge = F.branch_mass(target)
    weight = 1.0 / n

    def scores(selected: List[int]) -> np.ndarray:
        counts = np.zeros(F.n_edges)
        if selected:
            np.add.at(counts, F.paths[selected].ravel(), weight)
        change = np.zeros(F.n)
        for depth in range(4):
            edge = F.paths[:, depth]
            change += F.lengths[edge] * (np.abs(target_edge[edge] - counts[edge] - weight)
                                         - np.abs(target_edge[edge] - counts[edge]))
        return -change

    return scores


def breadth_step(F, K: np.ndarray, target: np.ndarray, rows: np.ndarray, candidates: np.ndarray):
    weights = (target[rows] / target[rows].sum()).astype(np.float32)
    Kc = K[np.ix_(rows, candidates)]

    def scores(selected: List[int]) -> np.ndarray:
        best = np.zeros(len(rows), dtype=np.float32)
        if selected:
            best = K[np.ix_(rows, selected)].max(axis=1)
        gains = ((np.maximum(Kc, best[:, None]) - best[:, None]) * weights[:, None]).sum(axis=0)
        out = np.full(F.n, -np.inf)
        out[candidates] = gains
        return out

    return scores


STATE: Dict[str, Any] = {}


def init_state() -> None:
    F = foundation()
    F.kernel()
    corrections_frame = pd.read_csv(CORRECTIONS, keep_default_na=False)
    corrections = {str(r["input_model"]).strip().lower(): r for r in corrections_frame.to_dict("records")}
    records, rep = load_repo_tables(F)
    meta = baseline.load_variant_metadata(F, Path(common.os.environ["COVERAGE_LINEAGE_FILE"]))
    STATE.update(F=F, corrections=corrections, records=records, rep=rep, meta=meta,
                 lookup={str(v): i for i, v in enumerate(F.variant["variant_id"].astype(str))},
                 metadata=F.data["paper_metadata"].set_index("paper_id", drop=False))


def score_row(F, entries, eligible, label: str) -> Dict[str, float]:
    scores = baseline.score_entries(F, entries, eligible)
    return {f"{label}_adoption_gap": scores["adoption_gap"], f"{label}_breadth": scores["breadth"],
            f"{label}_structural": scores["structural_linf"], f"{label}_C0": scores["coverage_q0"],
            f"{label}_C1": scores["coverage_q1"]}


def process_paper(pid: str) -> Dict[str, Any]:
    F, meta, rep = STATE["F"], STATE["meta"], STATE["rep"]
    panel = F.panels[pid]
    entries = panel["entries"]
    hardened = baseline.hardened_indices(panel)
    pairs = [facts_for(F, meta, rep, STATE["records"], STATE["corrections"], e, i) for e, i in zip(entries, hardened)]
    old, new = [p[0] for p in pairs], [p[1] for p in pairs]
    exclusions = old_rule_exclusions(old)
    excluded = [m.input_model for m, r in zip(old, exclusions) if r]

    # Scored entries: a corrected leaf replaces the soft resolution when the taxonomy has it.
    lookup = STATE["lookup"]
    scored_entries = []
    for entry, model, base in zip(entries, new, old):
        target = lookup.get(model.variant_id)
        if model.source == "sourced_correction" and target is not None:
            scored_entries.append({"candidate_idx": np.array([target]), "posterior": np.array([1.0]),
                                   "status": "corrected", "match_type": "corrected", "input_model": model.input_model})
        else:
            scored_entries.append(entry)

    dates = [m.release_date for m in new if pd.notna(m.release_date)]
    cutoff = max(dates) if dates else pd.Timestamp(year=int(float(STATE["metadata"].loc[pid, "year"])), month=12, day=31, tz="UTC")
    created = pd.to_datetime(meta.creation_date, utc=True)
    # Availability is decided by day: some dates carry only the day, so a time of day must not exclude a same-day release.
    cutoff_day = pd.Timestamp(cutoff).normalize()
    eligible = np.asarray((~pd.isna(created)) & (created.normalize() <= cutoff_day), dtype=bool)
    own_leaves = [lookup[m.variant_id] for m in new
                  if m.variant_id in lookup and pd.notna(m.release_date) and pd.Timestamp(m.release_date).normalize() <= cutoff_day]
    eligible[own_leaves] = True

    gate_max = max(m.gate_rank for m in new)
    roles = sorted({m.role for m in new if m.role})
    sizes = [m.size_rank for m in new if np.isfinite(m.size_rank)]
    size_max = max(sizes) if sizes else float("nan")
    n = int(panel["n_resolved"])

    core = eligible & (meta.gate_rank <= gate_max)
    if roles:
        core &= np.isin(meta.role, roles)
    if np.isfinite(size_max):
        core &= meta.size_known & (meta.size_rank <= size_max)

    # Preserve the original models: a mention that passes on its own facts keeps its leaf in F_P,
    # even when the leaf's variant-level metadata is coarser than the mention's.
    reasons_by_mention = [mention_reasons(m, cutoff, gate_max, roles, size_max) for m in new]
    preserved_leaves = [lookup[m.variant_id] for m, r in zip(new, reasons_by_mention) if not r and m.variant_id in lookup]
    core[preserved_leaves] = True

    license_ranks = [m.license_rank for m in new]
    license_verifiable = all(np.isfinite(license_ranks))
    license_cap = max(license_ranks) if license_verifiable else float("nan")
    contexts = [m.context for m in new if np.isfinite(m.context)]
    context_verifiable = len(contexts) >= max(1, len(new) / 2)
    context_floor = min(contexts) if context_verifiable else float("nan")
    floor = min(ADOPTION_FLOOR, min(m.downloads for m in new))

    rep_license = rep["license_rank"].to_numpy(float)
    rep_context = rep["context"].to_numpy(float)
    rep_downloads = rep["downloads_30d"].to_numpy(float)
    rep_params = rep["params_b"].to_numpy(float)

    pools = {"historical": eligible.copy(), "core": core.copy()}
    pools["cost"] = core.copy()
    pools["license"] = pools["cost"] & (rep_license <= license_cap) if license_verifiable else pools["cost"].copy()
    pools["context"] = pools["license"] & (rep_context >= context_floor) if context_verifiable else pools["license"].copy()
    pools["adoption_floor"] = pools["context"] & (rep_downloads >= floor)
    for rung in RUNGS:
        pools[rung][preserved_leaves] = True
    applied = {"historical": "applied", "core": "applied", "cost": "applied",
               "license": "applied" if license_verifiable else "unverifiable:panel_license_unknown",
               "context": "applied" if context_verifiable else "unverifiable:panel_context_unknown",
               "adoption_floor": "applied"}

    pi_a = baseline.target_adoption(F, eligible)
    pi_b = baseline.target_breadth(F, eligible)
    observed = score_row(F, scored_entries, eligible, "observed")
    observed_params = np.nansum([m.params_b for m in new])
    known_params = int(np.isfinite([m.params_b for m in new]).sum())
    rows_population = np.where(eligible)[0]
    K = F.kernel()

    ladder, members = [], []
    for rung in RUNGS:
        pool = pools[rung]
        quota = SlotQuota(meta, new) if rung not in {"historical", "core"} else None
        pool_size = int(pool.sum())
        row: Dict[str, Any] = {"paper_id": pid, "rung": rung, "rung_status": applied[rung], "pool_size": pool_size, "n": n}
        if pool_size < n:
            row["rung_status"] = "insufficient_candidates"
            ladder.append(row)
            continue
        for label, step in [("adoption", adoption_step(F, pi_a, n)),
                            ("breadth", breadth_step(F, K, pi_b, rows_population, np.where(pool)[0]))]:
            selected, relaxed = greedy(F, n, pool, step, quota)
            scores = score_row(F, baseline.exact_entries(F, selected), eligible, label)
            row.update(scores)
            row[f"{label}_selected"] = len(selected)
            row[f"{label}_relaxed_slots"] = relaxed
            params = rep_params[selected]
            row[f"{label}_cost_ratio"] = float(np.nansum(params) / observed_params) if observed_params > 0 else float("nan")
            row[f"{label}_low_adoption_members"] = int((rep_downloads[selected] < ADOPTION_FLOOR).sum())
            for rank, idx in enumerate(selected, 1):
                members.append({
                    "paper_id": pid, "rung": rung, "panel_type": f"{label}_greedy", "rank": rank,
                    "variant_id": str(F.variant.iloc[idx]["variant_id"]), "representative_repo": str(rep.iloc[idx]["repo_id"]),
                    "repo_downloads_30d": float(rep_downloads[idx]), "params_b": float(rep_params[idx]),
                    "license": str(rep.iloc[idx]["license"]), "context_length": float(rep_context[idx]),
                })
        gap = observed["observed_adoption_gap"]
        row["adoption_gap_reduction_pct"] = 100.0 * (gap - row["adoption_adoption_gap"]) / gap if gap > 0 else float("nan")
        row["breadth_gain_pct"] = 100.0 * (row["breadth_breadth"] - observed["observed_breadth"]) / observed["observed_breadth"]
        row["C0_gain"] = row["adoption_C0"] - observed["observed_C0"]
        row["C1_gain"] = row["adoption_C1"] - observed["observed_C1"]
        best_structural = min(row["adoption_structural"], row["breadth_structural"])
        row["structural_reduction"] = observed["observed_structural"] - best_structural
        ladder.append(row)

    meta_row = STATE["metadata"].loc[pid]
    summary = {
        "paper_id": pid, "title": str(meta_row.get("title", "")), "venue": str(meta_row.get("venue", "")),
        "year": meta_row.get("year"), "n_entries": n,
        "stratum": "problematic" if excluded else "valid",
        "original_rule_eligible": not excluded,
        "excluded_mentions": len(excluded), "excluded_models": "; ".join(excluded),
        "exclusion_reasons": "; ".join(sorted({r for r in exclusions if r})),
        "repaired_mentions": sum(m.source == "sourced_correction" for m in new),
        "repository_records_used": sum(m.source == "own_repository_record" for m in new),
        "corrections_without_leaf": sum(m.correction_target_missing for m in new),
        "repaired_core_eligible": not any(reasons_by_mention),
        "repaired_core_reasons": "; ".join(sorted({r for reasons in reasons_by_mention for r in reasons})),
        "panel_unknown_sizes": sum(not np.isfinite(m.size_rank) for m in new),
        "panel_unknown_licenses": sum(not np.isfinite(m.license_rank) for m in new),
        "panel_unknown_contexts": sum(not np.isfinite(m.context) for m in new),
        "panel_known_params": known_params, "panel_params_b": float(observed_params),
        "cutoff": cutoff.date().isoformat(), "gate_max": gate_max, "roles": ";".join(roles),
        "max_size_bucket": baseline.SIZE_ORDER[int(size_max)] if np.isfinite(size_max) else "",
        "license_cap": LICENSE_LABEL.get(int(license_cap), "") if license_verifiable else "",
        "context_floor": context_floor, "adoption_floor": floor,
        **observed,
    }
    context = {"cutoff": cutoff, "eligible": eligible, "core": core, "entries": scored_entries, "n": n,
               "facts": [m.__dict__ for m in new], "old_facts": [m.__dict__ for m in old],
               "pools": {rung: pools[rung].copy() for rung in RUNGS}, "applied": applied,
               "observed_params": float(observed_params)}
    return {"summary": summary, "ladder": ladder, "members": members, "context": (pid, context)}


def aggregate(ladder: pd.DataFrame, by_paper: pd.DataFrame) -> pd.DataFrame:
    frame = ladder.merge(by_paper[["paper_id", "stratum"]], on="paper_id")
    rows = []
    for stratum, group in list(frame.groupby("stratum")) + [("all", frame)]:
        for rung in RUNGS:
            part = group[group["rung"] == rung]
            matched = part[part["rung_status"] != "insufficient_candidates"]
            if not len(part):
                continue
            rows.append({
                "stratum": stratum, "rung": rung, "papers": len(part), "comparable": len(matched),
                "applied": int((part["rung_status"] == "applied").sum()),
                "median_pool": matched["pool_size"].median(),
                "adoption_improved": int((matched["adoption_gap_reduction_pct"] > 1e-9).sum()),
                "median_adoption_reduction_pct": matched["adoption_gap_reduction_pct"].median(),
                "breadth_improved": int((matched["breadth_gain_pct"] > 1e-9).sum()),
                "median_breadth_gain_pct": matched["breadth_gain_pct"].median(),
                "structural_improved": int((matched["structural_reduction"] > 1e-12).sum()),
                "C0_improved": int((matched["C0_gain"] > 1e-12).sum()),
                "C1_improved": int((matched["C1_gain"] > 1e-12).sum()),
                "median_C1_gain": matched["C1_gain"].median(),
                "median_cost_ratio": matched["adoption_cost_ratio"].median(),
                "papers_cost_below_half": int((matched["adoption_cost_ratio"] < 0.5).sum()),
                "low_adoption_members": int(matched["adoption_low_adoption_members"].sum()),
                "relaxed_slots": int(matched["adoption_relaxed_slots"].sum()),
            })
    return pd.DataFrame(rows)


def write_figure(strata: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    colors = {"valid": "#1f77b4", "problematic": "#d95f02", "all": "#4d4d4d"}
    for stratum, marker in [("all", "o"), ("valid", "s"), ("problematic", "^")]:
        part = strata[strata["stratum"] == stratum].set_index("rung").reindex(RUNGS)
        if part["median_adoption_reduction_pct"].notna().any():
            ax.plot(RUNGS, part["median_adoption_reduction_pct"], marker=marker, linewidth=2, markersize=7,
                    color=colors[stratum], label=stratum)
    ax.set_ylabel("Median adoption-gap reduction (%)")
    ax.set_xlabel("Constraints added, cumulative")
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color="#d0d0d0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, title="Papers")
    ax.set_title("Adoption-gap reduction available under each set of constraints")
    save_figure(fig, OUT, "gain_by_constraint")
    plt.close(fig)


def write_report(by_paper: pd.DataFrame, ladder: pd.DataFrame, strata: pd.DataFrame, members: pd.DataFrame) -> None:
    n_prob = int((by_paper["stratum"] == "problematic").sum())
    show = ["stratum", "rung", "comparable", "applied", "median_pool", "adoption_improved",
            "median_adoption_reduction_pct", "breadth_improved", "structural_improved", "C1_improved",
            "median_cost_ratio", "low_adoption_members", "relaxed_slots"]
    reasons = by_paper.loc[by_paper["excluded_mentions"] > 0, "exclusion_reasons"].value_counts().to_dict()
    table = common.markdown_table(strata[show], digits=2)
    all_core = strata[(strata["stratum"] == "all") & (strata["rung"] == "core")].iloc[0]
    all_cost = strata[(strata["stratum"] == "all") & (strata["rung"] == "cost")].iloc[0]
    license_applied = int(((ladder["rung"] == "license") & (ladder["rung_status"] == "applied")).sum())
    context_applied = int(((ladder["rung"] == "context") & (ladder["rung_status"] == "applied")).sum())
    insufficient = ladder[ladder["rung_status"] == "insufficient_candidates"].groupby("rung").size().to_dict()
    text = f"""# Feasible panels, release {RELEASE}

## Summary

{n_prob} of {len(by_paper)} papers are problematic under the original rule, meaning F_P excludes at least one of their
own models. Reasons: {reasons}. After repair, {int(by_paper['repaired_core_eligible'].sum())} of {len(by_paper)} panels sit
inside the rebuilt core F_P. {int(by_paper['repaired_mentions'].sum())} mentions use the sourced corrections file and
{int(by_paper['repository_records_used'].sum())} use their own repository record.

Under the core constraints, the adoption-greedy alternative improves the gap in {int(all_core['adoption_improved'])} of
{int(all_core['comparable'])} papers (median reduction {all_core['median_adoption_reduction_pct']:.1f}%), at a median cost
ratio of {all_core['median_cost_ratio']:.2f} in parameters. Keeping the panel's size-by-role composition, it improves
{int(all_cost['adoption_improved'])} of {int(all_cost['comparable'])} (median {all_cost['median_adoption_reduction_pct']:.1f}%),
at a median cost ratio of {all_cost['median_cost_ratio']:.2f}.

## Results by stratum and constraint

{table}

`applied` counts papers where the constraint was verifiable: license in {license_applied} of {len(by_paper)} and context in
{context_applied} of {len(by_paper)}. Rungs without enough candidates: {insufficient or 'none'}.

## How to read this

- Membership of the original panel in F_P is largely true by construction once its metadata are known, because the
  envelope comes from the panel's own models. It shows the comparison is well posed, not that the panels were feasible
  in practice. `repaired_core_eligible` should therefore be read as a data-quality check.
- `adoption_gap` improvement under `core` is also near mechanical: greedy optimizes that coordinate directly over a pool
  of hundreds of candidates, with no constraint keeping the panel's scale. The `cost` rung and later rungs are the
  informative comparison.
- Cost is a proxy based on parameters. No monetary or measured inference cost exists for these papers.
- License and context limits use the panel's own envelope because the papers do not state requirements.
- Facts for models outside the correction list come from inferred metadata, which no human has validated here.

## Files

- `feasibility_by_paper.csv`: one row per paper, with eligibility, exclusion reasons, and panel facts.
- `feasibility_ladder.csv`: one row per paper and rung, with pool, gains, cost ratio, and status.
- `feasibility_strata.csv`: valid, problematic, and all papers.
- `exclusions_by_mention.csv`: every mention the original rule excluded, with repaired facts and evidence.
- `selected_members.csv`: the alternative panels, with a representative repository for each variant.
"""
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    init_state()
    F = STATE["F"]
    pids = [pid for pid, panel in F.panels.items() if panel["entries"]]
    with ProcessPoolExecutor(max_workers=min(16, len(pids))) as pool:
        results = list(pool.map(process_paper, pids))

    by_paper = pd.DataFrame([r["summary"] for r in results]).sort_values(["stratum", "paper_id"])
    ladder = pd.DataFrame([row for r in results for row in r["ladder"]])
    members = pd.DataFrame([row for r in results for row in r["members"]])
    best = ladder[ladder["rung"] == "cost"].set_index("paper_id")
    for column, label in [("adoption_gap_reduction_pct", "cost_adoption_reduction_pct"), ("breadth_gain_pct", "cost_breadth_gain_pct"),
                          ("C1_gain", "cost_C1_gain"), ("adoption_cost_ratio", "cost_ratio_alternative_over_observed")]:
        by_paper[label] = by_paper["paper_id"].map(best[column])
    core = ladder[ladder["rung"] == "core"].set_index("paper_id")
    for column, label in [("adoption_gap_reduction_pct", "core_adoption_reduction_pct"), ("breadth_gain_pct", "core_breadth_gain_pct"),
                          ("C1_gain", "core_C1_gain"), ("adoption_cost_ratio", "core_cost_ratio_alternative_over_observed")]:
        by_paper[label] = by_paper["paper_id"].map(core[column])

    exclusion_rows = []
    for result in results:
        pid, context = result["context"]
        old_reasons = old_rule_exclusions([Facts(**f) for f in context["old_facts"]])
        for reason, oldf, newf in zip(old_reasons, context["old_facts"], context["facts"]):
            if reason:
                exclusion_rows.append({
                    "paper_id": pid, "input_model": oldf["input_model"], "exclusion_reason": reason,
                    "old_variant": oldf["variant_id"], "old_size_bucket": oldf["size_bucket"], "old_role": oldf["role"],
                    "repaired_variant": newf["variant_id"], "repaired_size_bucket": newf["size_bucket"], "repaired_role": newf["role"],
                    "repaired_release_date": str(newf["release_date"])[:10], "repaired_params_b": newf["params_b"],
                    "repaired_license": newf["license"], "repaired_context": newf["context"], "repair_source": newf["source"],
                    "evidence_url": newf["evidence_url"], "note": newf["note"],
                })
    exclusions = pd.DataFrame(exclusion_rows)
    strata = aggregate(ladder, by_paper)

    by_paper.to_csv(OUT / "feasibility_by_paper.csv", index=False)
    ladder.to_csv(OUT / "feasibility_ladder.csv", index=False)
    strata.to_csv(OUT / "feasibility_strata.csv", index=False)
    exclusions.to_csv(OUT / "exclusions_by_mention.csv", index=False)
    members.to_csv(OUT / "selected_members.csv", index=False)
    with (OUTPUT / "paper_contexts.pkl").open("wb") as handle:
        pickle.dump(dict(r["context"] for r in results), handle)
    write_figure(strata)
    write_report(by_paper, ladder, strata, members)
    print((OUT / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
