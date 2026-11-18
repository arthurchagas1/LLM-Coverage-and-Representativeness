"""Temporal-validity fix for the retrospective audit.

Critique: judging every paper against one present-day frozen ecosystem is
unfair, because the adoption target and the null-sampler pool contain models
released *after* the paper was written. A panel can look unrepresentative merely
because later models now dominate adoption. This script re-runs the audit under
a time-consistent eligibility set so the comparison only ever uses models that
demonstrably existed when the authors ran their experiments.

Eligibility cutoff (per paper): the release date of the *newest model the paper
itself evaluated*. If the authors used a model released on date D, then every
model released on or before D was available to them -- an unarguable, data-driven
cutoff that needs no calendar assumption. Calendar cutoffs by publication year
are reported as sensitivity.

Everything is recomputed inside the eligible set:
  * adoption target pi_a^T (present downloads renormalised over eligible models,
    same uniform floor);
  * the panel coordinate Ga^T;
  * null calibration -- uniform / popularity / family-stratified panels are drawn
    ONLY from eligible models, so "popularity" nulls cannot cheat with the future.

We then compare present-frozen vs time-consistent scoring directly, and report
whether the headline findings survive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from coverage_geometry import Foundation, load_foundation

from project_paths import DIAGNOSTICS as OUT
from project_paths import lineage_file

LINEAGE_FILE = str(lineage_file())
OUT.mkdir(parents=True, exist_ok=True)
N_NULL = 120
SEED = 20260612


def variant_creation_dates(F: Foundation) -> np.ndarray:
    """Earliest repository creation timestamp per variant, aligned to F order."""
    raw = pd.read_parquet(LINEAGE_FILE, columns=["variant_id", "created_at", "is_multimodal_leak"])
    raw = raw[~raw["is_multimodal_leak"].fillna(False)]
    raw["ca"] = pd.to_datetime(raw["created_at"], errors="coerce", utc=True)
    vmin = raw.groupby("variant_id")["ca"].min()
    order = F.variant["variant_id"].to_numpy()
    aligned = vmin.reindex(order)
    return aligned.to_numpy("datetime64[ns]")


def pi_a_eligible(F: Foundation, eligible: np.ndarray) -> np.ndarray:
    """Adoption target renormalised over the eligible support (present weights)."""
    prob = np.zeros(F.n)
    d = F.DOWNLOAD_PROB * eligible
    s = d.sum()
    prob = d / s if s > 0 else eligible / max(eligible.sum(), 1)
    uni = eligible / max(eligible.sum(), 1)
    return (1.0 - F.ALPHA) * prob + F.ALPHA * uni


FAM_CODES: np.ndarray  # set in main(); numpy family codes aligned to variant order


def null_gaps(F: Foundation, n: int, support: np.ndarray, target: np.ndarray,
              seed_off: int) -> Dict[str, np.ndarray]:
    """Same-size null-panel gaps under three samplers, all drawn from `support`.

    Pure-numpy (no pandas per draw). `support` fixes the eligible pool so that
    e.g. the popularity sampler can only pick models that existed.
    """
    pop = F.DOWNLOAD_PROB[support]
    pop = pop / pop.sum() if pop.sum() > 0 else np.full(len(support), 1.0 / len(support))
    fam_s = FAM_CODES[support]
    uniq, inv = np.unique(fam_s, return_inverse=True)
    fam_w = np.bincount(inv, weights=pop)
    fam_w = fam_w / fam_w.sum()
    target_edge = F.branch_mass(target)
    lengths = F.lengths

    def gap(idx: np.ndarray) -> float:
        sig = np.bincount(idx, minlength=F.n).astype(float) / n
        return float(np.dot(lengths, np.abs(target_edge - F.branch_mass(sig))))

    out: Dict[str, np.ndarray] = {}
    for strategy in ["uniform", "popularity", "family_stratified"]:
        rng = np.random.default_rng(SEED + 101 * n + seed_off + len(strategy))
        gaps = np.empty(N_NULL)
        for d in range(N_NULL):
            if strategy == "uniform":
                idx = rng.choice(support, n, replace=len(support) < n)
            elif strategy == "popularity":
                idx = rng.choice(support, n, replace=len(support) < n, p=pop)
            else:
                k = min(n, len(uniq))
                fam_pos = rng.choice(len(uniq), k, replace=False, p=fam_w)
                idx = np.empty(n, dtype=int)
                for j, pos in enumerate(fam_pos):
                    loc = support[fam_s == uniq[pos]]
                    lp = F.PI_A[loc]
                    idx[j] = rng.choice(loc, p=lp / lp.sum())
                if n > k:
                    idx[k:] = rng.choice(support, n - k, p=pop)
            gaps[d] = gap(idx)
        out[strategy] = gaps
    return out


def percentiles_from_gaps(gaps: Dict[str, np.ndarray], ga_panel: float, tag: str) -> Dict[str, float]:
    return {f"{s}_percentile_{tag}": float(np.mean(g >= ga_panel)) for s, g in gaps.items()}


def main() -> None:
    global FAM_CODES
    F = load_foundation()
    FAM_CODES = pd.factorize(F.variant["family_id"])[0]
    dates = variant_creation_dates(F)
    meta = F.data["paper_metadata"].set_index("paper_id")
    all_support = np.arange(F.n)

    present_gaps_cache: Dict[int, Dict[str, np.ndarray]] = {}

    rows: List[Dict] = []
    for pid, panel in F.panels.items():
        if not panel["entries"]:
            continue
        n = panel["n_resolved"]
        # hardened own models -> newest one defines the data-driven cutoff
        own = [int(e["candidate_idx"][int(np.argmax(e["posterior"]))]) for e in panel["entries"]]
        own_dates = dates[own]
        cutoff = np.nanmax(own_dates)
        eligible = (dates <= cutoff)
        n_elig = int(eligible.sum())

        sigma = F.sigma(panel["entries"])
        # present-frozen scoring (full population target)
        ga_present = F.w1(F.PI_A, sigma)
        # time-consistent scoring (eligible target)
        pi_a_T = pi_a_eligible(F, eligible)
        ga_T = F.w1(pi_a_T, sigma)

        row = {
            "paper_id": pid,
            "venue": str(meta.loc[pid, "venue"]) if pid in meta.index else "",
            "year": meta.loc[pid, "year"] if pid in meta.index else np.nan,
            "budget": n,
            "cutoff_date": pd.Timestamp(cutoff).date().isoformat() if not pd.isnull(cutoff) else "",
            "n_eligible_variants": n_elig,
            "eligible_fraction": n_elig / F.n,
            "Ga_present": ga_present,
            "Ga_timeconsistent": ga_T,
        }
        # present nulls: identical support/target across papers -> cache by size
        if n not in present_gaps_cache:
            present_gaps_cache[n] = null_gaps(F, n, all_support, F.PI_A, seed_off=0)
        row.update(percentiles_from_gaps(present_gaps_cache[n], ga_present, "present"))
        # time-consistent nulls: eligible-only pool, per paper
        elig_support = np.where(eligible)[0]
        tc_gaps = null_gaps(F, n, elig_support, pi_a_T, seed_off=17)
        row.update(percentiles_from_gaps(tc_gaps, ga_T, "timeconsistent"))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    df.to_csv(OUT / "temporal_audit.csv", index=False)

    # rank correlation present vs time-consistent
    from scipy.stats import kendalltau, spearmanr
    tau = kendalltau(df["Ga_present"].rank(), df["Ga_timeconsistent"].rank()).statistic
    rho = spearmanr(df["Ga_present"], df["Ga_timeconsistent"]).statistic

    summ = {
        "papers": len(df),
        "median_eligible_fraction": float(df["eligible_fraction"].median()),
        "median_eligible_variants": float(df["n_eligible_variants"].median()),
        "kendall_tau_rank_present_vs_timeconsistent": float(tau),
        "spearman_Ga_present_vs_timeconsistent": float(rho),
        "median_popularity_percentile_present": float(df["popularity_percentile_present"].median()),
        "median_popularity_percentile_timeconsistent": float(df["popularity_percentile_timeconsistent"].median()),
        "median_uniform_percentile_present": float(df["uniform_percentile_present"].median()),
        "median_uniform_percentile_timeconsistent": float(df["uniform_percentile_timeconsistent"].median()),
        "median_familystrat_percentile_present": float(df["family_stratified_percentile_present"].median()),
        "median_familystrat_percentile_timeconsistent": float(df["family_stratified_percentile_timeconsistent"].median()),
    }
    pd.Series(summ).to_csv(OUT / "temporal_audit_summary.csv", header=["value"])

    print("temporal audit ->", OUT / "temporal_audit.csv")
    print(f"papers: {summ['papers']} | median eligible: {summ['median_eligible_variants']:.0f}"
          f" variants ({summ['median_eligible_fraction']:.1%} of population)")
    print(f"rank stability present vs time-consistent: kendall tau={tau:.3f}, spearman={rho:.3f}")
    print("median null percentiles (higher = panel better than that null):")
    print(f"  popularity : present {summ['median_popularity_percentile_present']:.3f}"
          f"  ->  time-consistent {summ['median_popularity_percentile_timeconsistent']:.3f}")
    print(f"  uniform    : present {summ['median_uniform_percentile_present']:.3f}"
          f"  ->  time-consistent {summ['median_uniform_percentile_timeconsistent']:.3f}")
    print(f"  fam-strat  : present {summ['median_familystrat_percentile_present']:.3f}"
          f"  ->  time-consistent {summ['median_familystrat_percentile_timeconsistent']:.3f}")


if __name__ == "__main__":
    main()
