"""Post-cutoff replacement tables for the paper.

Recomputes the same-size null-calibration table (three references x three
coordinates) under the temporal-eligibility regime, so it can replace the
present-frozen table in the paper. Also emits a present-vs-time-consistent
companion for the adoption coordinate (the one the temporal critique targets).

Regimes
  present         : support = full frozen population, targets = pi_a / pi_b.
  timeconsistent  : per paper, support = variants with created_at <= t_P, where
                    t_P is the release date of the newest model the paper itself
                    evaluated; targets renormalised over that eligible support.

Coordinate percentiles are oriented so larger = better representation:
  adoption   : fraction of nulls with Ga  >= panel Ga
  breadth    : fraction of nulls with B   <= panel B
  structural : fraction of nulls with Dinf>= panel Dinf
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from coverage_geometry import Foundation, load_foundation

from project_paths import DIAGNOSTICS as OUT
from project_paths import lineage_file

LINEAGE_FILE = str(lineage_file())
OUT.mkdir(parents=True, exist_ok=True)
N_NULL = 120
SEED = 20260612
SAMPLERS = ["popularity", "family_stratified", "uniform"]
FAM_CODES: np.ndarray


def creation_dates(F: Foundation) -> np.ndarray:
    raw = pd.read_parquet(LINEAGE_FILE, columns=["variant_id", "created_at", "is_multimodal_leak"])
    raw = raw[~raw["is_multimodal_leak"].fillna(False)]
    raw["ca"] = pd.to_datetime(raw["created_at"], errors="coerce", utc=True)
    vmin = raw.groupby("variant_id")["ca"].min()
    return vmin.reindex(F.variant["variant_id"].to_numpy()).to_numpy("datetime64[ns]")


# ---- coordinate computation ------------------------------------------------
def structural_linf(F: Foundation, sigma: np.ndarray, pop_by_view: Dict[str, np.ndarray]) -> float:
    worst = 0.0
    for name, codes in F.discrepancy["codes"].items():
        pop = pop_by_view[name]
        observed = np.bincount(codes, weights=sigma, minlength=len(pop))
        worst = max(worst, float(np.max(np.abs(observed - pop))))
    return worst


def pop_views_for(F: Foundation, pi_b_target: np.ndarray) -> Dict[str, np.ndarray]:
    return {name: np.bincount(codes, weights=pi_b_target, minlength=int(codes.max()) + 1)
            for name, codes in F.discrepancy["codes"].items()}


def breadth_hard(Ksup: np.ndarray, sel: np.ndarray, weight_support: np.ndarray) -> float:
    best = Ksup[:, sel].max(axis=1)
    return float(weight_support @ best)


def breadth_soft(Ksup: np.ndarray, entries: List[Dict], weight_support: np.ndarray) -> float:
    best = np.zeros(Ksup.shape[0])
    for e in entries:
        exp = Ksup[:, e["candidate_idx"]] @ e["posterior"]
        best = np.maximum(best, exp)
    return float(weight_support @ best)


# ---- null sampling ---------------------------------------------------------
def sample_indices(F: Foundation, n: int, sampler: str, support: np.ndarray,
                   pop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if sampler == "uniform":
        return rng.choice(support, n, replace=len(support) < n)
    if sampler == "popularity":
        return rng.choice(support, n, replace=len(support) < n, p=pop)
    fam_s = FAM_CODES[support]
    uniq = np.unique(fam_s)
    fam_w = np.array([pop[fam_s == u].sum() for u in uniq])
    fam_w = fam_w / fam_w.sum()
    k = min(n, len(uniq))
    idx = np.empty(n, dtype=int)
    pos = rng.choice(len(uniq), k, replace=False, p=fam_w)
    for j, p in enumerate(pos):
        loc = support[fam_s == uniq[p]]
        lp = F.PI_A[loc]
        idx[j] = rng.choice(loc, p=lp / lp.sum())
    if n > k:
        idx[k:] = rng.choice(support, n - k, p=pop)
    return idx


def null_coords(F: Foundation, n: int, support: np.ndarray, target_a: np.ndarray,
                Ksup: np.ndarray, weight_b: np.ndarray, pop_views: Dict[str, np.ndarray],
                seed_off: int) -> Dict[str, Dict[str, np.ndarray]]:
    pop = target_a[support].copy()
    pop = pop / pop.sum()
    target_edge = F.branch_mass(target_a)
    lengths = F.lengths
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for sampler in SAMPLERS:
        rng = np.random.default_rng(SEED + 101 * n + seed_off + len(sampler))
        ga = np.empty(N_NULL); br = np.empty(N_NULL); st = np.empty(N_NULL)
        for d in range(N_NULL):
            sel = sample_indices(F, n, sampler, support, pop, rng)
            sig = np.bincount(sel, minlength=F.n).astype(float) / n
            ga[d] = float(np.dot(lengths, np.abs(target_edge - F.branch_mass(sig))))
            br[d] = breadth_hard(Ksup, sel if support is not None else sel, weight_b)
            st[d] = structural_linf(F, sig, pop_views)
        out[sampler] = {"ga": ga, "breadth": br, "structural": st}
    return out


def percentiles(obs: Tuple[float, float, float], nulls: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, float]]:
    oga, obr, ost = obs
    res = {}
    for sampler, arr in nulls.items():
        res[sampler] = {
            "adoption": float(np.mean(arr["ga"] >= oga)),
            "breadth": float(np.mean(arr["breadth"] <= obr)),
            "structural": float(np.mean(arr["structural"] >= ost)),
        }
    return res


def main() -> None:
    global FAM_CODES
    F = load_foundation()
    FAM_CODES = pd.factorize(F.variant["family_id"])[0]
    dates = creation_dates(F)
    K = F.kernel()  # (n, n) float32
    all_support = np.arange(F.n)

    # present-regime shared structures (support/targets identical across papers)
    pop_views_present = pop_views_for(F, F.PI_B)
    weight_b_present = (F.PI_B / F.PI_B.sum()).astype(np.float64)
    present_null_cache: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}

    per_paper: List[Dict] = []
    for pid, panel in F.panels.items():
        if not panel["entries"]:
            continue
        n = panel["n_resolved"]
        entries = panel["entries"]
        sigma = F.sigma(entries)
        own = [int(e["candidate_idx"][int(np.argmax(e["posterior"]))]) for e in entries]
        cutoff = np.nanmax(dates[own])
        eligible = dates <= cutoff
        elig_support = np.where(eligible)[0]

        # ---- present regime ----
        obs_present = (
            F.w1(F.PI_A, sigma),
            breadth_soft(K, entries, weight_b_present),
            structural_linf(F, sigma, pop_views_present),
        )
        if n not in present_null_cache:
            present_null_cache[n] = null_coords(
                F, n, all_support, F.PI_A, K, weight_b_present, pop_views_present, seed_off=0)
        pct_present = percentiles(obs_present, present_null_cache[n])

        # ---- time-consistent regime ----
        pa = F.DOWNLOAD_PROB * eligible
        pa = pa / pa.sum() if pa.sum() > 0 else eligible / eligible.sum()
        uni = eligible / eligible.sum()
        pi_a_T = (1 - F.ALPHA) * pa + F.ALPHA * uni
        pi_b_T = F.PI_B * eligible
        pi_b_T = pi_b_T / pi_b_T.sum()
        pop_views_T = pop_views_for(F, pi_b_T)
        Ksup_T = K[elig_support]
        weight_b_T = (pi_b_T[elig_support] / pi_b_T[elig_support].sum()).astype(np.float64)

        obs_T = (
            F.w1(pi_a_T, sigma),
            breadth_soft(Ksup_T, entries, weight_b_T),
            structural_linf(F, sigma, pop_views_T),
        )
        nulls_T = null_coords(F, n, elig_support, pi_a_T, Ksup_T, weight_b_T, pop_views_T, seed_off=17)
        pct_T = percentiles(obs_T, nulls_T)

        row = {"paper_id": pid, "n": n, "eligible_fraction": len(elig_support) / F.n}
        for reg, pct in [("present", pct_present), ("timeconsistent", pct_T)]:
            for sampler in SAMPLERS:
                for coord in ["adoption", "breadth", "structural"]:
                    row[f"{reg}_{sampler}_{coord}"] = pct[sampler][coord]
        per_paper.append(row)

    df = pd.DataFrame(per_paper)
    df.to_csv(OUT / "temporal_null_calibration_perpaper.csv", index=False)

    # ---- aggregate replacement table (medians) ----
    def med(reg: str, sampler: str, coord: str) -> float:
        return float(df[f"{reg}_{sampler}_{coord}"].median())

    label = {"popularity": "Popularity weighted", "family_stratified": "Family stratified",
             "uniform": "Uniform"}
    for reg, fname in [("timeconsistent", "table_null_calibration_timeconsistent.csv"),
                       ("present", "table_null_calibration_present.csv")]:
        tab = pd.DataFrame([
            {"Reference": label[s], "Adoption": med(reg, s, "adoption"),
             "Breadth": med(reg, s, "breadth"), "Structural": med(reg, s, "structural")}
            for s in SAMPLERS
        ])
        tab.to_csv(OUT / fname, index=False)
        print(f"\n=== null calibration [{reg}] (median percentiles, higher=better) ===")
        print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- companion: present vs time-consistent adoption percentile ----
    comp = pd.DataFrame([
        {"Reference": label[s],
         "Adoption_present": med("present", s, "adoption"),
         "Adoption_timeconsistent": med("timeconsistent", s, "adoption")}
        for s in SAMPLERS
    ])
    comp.to_csv(OUT / "table_adoption_present_vs_timeconsistent.csv", index=False)
    print("\n=== adoption percentile: present -> time-consistent ===")
    print(comp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nmedian eligible fraction: {df['eligible_fraction'].median():.1%} | papers: {len(df)}")


if __name__ == "__main__":
    main()
