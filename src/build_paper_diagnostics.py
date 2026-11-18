"""Priority 1: per-paper audit table.

Turns the abstract coordinates into an actionable, paper-level diagnostic:
which panels fail, on which coordinate, with the worst over/under-covered
lineage cell named explicitly, and how stable the ranking is across geometries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from coverage_geometry import Foundation, load_foundation

from project_paths import DIAGNOSTICS as OUT
OUT.mkdir(parents=True, exist_ok=True)

GEOMETRY_RANK_COLS = {
    "structural_variant_count": "rank_primary",
    "topological": "rank_topological",
    "structural_repo_count": "rank_repository",
    "adoption_weighted_ablation": "rank_adoption",
}


def family_mass(F: Foundation, mass: np.ndarray) -> pd.Series:
    fam = F.variant["family_id"].to_numpy()
    return pd.Series(mass, index=fam).groupby(level=0).sum()


def lineage_diagnostics(F: Foundation, sigma: np.ndarray) -> Dict[str, object]:
    """Name the worst-covered lineage cells for a panel."""
    pop_fam = family_mass(F, F.PI_A)               # adoption mass per family
    panel_fam = family_mass(F, sigma).reindex(pop_fam.index).fillna(0.0)
    gap_fam = pop_fam - panel_fam                  # +ve => panel under-covers
    over_fam = panel_fam - pop_fam                 # +ve => panel over-covers

    # most under-covered family carrying real adoption mass
    under = gap_fam.sort_values(ascending=False)
    worst_under = under.index[0]
    # most over-represented family relative to adoption
    over = over_fam.sort_values(ascending=False)
    worst_over = over.index[0]
    # single family with the largest adoption mass that the panel entirely misses
    missing = pop_fam[panel_fam <= 1e-12].sort_values(ascending=False)
    top_missing = missing.index[0] if len(missing) else ""
    top_missing_mass = float(missing.iloc[0]) if len(missing) else 0.0

    # redundancy at the base-model level: same base hit by multiple variants
    support = np.where(sigma > 0)[0]
    sub = F.variant.iloc[support]
    base_counts = (
        sub.assign(_s=sigma[support])
        .groupby(["family_id", "base_model_id"])
        .agg(n_variants=("variant_id", "nunique"), mass=("_s", "sum"))
        .reset_index()
        .sort_values(["n_variants", "mass"], ascending=False)
    )
    if len(base_counts) and int(base_counts.iloc[0]["n_variants"]) > 1:
        top_redundant_family = str(base_counts.iloc[0]["family_id"])
        top_redundant_base = str(base_counts.iloc[0]["base_model_id"])
        top_redundant_k = int(base_counts.iloc[0]["n_variants"])
    else:
        top_redundant_family, top_redundant_base, top_redundant_k = "", "", 1

    return {
        "worst_family_undercovered": str(worst_under),
        "undercovered_adoption_gap": float(under.iloc[0]),
        "worst_family_overcovered": str(worst_over),
        "overcovered_excess": float(over.iloc[0]),
        "top_missing_family": str(top_missing),
        "top_missing_adoption_mass": top_missing_mass,
        "top_redundant_family": top_redundant_family,
        "top_redundant_base": top_redundant_base,
        "top_redundant_k_variants": top_redundant_k,
    }


def build(F: Foundation) -> pd.DataFrame:
    meta = F.data["paper_metadata"].set_index("paper_id")
    geom = F.data["gt_geometry_scores"]
    rank_pivot = geom.pivot(index="paper_id", columns="geometry", values="rank")

    # greedy same-budget reference per distinct resolved size
    greedy_gap: Dict[int, float] = {}
    for n in sorted({p["n_resolved"] for p in F.panels.values() if p["n_resolved"] > 0}):
        greedy_gap[n] = F.greedy_adoption(n)[1]

    rows: List[Dict[str, object]] = []
    for pid, panel in F.panels.items():
        entries = panel["entries"]
        n_in, n_res = panel["n_input"], panel["n_resolved"]
        base = {
            "paper_id": pid,
            "venue": meta.loc[pid, "venue"] if pid in meta.index else "",
            "year": meta.loc[pid, "year"] if pid in meta.index else "",
            "n_mentions": n_in,
            "n_resolved": n_res,
            "n_unresolved": n_in - n_res,
            "resolution_rate": (n_res / n_in) if n_in else np.nan,
        }
        if not entries:
            rows.append(base)
            continue
        mb = F.metric_bundle(entries)
        sigma = F.sigma(entries)
        ga = mb["adoption_gap"]
        base.update({
            "Ga_primary": ga,
            "B_primary": mb["breadth"],
            "Dinf_primary": mb["structural_linf"],
            "Dinf_worst_partition": mb["structural_argmax_partition"],
            "Dinf_worst_cell": mb["structural_argmax_cell"],
            "C0": mb["coverage_q0"],
            "C1": mb["coverage_q1"],
            "C1_minus_C0": mb["coverage_delta_1_0"],
            "posterior_effective_variants": mb["posterior_effective_variants"],
            "same_budget_eff_adoption": (greedy_gap.get(n_res, np.nan) / ga) if ga else np.nan,
        })
        # geometry rank interval
        ranks = {}
        for gname, col in GEOMETRY_RANK_COLS.items():
            ranks[col] = float(rank_pivot.loc[pid, gname]) if (pid in rank_pivot.index and gname in rank_pivot.columns) else np.nan
        rank_vals = np.array([v for v in ranks.values() if np.isfinite(v)])
        base.update(ranks)
        base["rank_min"] = float(rank_vals.min()) if len(rank_vals) else np.nan
        base["rank_max"] = float(rank_vals.max()) if len(rank_vals) else np.nan
        base["rank_range"] = float(rank_vals.max() - rank_vals.min()) if len(rank_vals) else np.nan
        base.update(lineage_diagnostics(F, sigma))
        rows.append(base)

    df = pd.DataFrame(rows)
    return df.sort_values("Ga_primary", na_position="last").reset_index(drop=True)


def main() -> None:
    F = load_foundation()
    df = build(F)
    path = OUT / "paper_level_diagnostics.csv"
    df.to_csv(path, index=False)
    print("written:", path, "| rows:", len(df), "| scoreable:", df["Ga_primary"].notna().sum())
    cols = ["paper_id", "venue", "Ga_primary", "B_primary", "Dinf_primary",
            "C0", "rank_range", "top_missing_family", "top_redundant_family"]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
