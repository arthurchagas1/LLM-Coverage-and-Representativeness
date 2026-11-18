"""Priority 3: resolver ablation.

A reviewer will ask whether the conclusions hinge on how ambiguous mentions are
resolved. We compare three regimes built from the *same* candidate sets:

  exact_only            : keep only mentions that resolve to a single variant.
  posterior             : the paper's method -- each mention contributes one unit
                          of mass spread as a posterior over its candidates.
  descendant_expansion  : the naive baseline -- each candidate is counted as a
                          full member, so a k-candidate mention injects k units.

The expected story: posterior resolution conserves one unit of mass per mention,
avoids the coverage inflation of descendant expansion, and yields rankings closer
to the exact-only reference than expansion does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from coverage_geometry import Foundation, load_foundation

from project_paths import DIAGNOSTICS as OUT
OUT.mkdir(parents=True, exist_ok=True)


def singleton(idx: int) -> Dict:
    return {"candidate_idx": np.array([int(idx)]), "posterior": np.array([1.0]),
            "status": "exact", "match_type": "exact", "input_model": ""}


def regime_entries(panel: Dict, regime: str) -> List[Dict]:
    entries = panel["entries"]
    if regime == "posterior":
        return entries
    if regime == "exact_only":
        return [e for e in entries if len(e["candidate_idx"]) == 1]
    if regime == "descendant_expansion":
        flat: List[Dict] = []
        for e in entries:
            for c in e["candidate_idx"]:
                flat.append(singleton(int(c)))
        return flat
    raise ValueError(regime)


def main() -> None:
    F = load_foundation()
    regimes = ["exact_only", "posterior", "descendant_expansion"]

    rows: List[Dict] = []
    for pid, panel in F.panels.items():
        if not panel["entries"]:
            continue
        row: Dict[str, object] = {"paper_id": pid, "n_mentions": panel["n_input"],
                                  "n_resolved": panel["n_resolved"]}
        n_ambiguous = sum(1 for e in panel["entries"] if len(e["candidate_idx"]) > 1)
        n_candidates_total = sum(len(e["candidate_idx"]) for e in panel["entries"])
        row["n_ambiguous_mentions"] = n_ambiguous
        row["expansion_inflation_factor"] = n_candidates_total / max(panel["n_resolved"], 1)
        for regime in regimes:
            entries = regime_entries(panel, regime)
            if not entries:
                row[f"Ga_{regime}"] = np.nan
                row[f"B_{regime}"] = np.nan
                row[f"C0_{regime}"] = np.nan
                continue
            mb = F.metric_bundle(entries)
            row[f"Ga_{regime}"] = mb["adoption_gap"]
            row[f"B_{regime}"] = mb["breadth"]
            row[f"C0_{regime}"] = mb["coverage_q0"]
        rows.append(row)

    df = pd.DataFrame(rows)
    # ranks within each regime (lower Ga = better = rank 1)
    for regime in regimes:
        df[f"rank_{regime}"] = df[f"Ga_{regime}"].rank(method="average")
    df = df.sort_values("Ga_posterior").reset_index(drop=True)
    df.to_csv(OUT / "resolver_ablation.csv", index=False)

    # rank-stability summary: which regime tracks exact_only better?
    valid = df.dropna(subset=["rank_exact_only", "rank_posterior", "rank_descendant_expansion"])
    tau_post = kendalltau(valid["rank_exact_only"], valid["rank_posterior"]).statistic
    tau_exp = kendalltau(valid["rank_exact_only"], valid["rank_descendant_expansion"]).statistic
    amb = df[df["n_ambiguous_mentions"] > 0]
    summary = pd.DataFrame([
        {"comparison": "kendall_tau(exact_only, posterior)", "value": tau_post},
        {"comparison": "kendall_tau(exact_only, descendant_expansion)", "value": tau_exp},
        {"comparison": "papers_with_ambiguous_mentions", "value": int((df["n_ambiguous_mentions"] > 0).sum())},
        {"comparison": "mean_expansion_inflation_factor", "value": float(df["expansion_inflation_factor"].mean())},
        {"comparison": "max_expansion_inflation_factor", "value": float(df["expansion_inflation_factor"].max())},
        {"comparison": "mean_C0_posterior_over_expansion",
         "value": float((amb["C0_posterior"] / amb["C0_descendant_expansion"]).mean()) if len(amb) else np.nan},
    ])
    summary.to_csv(OUT / "resolver_ablation_summary.csv", index=False)

    print("resolver ablation ->", OUT / "resolver_ablation.csv")
    print(f"papers scored: {len(df)} | with ambiguous mentions: {(df['n_ambiguous_mentions']>0).sum()}")
    print(f"rank agreement with exact-only:  posterior tau={tau_post:.3f}  "
          f"expansion tau={tau_exp:.3f}")
    print(f"mean expansion inflation factor: {df['expansion_inflation_factor'].mean():.2f}x  "
          f"(max {df['expansion_inflation_factor'].max():.1f}x)")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
