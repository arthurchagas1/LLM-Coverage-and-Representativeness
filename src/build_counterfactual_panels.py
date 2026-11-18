"""Priority 2: same-budget counterfactual panels and single-swap recommendations.

For every real panel we construct, at the *same* model budget, three structured
alternatives -- adoption-greedy, breadth-greedy, and a hybrid -- and the single
remove/add swap that most reduces the adoption gap. This turns the metric from a
score into a constructive tool: "for the same number of models, this swap would
have reduced the adoption transport gap and increased lineage breadth."

Swap deltas are exact and O(edges-on-two-root-paths): removing variant r and
adding variant a only perturbs branch mass on the <=8 edges of their root paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from coverage_geometry import Foundation, load_foundation

from project_paths import COUNTERFACTUAL_PANELS as OUT_JSON
from project_paths import DIAGNOSTICS as OUT_TAB
OUT_JSON.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)


def add_change_vector(F: Foundation, counts: np.ndarray, target_edge: np.ndarray,
                      weight: float) -> np.ndarray:
    """Change in Ga from adding `weight` mass to each variant (greedy-step port)."""
    change = np.zeros(F.n)
    for depth in range(4):
        edge = F.paths[:, depth]
        before = np.abs(target_edge[edge] - counts[edge])
        after = np.abs(target_edge[edge] - counts[edge] - weight)
        change += F.lengths[edge] * (after - before)
    return change


def remove_delta(F: Foundation, counts: np.ndarray, target_edge: np.ndarray,
                 r: int, weight: float) -> Tuple[float, np.ndarray]:
    """Ga change and updated branch counts from removing `weight` mass at r."""
    counts_after = counts.copy()
    delta = 0.0
    for depth in range(4):
        e = F.paths[r, depth]
        before = abs(target_edge[e] - counts[e])
        after = abs(target_edge[e] - (counts[e] - weight))
        delta += F.lengths[e] * (after - before)
        counts_after[e] -= weight
    return delta, counts_after


def name(F: Foundation, idx: int) -> str:
    return str(F.variant.iloc[int(idx)]["variant_id"])


def fam(F: Foundation, idx: int) -> str:
    return str(F.variant.iloc[int(idx)]["family_id"])


def hardened_indices(panel: Dict) -> List[int]:
    """One concrete variant per resolved mention (posterior argmax)."""
    out = []
    for e in panel["entries"]:
        out.append(int(e["candidate_idx"][int(np.argmax(e["posterior"]))]))
    return out


def best_swaps(F: Foundation, indices: List[int], k: int = 3) -> List[Dict]:
    n = len(indices)
    if n == 0:
        return []
    weight = 1.0 / n
    sigma = np.bincount(indices, minlength=F.n).astype(float) / n
    target_edge = F.branch_mass(F.PI_A)
    counts = F.branch_mass(sigma)
    base_ga = float(np.dot(F.lengths, np.abs(target_edge - counts)))
    in_panel = set(int(i) for i in indices)

    swaps: List[Dict] = []
    for r in sorted(in_panel):
        rdelta, counts_after = remove_delta(F, counts, target_edge, r, weight)
        add_change = add_change_vector(F, counts_after, target_edge, weight)
        add_change[list(in_panel)] = np.inf            # cannot add an existing model
        a = int(np.argmin(add_change))
        total = rdelta + float(add_change[a])          # change in Ga (negative is better)
        swaps.append({
            "remove": name(F, r),
            "remove_family": fam(F, r),
            "add": name(F, a),
            "add_family": fam(F, a),
            "delta_Ga": round(total, 6),
            "new_Ga": round(base_ga + total, 6),
        })
    swaps.sort(key=lambda s: s["delta_Ga"])
    return swaps[:k]


def panel_summary(F: Foundation, indices: List[int]) -> Dict[str, float]:
    entries = F.exact_entries(indices)
    mb = F.metric_bundle(entries)
    return {"Ga": float(mb["adoption_gap"]), "B": float(mb["breadth"]),
            "Dinf": float(mb["structural_linf"]), "C0": float(mb["coverage_q0"])}


def hybrid_panel(F: Foundation, n: int) -> List[int]:
    """Interleave adoption-marginal and breadth-marginal picks into one set."""
    target_edge = F.branch_mass(F.PI_A)
    counts = np.zeros(F.n_edges)
    K = F.kernel()
    weight_b = (F.PI_B / F.PI_B.sum()).astype(np.float32)
    best = np.zeros(F.n, dtype=np.float32)
    selected: List[int] = []
    avail = np.ones(F.n, dtype=bool)
    w = 1.0 / n
    for k in range(n):
        if k % 2 == 0:  # adoption-greedy marginal
            change = add_change_vector(F, counts, target_edge, w)
            change[~avail] = np.inf
            j = int(np.argmin(change))
        else:           # breadth-greedy marginal
            gains = ((np.maximum(K, best[:, None]) - best[:, None]) * weight_b[:, None]).sum(axis=0)
            gains[~avail] = -np.inf
            j = int(np.argmax(gains))
        selected.append(j)
        avail[j] = False
        counts[F.paths[j]] += w
        best = np.maximum(best, K[:, j])
    return selected


def main() -> None:
    F = load_foundation()
    meta = F.data["paper_metadata"].set_index("paper_id")

    sizes = sorted({p["n_resolved"] for p in F.panels.values() if p["n_resolved"] > 0})
    print("distinct budgets:", sizes)
    greedy_a = {n: F.greedy_adoption(n)[0] for n in sizes}
    greedy_b = {n: F.greedy_breadth(n)[0] for n in sizes}
    hybrid = {n: hybrid_panel(F, n) for n in sizes}
    ref_summ = {
        n: {"greedy_adoption": panel_summary(F, greedy_a[n]),
            "greedy_breadth": panel_summary(F, greedy_b[n]),
            "hybrid": panel_summary(F, hybrid[n])}
        for n in sizes
    }

    rows: List[Dict] = []
    for pid, panel in F.panels.items():
        if not panel["entries"]:
            continue
        n = panel["n_resolved"]
        observed_idx = hardened_indices(panel)
        obs = panel_summary(F, observed_idx)
        refs = ref_summ[n]
        swaps = best_swaps(F, observed_idx, k=3)

        record = {
            "paper_id": pid,
            "venue": str(meta.loc[pid, "venue"]) if pid in meta.index else "",
            "budget": n,
            "observed_panel": [e["input_model"] for e in panel["entries"]],
            "observed_panel_resolved": [name(F, i) for i in observed_idx],
            "observed_scores": {k: round(v, 6) for k, v in obs.items()},
            "greedy_adoption_panel": [name(F, i) for i in greedy_a[n]],
            "greedy_breadth_panel": [name(F, i) for i in greedy_b[n]],
            "hybrid_panel": [name(F, i) for i in hybrid[n]],
            "reference_scores": {k: {m: round(val, 6) for m, val in v.items()}
                                 for k, v in refs.items()},
            "improvement_vs_hybrid": {
                "delta_Ga": round(obs["Ga"] - refs["hybrid"]["Ga"], 6),     # +ve: hybrid better
                "delta_B": round(refs["hybrid"]["B"] - obs["B"], 6),         # +ve: hybrid better
                "delta_Dinf": round(obs["Dinf"] - refs["hybrid"]["Dinf"], 6),
                "delta_C0": round(refs["hybrid"]["C0"] - obs["C0"], 6),
            },
            "recommended_swaps": swaps,
        }
        with (OUT_JSON / f"{pid}.json").open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)

        rows.append({
            "paper_id": pid, "budget": n,
            "observed_Ga": obs["Ga"], "hybrid_Ga": refs["hybrid"]["Ga"],
            "greedy_adoption_Ga": refs["greedy_adoption"]["Ga"],
            "greedy_breadth_Ga": refs["greedy_breadth"]["Ga"],
            "observed_B": obs["B"], "hybrid_B": refs["hybrid"]["B"],
            "greedy_breadth_B": refs["greedy_breadth"]["B"],
            "best_single_swap_delta_Ga": swaps[0]["delta_Ga"] if swaps else np.nan,
            "best_swap_remove": swaps[0]["remove"] if swaps else "",
            "best_swap_add": swaps[0]["add"] if swaps else "",
        })

    summary = pd.DataFrame(rows).sort_values("observed_Ga")
    summary.to_csv(OUT_TAB / "counterfactual_summary.csv", index=False)
    print("counterfactual JSONs:", len(rows), "->", OUT_JSON)
    frac_hybrid_better = (summary["hybrid_Ga"] < summary["observed_Ga"]).mean()
    frac_swap_helps = (summary["best_single_swap_delta_Ga"] < 0).mean()
    print(f"hybrid beats observed on Ga: {frac_hybrid_better:.1%} of panels")
    print(f"a single swap reduces Ga:    {frac_swap_helps:.1%} of panels")
    print(f"median observed Ga={summary.observed_Ga.median():.3f} | "
          f"hybrid={summary.hybrid_Ga.median():.3f} | "
          f"greedy_adoption={summary.greedy_adoption_Ga.median():.3f}")


if __name__ == "__main__":
    main()
