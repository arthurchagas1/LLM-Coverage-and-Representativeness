#!/usr/bin/env python3
"""P1.4: how useful is the Wasserstein guarantee, beyond being true?

The bound |panel mean - population mean| <= ell * G_a holds by Kantorovich-Rubinstein duality,
so checking that the ratio stays below one shows nothing. This script asks what the bound tells a
reader: how far above the realized error it sits, how often it beats the trivial limit any
panel obeys, which model pairs set ell, and how often lineage-close models behave very differently.

Outputs in deliverables/<release>/P1_4_bound/.
"""

from __future__ import annotations

from typing import Any, Dict, List

import common
from common import (COORDINATE_NAMES, OUTCOME_LABELS, RELEASE, TABLES, ConditionedPopulation, deliverable, distances_on, foundation,
                    load_outcomes, markdown_table, save_figure, simulate_panels, simulation_sizes)

import numpy as np
import pandas as pd

OUT = deliverable("P1_4_bound")
PER_CELL = 50
ORIGINAL_SEED = 20260612
LEVELS = ["same_base", "same_series", "same_family", "different_family"]
LEVEL_LABELS = {"same_base": "Same base model", "same_series": "Same series", "same_family": "Same family", "different_family": "Different family"}


def relation_matrix(F, support: np.ndarray) -> np.ndarray:
    """Deepest shared lineage level of each pair: 0 same base, 1 same series, 2 same family, 3 different."""
    paths = F.paths[support]
    relation = np.full((len(support), len(support)), 3, dtype=int)
    for level, depth in [(2, 0), (1, 1), (0, 2)]:
        same = paths[:, depth][:, None] == paths[:, depth][None, :]
        relation[same] = level
    return relation


def main() -> None:
    F = foundation()
    variant_ids = F.variant["variant_id"].astype(str).to_numpy()
    summary_rows: List[Dict[str, Any]] = []
    relation_rows: List[Dict[str, Any]] = []
    extreme_rows: List[Dict[str, Any]] = []
    panel_frames: List[pd.DataFrame] = []

    for group, (support, outcomes) in load_outcomes().items():
        pop = ConditionedPopulation(F, support)
        values = outcomes.to_numpy(float)
        panels = simulate_panels(pop, values, simulation_sizes(len(support)), PER_CELL,
                                 lambda n, s, _g=group: np.random.default_rng(ORIGINAL_SEED + 1009 * n + 37 * s + len(_g)))
        gap = panels["X"][:, COORDINATE_NAMES.index("adoption_gap")]
        distance = distances_on(F, support)
        relation = relation_matrix(F, support)
        upper = np.triu_indices(len(support), 1)
        pair_distance = distance[upper]
        pair_relation = relation[upper]
        positive = pair_distance > 0

        for k, outcome in enumerate(outcomes.columns):
            g = values[:, k]
            local, mean, sd = pop.targets_for(values)
            mean_k, sd_k = float(mean[k]), float(sd[k])
            delta = np.abs(g[:, None] - g[None, :])[upper]
            slopes = delta[positive] / pair_distance[positive]
            ell = float(slopes.max())
            where = np.flatnonzero(positive)[int(np.argmax(slopes))]
            a, b = upper[0][where], upper[1][where]
            error = panels["E"][:, k] * sd_k
            bound = ell * gap
            ratio = error / np.maximum(bound, 1e-15)
            trivial = float(np.abs(g - mean_k).max())
            quantile_ells = {q: float(np.quantile(slopes, q)) for q in [0.5, 0.9, 0.99]}
            envelope = float(np.quantile(error / np.maximum(gap, 1e-15), 0.95))
            summary_rows.append({
                "group": group, "outcome": outcome, "population": len(support),
                "ell": ell, "ell_q50": quantile_ells[0.5], "ell_q90": quantile_ells[0.9], "ell_q99": quantile_ells[0.99],
                "pairs_above_half_ell_pct": 100.0 * float((slopes > 0.5 * ell).mean()),
                "ell_set_by_a": variant_ids[support[a]], "ell_set_by_b": variant_ids[support[b]],
                "ell_pair_distance": float(pair_distance[where]), "ell_pair_difference_points": float(delta[where] * 100),
                "max_ratio": float(ratio.max()), "median_ratio": float(np.median(ratio)), "median_bound_over_error": float(np.median(1.0 / np.maximum(ratio, 1e-12))),
                "median_bound_points": float(np.median(bound) * 100), "median_error_points": float(np.median(error) * 100),
                "population_sd_points": sd_k * 100, "trivial_limit_points": trivial * 100,
                "panels_bound_below_trivial_pct": 100.0 * float((bound < trivial).mean()),
                "panels_bound_below_one_sd_pct": 100.0 * float((bound < sd_k).mean()),
                "empirical_95_slope": envelope, "empirical_slope_over_ell": envelope / ell,
                "coverage_with_ell_q90_pct": 100.0 * float((error <= quantile_ells[0.9] * gap).mean()),
                "coverage_with_ell_q99_pct": 100.0 * float((error <= quantile_ells[0.99] * gap).mean()),
            })

            pair_sd = delta / sd_k
            for level, label in enumerate(LEVELS):
                mask = pair_relation == level
                if not mask.any():
                    continue
                relation_rows.append({
                    "group": group, "outcome": outcome, "relation": label, "pairs": int(mask.sum()),
                    "median_tree_distance": float(np.median(pair_distance[mask])),
                    "median_difference_points": float(np.median(delta[mask]) * 100),
                    "p90_difference_points": float(np.quantile(delta[mask], 0.9) * 100),
                    "median_difference_sd": float(np.median(pair_sd[mask])),
                    "pairs_differing_over_one_sd_pct": 100.0 * float((pair_sd[mask] > 1.0).mean()),
                    "pairs_differing_over_two_sd_pct": 100.0 * float((pair_sd[mask] > 2.0).mean()),
                })
            near = np.flatnonzero(pair_relation <= 1)
            for position in near[np.argsort(-delta[near])[:15]]:
                i, j = upper[0][position], upper[1][position]
                extreme_rows.append({
                    "group": group, "outcome": outcome, "variant_a": variant_ids[support[i]], "variant_b": variant_ids[support[j]],
                    "relation": LEVELS[pair_relation[position]], "tree_distance": float(pair_distance[position]),
                    "score_a_points": float(g[i] * 100), "score_b_points": float(g[j] * 100),
                    "difference_points": float(delta[position] * 100), "difference_sd": float(delta[position] / sd_k),
                })
            panel_frames.append(pd.DataFrame({"group": group, "outcome": outcome, "panel_size": panels["n"], "strategy": panels["strategy"],
                                              "adoption_gap": gap, "bound_points": bound * 100, "error_points": error * 100, "ratio": ratio}))

    summary = pd.DataFrame(summary_rows)
    relations = pd.DataFrame(relation_rows)
    extremes = pd.DataFrame(extreme_rows)
    panels_frame = pd.concat(panel_frames, ignore_index=True)
    summary.to_csv(OUT / "bound_tightness_by_benchmark.csv", index=False)
    relations.to_csv(OUT / "behavior_by_lineage_relation.csv", index=False)
    extremes.to_csv(OUT / "lineage_close_pairs_with_large_differences.csv", index=False)
    panels_frame.to_csv(OUT / "simulated_panel_bounds.csv", index=False)

    real_path = TABLES / "paper_behavioral_bounds_conditioned.csv"
    real = pd.read_csv(real_path) if real_path.exists() else pd.DataFrame()
    real_summary = pd.DataFrame()
    if len(real):
        real = real.dropna(subset=["error_to_bound_ratio"])
        real_summary = real.groupby("outcome").agg(
            papers=("paper_id", "nunique"), median_overlap_mass=("outcome_overlap_mass", "median"),
            median_error_points=("realized_mean_error", lambda s: float(s.median() * 100)),
            median_bound_points=("lipschitz_bound", lambda s: float(s.median() * 100)),
            median_ratio=("error_to_bound_ratio", "median"), max_ratio=("error_to_bound_ratio", "max")).reset_index()
        real_summary.to_csv(OUT / "audited_panel_bounds.csv", index=False)

    write_figures(panels_frame, relations, real)
    write_report(summary, relations, extremes, real_summary)
    print((OUT / "REPORT.md").read_text(encoding="utf-8"))


def write_figures(panels: pd.DataFrame, relations: pd.DataFrame, real: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outcomes = list(OUTCOME_LABELS)
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.6), sharex=True, sharey=True)
    for ax, outcome in zip(axes.ravel(), outcomes):
        part = panels[panels["outcome"] == outcome]
        ax.scatter(part["bound_points"], part["error_points"].clip(lower=1e-3), s=5, alpha=0.25, color="#1f77b4", linewidths=0, label="Simulated panels")
        if len(real):
            audited = real[real["outcome"] == outcome]
            ax.scatter(audited["lipschitz_bound"] * 100, (audited["realized_mean_error"] * 100).clip(lower=1e-3), s=16, color="#d95f02",
                       edgecolor="white", linewidth=0.4, label="Audited papers", zorder=3)
        limits = np.array([1e-1, 1e3])
        ax.plot(limits, limits, color="#4d4d4d", linewidth=0.8, linestyle="--", label="Error = bound")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 1e3)
        ax.set_ylim(1e-3, 1e3)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=10)
        ax.grid(True, color="#e6e6e6", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.supxlabel("Wasserstein bound, benchmark points")
    fig.supylabel("Realized error of the panel mean, benchmark points")
    fig.suptitle("Wasserstein bound against the realized error of the panel mean", fontsize=11)
    fig.tight_layout()
    save_figure(fig, OUT, "bound_tightness")
    plt.close(fig)

    pooled = relations.groupby("relation")[["median_difference_sd", "pairs_differing_over_one_sd_pct", "pairs_differing_over_two_sd_pct"]].median().reindex(LEVELS)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(LEVELS))
    ax.barh(y - 0.17, pooled["pairs_differing_over_one_sd_pct"], height=0.32, color="#1f77b4", label="Differ by over 1 SD")
    ax.barh(y + 0.17, pooled["pairs_differing_over_two_sd_pct"], height=0.32, color="#9ecae1", label="Differ by over 2 SD")
    ax.set_yticks(y, [LEVEL_LABELS[l] for l in LEVELS])
    ax.invert_yaxis()
    ax.set_xlabel("Share of model pairs (%), median across benchmarks")
    ax.xaxis.grid(True, color="#e0e0e0", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.set_title("Behavioral differences between model pairs, by lineage relation")
    fig.tight_layout()
    save_figure(fig, OUT, "behavior_by_lineage_relation")
    plt.close(fig)


def write_report(summary: pd.DataFrame, relations: pd.DataFrame, extremes: pd.DataFrame, real_summary: pd.DataFrame) -> None:
    tight = summary[["outcome", "ell", "ell_q90", "max_ratio", "median_ratio", "median_bound_over_error", "median_bound_points",
                     "median_error_points", "trivial_limit_points", "panels_bound_below_trivial_pct", "panels_bound_below_one_sd_pct", "empirical_slope_over_ell"]]
    near = relations[relations["relation"] == "same_base"]
    top_pairs = extremes.sort_values("difference_sd", ascending=False).head(10)[
        ["outcome", "variant_a", "variant_b", "relation", "score_a_points", "score_b_points", "difference_sd"]]
    pooled = relations.groupby("relation")[["median_difference_sd", "pairs_differing_over_one_sd_pct", "pairs_differing_over_two_sd_pct"]].median().reindex(LEVELS).reset_index()
    real_block = markdown_table(real_summary) if len(real_summary) else "Not available for this release."
    text = f"""# P1.4 Value of the Wasserstein guarantee, release {RELEASE}

## Summary

- **Validity is not informative here.** With ell computed on the same population as the panels, the ratio cannot exceed one,
  so `max_ratio` below one verifies arithmetic. Its maximum here is {summary['max_ratio'].max():.3f}.
- **Tightness.** The median bound is {summary['median_bound_over_error'].median():.0f} times the realized error (range
  {summary['median_bound_over_error'].min():.0f} to {summary['median_bound_over_error'].max():.0f} across benchmarks).
- **Non-vacuity.** The bound beats the trivial limit, the largest deviation of any single model from the population mean, in a
  median of {summary['panels_bound_below_trivial_pct'].median():.1f}% of simulated panels, and it falls below one population SD in
  {summary['panels_bound_below_one_sd_pct'].median():.1f}%.
- **Which pairs set ell.** It comes from one extreme pair, and only a median {summary['pairs_above_half_ell_pct'].median():.2f}% of pairs have
  over half that slope. The 95th percentile of error per unit of gap is a median {summary['empirical_slope_over_ell'].median():.3f} of ell.
- **Close models, different behavior.** Among pairs sharing a base model, a median {near['pairs_differing_over_one_sd_pct'].median():.1f}% differ by
  over one population SD, which makes the smoothness assumption weakest where the geometry treats models as nearly interchangeable.

## Bound against realized error, simulated panels

{markdown_table(tight)}

## Audited papers

{real_block}

## Behavior by lineage relation, median across benchmarks

{markdown_table(pooled)}

## Largest behavioral differences between lineage-close variants

{markdown_table(top_pairs)}

Variants here are taxonomy leaves, so a large difference can also reflect a taxonomy error (two unrelated models in one
leaf) rather than real behavioral divergence. These rows are worth inspecting by hand before citing them.

## Discussion for the paper

The guarantee explains why the adoption gap is the right quantity to compute: it is the exact worst case over all
lineage-smooth properties, so a panel with a small gap cannot be badly wrong about any property that varies smoothly on the tree.
It does not supply a useful numeric error bar. Worst-case ell is driven by a few extreme pairs, the resulting bound sits well
above realized errors, and it is often no better than limits any panel obeys. The honest claim is qualitative and relative, in
that lower gap tightens the worst case and, as P1.1 measures, tends to go with lower error, rather than a promise about error size.
The quantile slopes show the gap between the worst case and typical behavior, but `ell_q90` and `ell_q99` are descriptive and carry no
guarantee: the coverage they reach on these panels is an empirical observation, not a certificate.

## Files

`bound_tightness_by_benchmark.csv`, `behavior_by_lineage_relation.csv`, `lineage_close_pairs_with_large_differences.csv`,
`simulated_panel_bounds.csv`, `audited_panel_bounds.csv`, and the figures `bound_tightness` and `behavior_by_lineage_relation`.
"""
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
