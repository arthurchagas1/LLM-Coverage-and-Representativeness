#!/usr/bin/env python3
"""P1.2: sensitivity of the adoption target.

Each target redefines which models matter. For every target this script recomputes the
paper-level adoption gap and C(1), the greedy gain, the paper ranking, and the validation
against population means under that same target. The tree geometry stays fixed, so gaps
are comparable across targets in the same distance units.

Outputs in deliverables/<release>/P1_2_target_sensitivity/. Needs paper_contexts.pkl from
f_feasibility.py.
"""

from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

import common
from common import (CAPABILITY_OUTCOMES, OUTCOME_LABELS, OUTPUT, RELEASE, SAFETY_OUTCOMES, ConditionedPopulation, cross_validate,
                    deliverable, family_multiplicity, foundation, kernel_on, load_outcomes, markdown_table, rank_of,
                    save_figure, simulate_panels, simulation_sizes)

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import build_appendix_c as baseline  # noqa: E402

OUT = deliverable("P1_2_target_sensitivity")
PER_CELL = 50
B_FAMILY = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 200
WORKERS = 24
BASELINE = "alpha_0.10"
ORIGINAL_SEED = 20260612
PREDICTOR = {"adoption_gap": ["adoption_gap"]}

STATE: Dict[str, Any] = {}


def build_scores(F) -> Dict[str, np.ndarray]:
    """Unnormalized adoption scores per variant for each target family."""
    scores = {"downloads": F.DOWNLOADS, "log_downloads": np.log1p(F.DOWNLOADS), "likes": F.LIKES}
    if np.nansum(F.DOWNLOADS_ALL_TIME) > 0:
        scores["all_time_downloads"] = F.DOWNLOADS_ALL_TIME
    return scores


def target_definitions(F) -> Dict[str, Dict[str, Any]]:
    definitions: Dict[str, Dict[str, Any]] = {}
    for alpha in [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]:
        name = "raw_downloads" if alpha == 0 else f"alpha_{alpha:.2f}"
        definitions[name] = {"score": "downloads", "alpha": alpha, "label": "Raw downloads (alpha 0)" if alpha == 0 else f"Downloads, alpha {alpha:.2f}"}
    definitions["log_downloads"] = {"score": "log_downloads", "alpha": 0.0, "label": "Log downloads"}
    definitions["likes"] = {"score": "likes", "alpha": 0.10, "label": "Likes, alpha 0.10"}
    definitions["uniform"] = {"score": None, "alpha": 1.0, "label": "Uniform"}
    if np.nansum(F.DOWNLOADS_ALL_TIME) > 0:
        definitions["all_time_downloads"] = {"score": "all_time_downloads", "alpha": 0.10, "label": "All-time downloads, alpha 0.10"}
    return definitions


def make_target(definition: Dict[str, Any], scores: Dict[str, np.ndarray], support: np.ndarray) -> np.ndarray:
    """Target restricted to a boolean support, mixing the score share with a uniform floor."""
    mask = support.astype(float)
    uniform = mask / mask.sum()
    if definition["score"] is None:
        return uniform
    score = scores[definition["score"]] * mask
    share = score / score.sum() if score.sum() > 0 else uniform
    target = (1.0 - definition["alpha"]) * share + definition["alpha"] * uniform
    return target / target.sum()


def paper_job(job) -> List[Dict[str, Any]]:
    pid, context = job
    F, scores, definitions = STATE["F"], STATE["scores"], STATE["definitions"]
    sigma = F.sigma(context["entries"])
    eligible, n = context["eligible"], context["n"]
    candidates = np.where(context["core"])[0]
    rows = []
    for name, definition in definitions.items():
        target = make_target(definition, scores, eligible)
        gap = F.w1(target, sigma)
        inclusion = F.inclusion_probability(context["entries"])
        c1 = float(np.dot(inclusion, target))
        selected = baseline.greedy_adoption_support(F, n, target, candidates) if len(candidates) >= n else []
        greedy_gap = F.w1(target, np.bincount(selected, minlength=F.n) / n) if selected else np.nan
        selected_all = baseline.greedy_adoption_support(F, n, target, np.where(eligible)[0])
        rows.append({
            "paper_id": pid, "target": name, "adoption_gap": gap, "C1": c1,
            "greedy_core_gap": greedy_gap, "reduction_core_pct": 100.0 * (gap - greedy_gap) / gap if gap > 0 else np.nan,
            "reduction_historical_pct": 100.0 * (gap - F.w1(target, np.bincount(selected_all, minlength=F.n) / n)) / gap if gap > 0 else np.nan,
            "effective_support": float(1.0 / np.sum(target[target > 0] ** 2)),
        })
    return rows


def validation_job(job) -> Dict[str, Any]:
    group, name, replicate = job
    state = STATE["groups"][group]
    F = STATE["F"]
    pi_a = make_target(STATE["definitions"][name], STATE["scores"], np.ones(F.n, dtype=bool))
    multiplicity = None
    if replicate >= 0:
        rng = np.random.default_rng([common.SEED, 22, replicate, len(group)])
        multiplicity = family_multiplicity(state["pop"], rng)
    pop = ConditionedPopulation(F, state["support"], kernel=state["kernel"], pi_a=pi_a, multiplicity=multiplicity)

    def rng_for(n: int, s: int) -> np.random.Generator:
        # The point estimate uses the seeds of the original design, as P1.1 does.
        if replicate < 0:
            return np.random.default_rng(ORIGINAL_SEED + 1009 * n + 37 * s + len(group))
        return np.random.default_rng([common.SEED, 23, replicate, n, s, len(group)])

    panels = simulate_panels(pop, state["values"], state["sizes"], PER_CELL, rng_for)
    by_size = cross_validate(panels["X"], panels["E"], panels["n"], PREDICTOR)["adoption_gap"]
    by_strategy = cross_validate(panels["X"], panels["E"], panels["strategy"], PREDICTOR)["adoption_gap"]
    return {"group": group, "target": name, "replicate": replicate, "size": by_size, "strategy": by_strategy, "names": state["names"]}


def main() -> None:
    F = foundation()
    contexts = pickle.load((OUTPUT / "paper_contexts.pkl").open("rb"))
    scores = build_scores(F)
    definitions = target_definitions(F)
    groups: Dict[str, Dict[str, Any]] = {}
    for group, (support, outcomes) in load_outcomes().items():
        kernel = kernel_on(F, support)
        groups[group] = {"support": support, "values": outcomes.to_numpy(float), "names": list(outcomes.columns), "kernel": kernel,
                         "sizes": simulation_sizes(len(support)), "pop": ConditionedPopulation(F, support, kernel=kernel)}
    STATE.update(F=F, scores=scores, definitions=definitions, groups=groups)

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        paper_rows = [row for chunk in pool.map(paper_job, list(contexts.items())) for row in chunk]
        jobs = [(g, name, r) for g in groups for name in definitions for r in [-1, *range(B_FAMILY)]]
        results = list(pool.map(validation_job, jobs, chunksize=4))

    papers = pd.DataFrame(paper_rows)
    papers.to_csv(OUT / "paper_metrics_by_target.csv", index=False)

    # Metric and ranking stability against the baseline target.
    pivot = papers.pivot(index="paper_id", columns="target", values="adoption_gap")
    stability_rows = []
    for name, definition in definitions.items():
        part = papers[papers["target"] == name]
        rank_shift = np.abs(rank_of(pivot[name].to_numpy()) - rank_of(pivot[BASELINE].to_numpy()))
        tau = kendalltau(pivot[name], pivot[BASELINE]).statistic
        top = set(pivot[name].nsmallest(10).index) & set(pivot[BASELINE].nsmallest(10).index)
        stability_rows.append({
            "target": name, "label": definition["label"], "effective_support": part["effective_support"].median(),
            "median_adoption_gap": part["adoption_gap"].median(), "median_C1": part["C1"].median(),
            "median_reduction_core_pct": part["reduction_core_pct"].median(),
            "median_reduction_historical_pct": part["reduction_historical_pct"].median(),
            "papers_improved_core": int((part["reduction_core_pct"] > 1e-9).sum()),
            "kendall_tau_vs_baseline": float(tau), "median_rank_shift": float(np.median(rank_shift)), "max_rank_shift": float(rank_shift.max()),
            "best_10_overlap": len(top),
        })
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(OUT / "stability_by_target.csv", index=False)
    pairwise = pd.DataFrame({a: {b: kendalltau(pivot[a], pivot[b]).statistic for b in definitions} for a in definitions})
    pairwise.to_csv(OUT / "rank_correlation_between_targets.csv")

    # Validation against the population mean defined by each target.
    validation_rows = []
    for group, state in groups.items():
        for name in definitions:
            here = [r for r in results if r["group"] == group and r["target"] == name]
            point = next(r for r in here if r["replicate"] == -1)
            boots = [r for r in here if r["replicate"] >= 0]
            for scheme in ["size", "strategy"]:
                for k, outcome in enumerate(state["names"]):
                    r2 = [b[scheme]["delta_r2"][k] for b in boots]
                    mae = [b[scheme]["delta_mae"][k] for b in boots]
                    validation_rows.append({
                        "group": group, "outcome": outcome, "target": name, "scheme": scheme,
                        "delta_r2": point[scheme]["delta_r2"][k], "delta_r2_low": np.nanquantile(r2, 0.025), "delta_r2_high": np.nanquantile(r2, 0.975),
                        "delta_mae": point[scheme]["delta_mae"][k], "delta_mae_low": np.nanquantile(mae, 0.025), "delta_mae_high": np.nanquantile(mae, 0.975),
                    })
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(OUT / "error_improvement_by_target.csv", index=False)
    means = validation.groupby(["target", "scheme"]).agg(mean_delta_r2=("delta_r2", "mean"), positive=("delta_r2", lambda s: int((s > 0).sum())),
                                                          ci_above_zero=("delta_r2_low", lambda s: int((s > 0).sum())), mean_delta_mae=("delta_mae", "mean")).reset_index()
    means.to_csv(OUT / "error_improvement_summary.csv", index=False)

    write_figures(stability, pairwise, validation, definitions)
    write_report(stability, means, definitions)
    print((OUT / "REPORT.md").read_text(encoding="utf-8"))


def write_figures(stability: pd.DataFrame, pairwise: pd.DataFrame, validation: pd.DataFrame, definitions) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(definitions)
    labels = [definitions[k]["label"] for k in order]
    frame = stability.set_index("target").reindex(order)
    y = np.arange(len(order))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), sharey=True)
    for ax, column, title in [(axes[0], "median_adoption_gap", "Median adoption gap"),
                              (axes[1], "median_reduction_core_pct", "Median greedy reduction, core pool (%)"),
                              (axes[2], "median_C1", "Median C(1)")]:
        ax.hlines(y, 0, frame[column], color="#b8c7d9", linewidth=1.2)
        ax.plot(frame[column], y, "o", color="#1f77b4", markersize=6)
        ax.set_title(title, fontsize=10)
        ax.xaxis.grid(True, color="#e0e0e0", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Paper-level metrics under each adoption target", fontsize=11)
    fig.tight_layout()
    save_figure(fig, OUT, "metric_stability")
    plt.close(fig)

    matrix = pairwise.reindex(index=order, columns=order)
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    image = ax.imshow(matrix.to_numpy(float), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(y, labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(y, labels, fontsize=8)
    for i in range(len(order)):
        for j in range(len(order)):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value > 0.6 else "#222222")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Kendall tau of paper rankings")
    ax.set_title("Agreement of paper rankings between targets")
    fig.tight_layout()
    save_figure(fig, OUT, "rank_stability")
    plt.close(fig)

    size = validation[validation["scheme"] == "size"]
    outcomes = CAPABILITY_OUTCOMES + SAFETY_OUTCOMES
    grid = size.pivot_table(index="target", columns="outcome", values="delta_r2").reindex(index=order, columns=outcomes)
    limit = float(np.nanmax(np.abs(grid.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    image = ax.imshow(grid.to_numpy(float), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(outcomes)), [OUTCOME_LABELS[o] for o in outcomes], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(y, labels, fontsize=8)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(value) > 0.6 * limit else "#222222")
    fig.colorbar(image, ax=ax, shrink=0.8, label=r"$\Delta R^2_{\mathrm{OOS}}$ from the adoption gap")
    ax.set_title("Error improvement under each target, leave one size out")
    fig.tight_layout()
    save_figure(fig, OUT, "error_improvement")
    plt.close(fig)


def write_report(stability: pd.DataFrame, means: pd.DataFrame, definitions) -> None:
    base = stability.set_index("target").loc[BASELINE]
    near = stability[stability["target"].str.startswith("alpha_") | (stability["target"] == "raw_downloads")]
    far = stability[~stability["target"].isin(near["target"])]
    keep = ["target", "effective_support", "median_adoption_gap", "median_reduction_core_pct", "papers_improved_core",
            "kendall_tau_vs_baseline", "median_rank_shift", "max_rank_shift", "best_10_overlap"]
    wide = means.pivot(index="target", columns="scheme", values="mean_delta_r2").reset_index()
    wide = wide.merge(means[means["scheme"] == "size"][["target", "positive", "ci_above_zero"]], on="target")
    text = f"""# P1.2 Sensitivity of the adoption target, release {RELEASE}

## Summary

Baseline target: downloads with alpha 0.10, effective support {base['effective_support']:.0f}, median gap {base['median_adoption_gap']:.3f}.

Across download-based alpha values from 0 to 0.50, paper rankings keep a Kendall tau of at least
{near['kendall_tau_vs_baseline'].min():.3f} with the baseline. Across the other targets (log downloads, likes, uniform, and all-time
downloads when available), the lowest tau is {far['kendall_tau_vs_baseline'].min():.3f} and the largest single rank shift is
{far['max_rank_shift'].max():.0f} positions.

## Metrics and rankings

{markdown_table(stability[keep])}

## Error improvement against each target's own population mean

Mean delta R2 across benchmarks, by separation scheme. `positive` and `ci_above_zero` refer to the leave-size-out scheme,
with family-cluster intervals from {B_FAMILY} replicates.

{markdown_table(wide)}

## How to read this

- The tree is the same under every target, so the gaps share one distance scale.
- Each target changes the estimand: the population mean being recovered changes with it, so error improvements describe
  how well the gap tracks recovery of that target's own mean, not of one fixed mean.
- A panel's rank can change because the target moved, not because the panel did. Use `rank_correlation_between_targets.csv`
  to see which targets are interchangeable.
- All-time downloads exist only in snapshots collected from 2026-11 on.

## Files

`paper_metrics_by_target.csv`, `stability_by_target.csv`, `rank_correlation_between_targets.csv`, `error_improvement_by_target.csv`,
`error_improvement_summary.csv`, and the figures `metric_stability`, `rank_stability`, and `error_improvement`.
"""
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
