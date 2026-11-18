#!/usr/bin/env python3
"""P1.1: statistical validation that accounts for panels reusing the same models.

The original validation left one panel size out and bootstrapped panels. Panels from
different sizes still share models, so its "out of sample" did not cover new models, and its
intervals treated panels as independent. This script evaluates each coordinate under:

  size            leave one panel size out (original scheme)
  strategy        leave one selection strategy out
  family_disjoint train and test on panels from disjoint family sets, so no model is shared

Intervals come from two bootstraps, reported side by side:

  panel           resample panels within size strata (original method, ignores dependence)
  family_cluster  resample model families, rebuild the population, redraw all panels, and rerun

Aggregates across benchmarks use the replicate-level mean, which gives a real interval for
the mean instead of the average of per-benchmark lower endpoints.

Outputs in deliverables/<release>/P1_1_validation/.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

import common
from common import (CAPABILITY_OUTCOMES, CONTROLS, COORDINATE_NAMES, OUTCOME_LABELS, PREDICTOR_LABELS, PREDICTOR_SETS,
                    PROFILE, RELEASE, SAFETY_OUTCOMES, STRATEGIES, ConditionedPopulation, cross_validate, deliverable,
                    family_disjoint_cv, family_multiplicity, foundation, kernel_on, load_outcomes, markdown_table,
                    save_figure, simulate_panels, simulation_sizes)

import numpy as np
import pandas as pd

OUT = deliverable("P1_1_validation")
PER_CELL = 50          # panels per size-by-strategy cell, as in the original design
FAMILY_PER_CELL = 25   # per cell and side in the family-disjoint design
FAMILY_SPLITS = 10     # random family splits for the point estimate
FAMILY_SPLITS_BOOT = 2 # random family splits inside each bootstrap replicate
B_PANEL = 150
SEED_DRAWS = 100       # fresh panel draws on the same population, to size the sampling noise of the design
B_FAMILY = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 500
WORKERS = 24
SCHEMES = ["size", "strategy", "family_disjoint"]
METRICS = ["delta_r2", "delta_mae"]
ORIGINAL_SEED = 20260612

GROUPS: Dict[str, Dict[str, Any]] = {}


def setup() -> None:
    F = foundation()
    for group, (support, outcomes) in load_outcomes().items():
        kernel = kernel_on(F, support)
        GROUPS[group] = {"F": F, "support": support, "values": outcomes.to_numpy(float),
                         "names": list(outcomes.columns), "kernel": kernel,
                         "sizes": simulation_sizes(len(support)), "pop": ConditionedPopulation(F, support, kernel=kernel)}


def original_rng(group: str):
    """Same seeds as the source, so the point estimates start from the same panels."""
    def rng_for(n: int, strategy_index: int) -> np.random.Generator:
        return np.random.default_rng(ORIGINAL_SEED + 1009 * n + 37 * strategy_index + len(group))
    return rng_for


def evaluate(pop: ConditionedPopulation, values: np.ndarray, sizes, panels, seed: int, splits: int) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    return {
        "size": cross_validate(panels["X"], panels["E"], panels["n"]),
        "strategy": cross_validate(panels["X"], panels["E"], panels["strategy"]),
        "family_disjoint": family_disjoint_cv(pop, values, sizes, FAMILY_PER_CELL, splits, seed),
    }


def within_size_spearman(X: np.ndarray, E: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Rank correlation between gap and error inside each panel size, averaged over sizes.

    Needs no fitted model, so it stays stable where R2 explodes on a narrow test set.
    """
    gap = X[:, COORDINATE_NAMES.index("adoption_gap")]
    out = []
    for n in np.unique(sizes):
        mask = sizes == n
        gap_rank = pd.Series(gap[mask]).rank().to_numpy()
        error_rank = pd.DataFrame(E[mask]).rank().to_numpy()
        gap_c = gap_rank - gap_rank.mean()
        error_c = error_rank - error_rank.mean(axis=0)
        denominator = np.sqrt((gap_c ** 2).sum() * (error_c ** 2).sum(axis=0))
        out.append((gap_c @ error_c) / np.where(denominator > 0, denominator, np.nan))
    return np.nanmean(out, axis=0)


def disjoint_side_association(pop: ConditionedPopulation, values: np.ndarray, sizes, seed: int, splits: int) -> np.ndarray:
    """The same association on panels drawn from one side of a family split at a time."""
    rng = np.random.default_rng(seed)
    results = []
    for split in range(splits):
        side_one = common.family_split(pop, rng)
        for side, subset in enumerate([np.where(~side_one[pop.family_codes])[0], np.where(side_one[pop.family_codes])[0]]):
            if len(subset) == 0:
                continue
            def rng_for(n, s, _split=split, _side=side):
                return np.random.default_rng([seed, 7, _split, _side, n, s])
            panels = simulate_panels(pop, values, sizes, FAMILY_PER_CELL, rng_for, subset=subset)
            results.append(within_size_spearman(panels["X"], panels["E"], panels["n"]))
    return np.nanmean(results, axis=0)


def panel_bootstrap(job) -> Dict[str, np.ndarray]:
    group, replicate = job
    state = GROUPS[group]
    panels = state["base_panels"]
    rng = np.random.default_rng([common.SEED, 1, replicate, len(group)])
    index = np.concatenate([rng.choice(np.where(panels["n"] == n)[0], (panels["n"] == n).sum()) for n in np.unique(panels["n"])])
    fits = cross_validate(panels["X"][index], panels["E"][index], panels["n"][index])
    return {name: fit["delta_r2"] for name, fit in fits.items()}


def family_bootstrap(job) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    group, replicate = job
    state = GROUPS[group]
    rng = np.random.default_rng([common.SEED, 2, replicate, len(group)])
    multiplicity = family_multiplicity(state["pop"], rng)
    pop = ConditionedPopulation(state["F"], state["support"], kernel=state["kernel"], multiplicity=multiplicity)
    def rng_for(n: int, s: int) -> np.random.Generator:
        return np.random.default_rng([common.SEED, 3, replicate, n, s, len(group)])
    panels = simulate_panels(pop, state["values"], state["sizes"], PER_CELL, rng_for)
    fits = evaluate(pop, state["values"], state["sizes"], panels, seed=common.SEED + replicate, splits=FAMILY_SPLITS_BOOT)
    out = {scheme: {name: {m: fit[m] for m in METRICS} for name, fit in by_name.items()} for scheme, by_name in fits.items()}
    out["association"] = {"all_models": within_size_spearman(panels["X"], panels["E"], panels["n"]),
                          "disjoint_families": disjoint_side_association(pop, state["values"], state["sizes"], common.SEED + replicate, FAMILY_SPLITS_BOOT)}
    return out


def seed_draw(job) -> Dict[str, Any]:
    """Redraw all panels with a new seed on the unchanged population."""
    group, draw = job
    state = GROUPS[group]
    def rng_for(n: int, s: int) -> np.random.Generator:
        return np.random.default_rng([common.SEED, 9, draw, n, s, len(group)])
    panels = simulate_panels(state["pop"], state["values"], state["sizes"], PER_CELL, rng_for)
    return {scheme: {name: fit["delta_r2"] for name, fit in
                     cross_validate(panels["X"], panels["E"], panels[column]).items()}
            for scheme, column in [("size", "n"), ("strategy", "strategy")]}


def leave_family_out(job) -> List[Dict[str, Any]]:
    """Influence of each large family: drop it from the population and rerun the size scheme."""
    group, family_position = job
    state = GROUPS[group]
    pop = state["pop"]
    multiplicity = (pop.family_codes != family_position).astype(float)
    reduced = ConditionedPopulation(state["F"], state["support"], kernel=state["kernel"], multiplicity=multiplicity)
    panels = simulate_panels(reduced, state["values"], state["sizes"], PER_CELL, original_rng(group))
    fit = cross_validate(panels["X"], panels["E"], panels["n"], {"adoption_gap": ["adoption_gap"]})["adoption_gap"]
    return [{"group": group, "dropped_family": pop.family_names[family_position], "variants_dropped": int((multiplicity == 0).sum()),
             "adoption_mass_dropped": float(pop.sampling_prob[multiplicity == 0].sum()), "outcome": outcome,
             "delta_r2": float(fit["delta_r2"][k]), "delta_mae": float(fit["delta_mae"][k])}
            for k, outcome in enumerate(state["names"])]


def percentile(samples: List[float]) -> tuple:
    array = np.asarray(samples, float)
    array = array[np.isfinite(array)]
    return (float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))) if len(array) else (np.nan, np.nan)


def main() -> None:
    setup()
    rows: List[Dict[str, Any]] = []
    summary_samples: Dict[tuple, List[float]] = {}
    panel_samples_all: Dict[tuple, List[float]] = {}
    per_group_family: Dict[str, List] = {}
    per_group_panel: Dict[str, List] = {}
    tercile_rows: List[Dict[str, Any]] = []

    for group, state in GROUPS.items():
        pop = state["pop"]
        state["base_panels"] = simulate_panels(pop, state["values"], state["sizes"], PER_CELL, original_rng(group))
        panels = state["base_panels"]
        print(f"[{group}] {len(panels['E'])} panels, {len(pop.family_names)} families", flush=True)
        state["point"] = evaluate(pop, state["values"], state["sizes"], panels, seed=common.SEED, splits=FAMILY_SPLITS)

    # Workers inherit the state above through fork, so the executor starts after it exists.
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for group, state in GROUPS.items():
            per_group_panel[group] = list(pool.map(panel_bootstrap, [(group, b) for b in range(B_PANEL)], chunksize=4))
            print(f"[{group}] panel bootstrap done", flush=True)
            per_group_family[group] = list(pool.map(family_bootstrap, [(group, b) for b in range(B_FAMILY)], chunksize=2))
            print(f"[{group}] family bootstrap done", flush=True)
        influence_jobs = []
        for group, state in GROUPS.items():
            pop = state["pop"]
            ranked = np.argsort(-pop.family_prob)[:6]
            influence_jobs += [(group, int(k)) for k in ranked if len(pop.family_members[int(k)]) < len(pop.support)]
        influence = [row for chunk in pool.map(leave_family_out, influence_jobs) for row in chunk]
        seed_results = {group: list(pool.map(seed_draw, [(group, d) for d in range(SEED_DRAWS)], chunksize=4)) for group in GROUPS}

    for group, state in GROUPS.items():
        names, point = state["names"], state["point"]
        sd = state["base_panels"]["sd"]
        for scheme in SCHEMES:
            for predictor in PREDICTOR_SETS:
                fit = point[scheme][predictor]
                for k, outcome in enumerate(names):
                    samples_r2 = [rep[scheme][predictor]["delta_r2"][k] for rep in per_group_family[group]]
                    samples_mae = [rep[scheme][predictor]["delta_mae"][k] for rep in per_group_family[group]]
                    low_r2, high_r2 = percentile(samples_r2)
                    low_mae, high_mae = percentile(samples_mae)
                    row = {
                        "group": group, "outcome": outcome, "predictor": predictor, "scheme": scheme,
                        "r2_base": fit["r2_base"][k], "r2_augmented": fit["r2_aug"][k], "delta_r2": fit["delta_r2"][k],
                        "delta_r2_family_low": low_r2, "delta_r2_family_high": high_r2,
                        "mae_base": fit["mae_base"][k], "mae_augmented": fit["mae_aug"][k], "delta_mae": fit["delta_mae"][k],
                        "delta_mae_family_low": low_mae, "delta_mae_family_high": high_mae,
                        "delta_mae_benchmark_points": fit["delta_mae"][k] * sd[k] * 100.0,
                        "population_sd_points": sd[k] * 100.0,
                    }
                    if scheme == "size":
                        row["delta_r2_panel_low"], row["delta_r2_panel_high"] = percentile([rep[predictor][k] for rep in per_group_panel[group]])
                    rows.append(row)

        # Realized error in benchmark points by adoption-gap tercile, within each panel size.
        panels = state["base_panels"]
        gap = panels["X"][:, COORDINATE_NAMES.index("adoption_gap")]
        raw_error = panels["E"] * panels["sd"] * 100.0
        for n in np.unique(panels["n"]):
            mask = panels["n"] == n
            cuts = np.quantile(gap[mask], [1 / 3, 2 / 3])
            for label, selection in [("lowest_gap", gap[mask] <= cuts[0]), ("highest_gap", gap[mask] >= cuts[1])]:
                for k, outcome in enumerate(state["names"]):
                    tercile_rows.append({"group": group, "panel_size": int(n), "gap_tercile": label, "outcome": outcome,
                                         "mean_error_points": raw_error[mask][selection, k].mean(),
                                         "median_error_points": float(np.median(raw_error[mask][selection, k]))})

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "robustness_by_benchmark.csv", index=False)

    association_rows = []
    for group, state in GROUPS.items():
        panels = state["base_panels"]
        point_all = within_size_spearman(panels["X"], panels["E"], panels["n"])
        point_disjoint = disjoint_side_association(state["pop"], state["values"], state["sizes"], common.SEED, FAMILY_SPLITS)
        for k, outcome in enumerate(state["names"]):
            for label, point in [("all_models", point_all), ("disjoint_families", point_disjoint)]:
                low, high = percentile([rep["association"][label][k] for rep in per_group_family[group]])
                association_rows.append({"group": group, "outcome": outcome, "panels": label, "within_size_spearman": point[k], "family_low": low, "family_high": high})
    association = pd.DataFrame(association_rows)
    association.to_csv(OUT / "gap_error_association.csv", index=False)
    pd.DataFrame(influence).to_csv(OUT / "family_influence.csv", index=False)
    tercile = pd.DataFrame(tercile_rows)
    tercile.to_csv(OUT / "error_by_gap_tercile.csv", index=False)

    # Replicate-level means across the eight benchmarks, pairing replicates across the two groups.
    summary_rows = []
    groups = list(GROUPS)
    scopes = {"all": groups, **{g: [g] for g in groups}}
    for scope, members in scopes.items():
      for scheme in SCHEMES:
        for predictor in PREDICTOR_SETS:
            samples = {metric: [] for metric in METRICS}
            for b in range(B_FAMILY):
                for metric in METRICS:
                    values = np.concatenate([per_group_family[g][b][scheme][predictor][metric] for g in members])
                    samples[metric].append(float(np.nanmean(values)))
            part = table[(table["scheme"] == scheme) & (table["predictor"] == predictor) & (table["group"].isin(members))]
            low_r2, high_r2 = percentile(samples["delta_r2"])
            low_mae, high_mae = percentile(samples["delta_mae"])
            summary_rows.append({
                "scope": scope, "scheme": scheme, "predictor": predictor,
                "mean_delta_r2": part["delta_r2"].mean(), "mean_delta_r2_low": low_r2, "mean_delta_r2_high": high_r2,
                "benchmarks_positive": int((part["delta_r2"] > 0).sum()),
                "benchmarks_ci_above_zero": int((part["delta_r2_family_low"] > 0).sum()),
                "mean_delta_mae": part["delta_mae"].mean(), "mean_delta_mae_low": low_mae, "mean_delta_mae_high": high_mae,
                "benchmarks": len(part),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "robustness_summary.csv", index=False)

    # How much the headline moves from redrawing panels alone, with nothing else changed.
    seed_rows = []
    for scope, members in scopes.items():
      for scheme in ["size", "strategy"]:
        for predictor in PREDICTOR_SETS:
            means = [float(np.mean(np.concatenate([seed_results[g][d][scheme][predictor] for g in members]))) for d in range(SEED_DRAWS)]
            positive = [int(sum((seed_results[g][d][scheme][predictor] > 0).sum() for g in members)) for d in range(SEED_DRAWS)]
            seed_rows.append({"scope": scope, "scheme": scheme, "predictor": predictor, "mean_of_means": float(np.mean(means)), "sd_of_means": float(np.std(means)),
                              "p2_5": float(np.quantile(means, 0.025)), "p97_5": float(np.quantile(means, 0.975)),
                              "median_benchmarks_positive": float(np.median(positive)), "min_benchmarks_positive": int(np.min(positive))})
    seed_stability = pd.DataFrame(seed_rows)
    seed_stability.to_csv(OUT / "seed_stability.csv", index=False)

    write_figure(table)
    write_report(table, summary, pd.DataFrame(influence), tercile, association, seed_stability)
    print((OUT / "REPORT.md").read_text(encoding="utf-8"))


def write_figure(table: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outcomes = CAPABILITY_OUTCOMES + SAFETY_OUTCOMES
    gap = table[table["predictor"] == "adoption_gap"]
    colors = {"size": "#1f77b4", "strategy": "#d95f02", "family_disjoint": "#2a9d4b"}
    labels = {"size": "Leave size out", "strategy": "Leave strategy out", "family_disjoint": "Disjoint families"}
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    offsets = {"size": -0.22, "strategy": 0.0, "family_disjoint": 0.22}
    # R2 collapses far below zero in thin family-disjoint test sets, so the axis is limited and truncated intervals carry an arrow.
    x_low, x_high = -1.0, 1.6
    for scheme in SCHEMES:
        part = gap[gap["scheme"] == scheme].set_index("outcome").reindex(outcomes)
        y = np.arange(len(outcomes)) + offsets[scheme]
        point = part["delta_r2"].clip(x_low, x_high).to_numpy()
        low = part["delta_r2_family_low"].clip(x_low, x_high).to_numpy()
        high = part["delta_r2_family_high"].clip(x_low, x_high).to_numpy()
        ax.hlines(y, low, high, color=colors[scheme], linewidth=1.3)
        ax.plot(point, y, "o", color=colors[scheme], markersize=5, label=labels[scheme], zorder=3)
        for row, position in zip(part.itertuples(), y):
            if row.delta_r2_family_low < x_low:
                ax.plot(x_low, position, "<", color=colors[scheme], markersize=5, clip_on=False)
            if row.delta_r2_family_high > x_high:
                ax.plot(x_high, position, ">", color=colors[scheme], markersize=5, clip_on=False)
    ax.axvline(0, color="#4d4d4d", linewidth=0.8)
    ax.set_xlim(x_low, x_high)
    ax.xaxis.grid(True, color="#d0d0d0", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_yticks(np.arange(len(outcomes)), [OUTCOME_LABELS[o] for o in outcomes])
    ax.invert_yaxis()
    ax.set_xlabel(r"Incremental out-of-sample fit from the adoption gap, $\Delta R^2_{\mathrm{OOS}}$" + "\nFamily-cluster 95% intervals; arrows mark intervals past the axis limit")
    ax.set_title("Adoption-gap contribution under three separations")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    save_figure(fig, OUT, "delta_r2_by_scheme")
    plt.close(fig)


def write_report(table: pd.DataFrame, summary: pd.DataFrame, influence: pd.DataFrame, tercile: pd.DataFrame, association: pd.DataFrame,
                 seed_stability: pd.DataFrame) -> None:
    show = summary.copy()
    show["predictor"] = show["predictor"].map(PREDICTOR_LABELS)
    gap_all = summary[summary["predictor"] == "adoption_gap"]
    gap_table = table[table["predictor"] == "adoption_gap"][
        ["outcome", "scheme", "delta_r2", "delta_r2_family_low", "delta_r2_family_high", "delta_mae", "delta_mae_benchmark_points"]]
    wide = tercile.pivot_table(index=["group", "outcome", "panel_size"], columns="gap_tercile", values="mean_error_points").reset_index()
    wide["error_ratio_high_over_low"] = wide["highest_gap"] / wide["lowest_gap"]
    influence_gap = influence.groupby(["group", "dropped_family"]).agg(
        variants_dropped=("variants_dropped", "first"), adoption_mass_dropped=("adoption_mass_dropped", "first"),
        mean_delta_r2=("delta_r2", "mean"), benchmarks_positive=("delta_r2", lambda s: int((s > 0).sum()))).reset_index()
    capability_ratio = wide[wide["group"] == "capability"]["error_ratio_high_over_low"].median()
    text = f"""# P1.1 Validation under model reuse, release {RELEASE}

## Summary

Mean incremental fit from the adoption gap, with family-cluster 95% intervals. Capability and safety are shown apart because
the safety population, with 29 variants, can send R2 far below zero and dominate a joint mean.

| Benchmarks | Separation | Mean delta R2 | 95% interval | Positive benchmarks | Interval above zero |
| --- | --- | --- | --- | --- | --- |
""" + "\n".join(
        f"| {r.scope} | {r.scheme} | {r.mean_delta_r2:.3f} | [{r.mean_delta_r2_low:.3f}, {r.mean_delta_r2_high:.3f}] "
        f"| {int(r.benchmarks_positive)}/{int(r.benchmarks)} | {int(r.benchmarks_ci_above_zero)}/{int(r.benchmarks)} |"
        for r in gap_all.itertuples()) + f"""

Within a panel size, the highest-gap third of capability panels has a median {capability_ratio:.2f} times the mean error of
the lowest-gap third (`error_by_gap_tercile.csv`).

The panel-bootstrap intervals in `robustness_by_benchmark.csv` reproduce the original method. Compare them with the family
interval on the same row to see how much the dependence between panels matters.

R2 is unstable in the family-disjoint and strategy schemes: a test set with little variance sends it far below zero, and a
very poor control model pushes delta R2 above one. Check `r2_base` before reading delta R2 there, and prefer the rank
association below, which needs no fitted model.

## Rank association between gap and error, within panel size

Positive means higher-gap panels have larger errors. `disjoint_families` uses panels from one side of a family split at a time.

{markdown_table(association)}

## Sampling noise of the design

Mean delta R2 when only the random panel draws change ({SEED_DRAWS} redraws, same population). A difference between two runs
smaller than this range says nothing about the coordinate.

{markdown_table(seed_stability)}

## All coordinates

{markdown_table(show)}

## Adoption gap by benchmark

{markdown_table(gap_table)}

## Influence of large families (leave one family out, size scheme)

{markdown_table(influence_gap)}

## Design

- Panels: {PER_CELL} per size-by-strategy cell, with the original seeds. Family-disjoint uses {FAMILY_PER_CELL} per cell and side over {FAMILY_SPLITS} random splits.
- Family-cluster bootstrap: {B_FAMILY} replicates. Each draws families with replacement, reweights the population, recomputes
  targets and benchmark means, redraws every panel, and reruns all three separations.
- `delta_mae` is the control model's MAE minus the augmented model's MAE, in standardized error, so positive means better.
  `delta_mae_benchmark_points` converts it to points of that benchmark.
- The safety population is small, with few families, so its intervals are wide and its family-disjoint splits are thin.
"""
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
