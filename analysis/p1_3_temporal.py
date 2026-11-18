#!/usr/bin/env python3
"""P1.3: how much do the temporal choices matter?

The audit needs two things that history does not fully give it: when each experiment ran, and how
much each model was used at that time. This script separates them:

  Cutoff, under several rules:
    panel_newest         release of the newest model in the panel (the rule used so far)
    verified_experiment  a date the paper itself states, checked by reading the passage
    first_public         the paper's first public date
    public_bound         first public date, or the latest arXiv version when the panel holds newer models
    year_end             December 31 of the publication year, the loosest rule
    best_available       verified_experiment, else public_bound, else panel_newest

  Analysis type:
    historical     uses only what existed at the cutoff: which models were available and their structure
    retrospective  uses adoption observed at the snapshot, restricted to models available at the cutoff

Historical download counts do not exist, so no analysis here measures adoption at the time of a paper. With two
snapshots, the script also measures how far adoption moves between them, which bounds how date-dependent the retrospective
results are.

Outputs in deliverables/<release>/P1_3_temporal/. Needs paper_contexts.pkl from f_feasibility.py.
"""

from __future__ import annotations

import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import common
from common import OUTPUT, RELEASE, deliverable, foundation, markdown_table, rank_of, save_figure

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

import build_appendix_c as baseline  # noqa: E402
import release_config  # noqa: E402

ROOT = common.ROOT
OUT = deliverable("P1_3_temporal")
DATES = ROOT / "data/paper_dates.csv"
VERIFIED = ROOT / "data/paper_experiment_dates_verified.csv"
SCHEMES = ["panel_newest", "verified_experiment", "first_public", "public_bound", "year_end", "best_available", "no_cutoff"]
LABELS = {"panel_newest": "Newest model in panel", "verified_experiment": "Date stated in the paper", "first_public": "First public date", "public_bound": "First version that fits the panel",
          "year_end": "End of publication year", "best_available": "Best available date", "no_cutoff": "No cutoff"}
STATE: Dict[str, Any] = {}


def cutoff_table(contexts: Dict[str, Any], metadata: pd.DataFrame) -> pd.DataFrame:
    dates = pd.read_csv(DATES) if DATES.exists() else pd.DataFrame(columns=["paper_id"])
    verified = pd.read_csv(VERIFIED) if VERIFIED.exists() else pd.DataFrame(columns=["paper_id", "experiment_date"])
    rows = []
    for pid, context in contexts.items():
        row: Dict[str, Any] = {"paper_id": pid, "panel_newest": context["cutoff"]}
        match = dates[dates["paper_id"] == pid]
        if len(match) and bool(match.iloc[0].get("arxiv_match", False)):
            row["first_public"] = pd.to_datetime(match.iloc[0]["first_public_date"], utc=True, errors="coerce")
            row["latest_version"] = pd.to_datetime(match.iloc[0]["last_updated_date"], utc=True, errors="coerce")
            row["arxiv_id"] = match.iloc[0].get("arxiv_id")
            # A paper first posted before its newest model must have added that model in a later version.
            row["public_bound"] = row["first_public"] if row["first_public"] >= context["cutoff"] else row["latest_version"]
        stated = verified[verified["paper_id"] == pid]
        if len(stated):
            row["verified_experiment"] = pd.to_datetime(stated.iloc[0]["experiment_date"], utc=True, errors="coerce")
            row["verified_quote"] = stated.iloc[0].get("quote", "")
            row["verified_precision"] = stated.iloc[0].get("precision", "")
        year = pd.to_numeric(metadata.loc[pid].get("year") if pid in metadata.index else np.nan, errors="coerce")
        if pd.notna(year):
            row["year_end"] = pd.Timestamp(year=int(year), month=12, day=31, tz="UTC")
        best = row.get("verified_experiment")
        source = "verified_experiment"
        if best is None or pd.isna(best):
            best, source = row.get("public_bound"), "public_bound"
        if best is None or pd.isna(best):
            best, source = row["panel_newest"], "panel_newest"
        row["best_available"], row["best_available_source"] = best, source
        rows.append(row)
    return pd.DataFrame(rows)


def mask_for(cutoff, context, created) -> np.ndarray:
    """Variants available by the cutoff, keeping panel leaves whose own release passes."""
    if cutoff is None or pd.isna(cutoff):
        return np.array([])
    # Availability is decided by day, as in f_feasibility.py.
    cutoff_day = pd.Timestamp(cutoff).normalize()
    mask = np.asarray((~pd.isna(created)) & (created.normalize() <= cutoff_day), dtype=bool)
    lookup = STATE["lookup"]
    for fact in context["facts"]:
        idx = lookup.get(fact["variant_id"])
        if idx is not None and pd.notna(fact["release_date"]) and pd.Timestamp(fact["release_date"]).normalize() <= cutoff_day:
            mask[idx] = True
    return mask


def score(job) -> List[Dict[str, Any]]:
    pid, cutoffs = job
    F, meta, contexts = STATE["F"], STATE["meta"], STATE["contexts"]
    context = contexts[pid]
    created = pd.to_datetime(meta.creation_date, utc=True)
    n = context["n"]
    entries = context["entries"]
    sigma = F.sigma(entries)
    rows = []
    for scheme in SCHEMES:
        cutoff = cutoffs.get(scheme) if scheme != "no_cutoff" else None
        if scheme == "no_cutoff":
            eligible = np.ones(F.n, dtype=bool)
        else:
            eligible = mask_for(cutoff, context, created)
            if eligible.size == 0:
                continue
        if eligible.sum() < n:
            rows.append({"paper_id": pid, "scheme": scheme, "cutoff": cutoff, "status": "too_few_models_available", "eligible_variants": int(eligible.sum())})
            continue
        scores = baseline.score_entries(F, entries, eligible)
        uniform = eligible.astype(float) / eligible.sum()
        panel_after_cutoff = sum(pd.notna(f["release_date"]) and cutoff is not None and pd.Timestamp(f["release_date"]).normalize() > pd.Timestamp(cutoff).normalize() for f in context["facts"])
        selected = baseline.greedy_adoption_support(F, n, baseline.target_adoption(F, eligible), np.where(eligible)[0])
        greedy_gap = F.w1(baseline.target_adoption(F, eligible), np.bincount(selected, minlength=F.n) / n)
        selected_uniform = baseline.greedy_adoption_support(F, n, uniform, np.where(eligible)[0])
        uniform_gap = F.w1(uniform, sigma)
        rows.append({
            "paper_id": pid, "scheme": scheme, "cutoff": cutoff, "status": "ok", "eligible_variants": int(eligible.sum()),
            "panel_models_after_cutoff": int(panel_after_cutoff), "panel_mass_available": scores["panel_mass_in_population"],
            # Retrospective: adoption comes from the snapshot, after the paper.
            "retro_adoption_gap": scores["adoption_gap"], "retro_C1": scores["coverage_q1"],
            "retro_greedy_reduction_pct": 100.0 * (scores["adoption_gap"] - greedy_gap) / scores["adoption_gap"],
            # Historical: only availability and structure at the cutoff.
            "hist_breadth": scores["breadth"], "hist_structural": scores["structural_linf"], "hist_C0": scores["coverage_q0"],
            "hist_uniform_gap": uniform_gap,
            "hist_greedy_reduction_pct": 100.0 * (uniform_gap - F.w1(uniform, np.bincount(selected_uniform, minlength=F.n) / n)) / uniform_gap,
        })
    return rows


def snapshot_drift(F, contexts) -> Dict[str, Any]:
    """Adoption of the same repositories in the June snapshot, mapped to this release's taxonomy.

    data/june2026_core_downloads.parquet holds repo_id and downloads_30d of the core population of the
    2026-06-03 snapshot (the only two columns this comparison needs; see data/README.md).
    """
    if RELEASE == "june2026":
        return {}
    june_file = ROOT / "data" / "june2026_core_downloads.parquet"
    now_file = Path(os.environ["COVERAGE_LINEAGE_FILE"])
    june = pd.read_parquet(june_file, columns=["repo_id", "downloads_30d"]).rename(columns={"downloads_30d": "june"})
    now = pd.read_parquet(now_file, columns=["repo_id", "variant_id", "downloads_30d", "created_at", "is_multimodal_leak"])
    now = now[~now["is_multimodal_leak"].fillna(False).astype(bool)]
    merged = now.merge(june, on="repo_id", how="left")
    merged["june"] = pd.to_numeric(merged["june"], errors="coerce").fillna(0.0)
    merged["downloads_30d"] = pd.to_numeric(merged["downloads_30d"], errors="coerce").fillna(0.0)
    both = merged[merged["repo_id"].isin(set(june["repo_id"]))]
    by_variant = merged.groupby("variant_id")["june"].sum().reindex(F.variant["variant_id"].astype(str)).fillna(0.0).to_numpy(float)
    now_variant = F.DOWNLOADS
    keep = by_variant > 0
    pi_june = 0.9 * by_variant / by_variant.sum() + 0.1 / F.n
    top = lambda values, k: set(np.argsort(-values)[:k])
    gaps_now, gaps_june = [], []
    for context in contexts.values():
        sigma = F.sigma(context["entries"])
        gaps_now.append(F.w1(F.PI_A, sigma))
        gaps_june.append(F.w1(pi_june, sigma))
    return {
        "repositories_in_both": int(len(both)), "repositories_new_since_june": int(len(merged) - len(both)),
        "new_repositories_download_share": float(merged.loc[~merged["repo_id"].isin(set(june["repo_id"])), "downloads_30d"].sum() / max(merged["downloads_30d"].sum(), 1)),
        "repo_spearman_june_vs_now": float(spearmanr(both["june"], both["downloads_30d"]).statistic),
        "variant_spearman_june_vs_now": float(spearmanr(by_variant[keep], now_variant[keep]).statistic),
        "top_100_variant_overlap": len(top(by_variant, 100) & top(now_variant, 100)),
        "total_variation_june_vs_now": float(0.5 * np.abs(pi_june - F.PI_A).sum()),
        "paper_rank_kendall_june_vs_now_adoption": float(kendalltau(gaps_now, gaps_june).statistic),
    }


def main() -> None:
    F = foundation()
    meta = baseline.load_variant_metadata(F, Path(os.environ["COVERAGE_LINEAGE_FILE"]))
    contexts = pickle.load((OUTPUT / "paper_contexts.pkl").open("rb"))
    metadata = F.data["paper_metadata"].set_index("paper_id", drop=False)
    STATE.update(F=F, meta=meta, contexts=contexts, lookup={str(v): i for i, v in enumerate(F.variant["variant_id"].astype(str))})
    cutoffs = cutoff_table(contexts, metadata)
    jobs = [(row["paper_id"], {s: row.get(s) for s in SCHEMES}) for row in cutoffs.to_dict("records")]
    with ProcessPoolExecutor(max_workers=16) as pool:
        rows = [row for chunk in pool.map(score, jobs) for row in chunk]
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "temporal_by_paper.csv", index=False)
    export = cutoffs.copy()
    for column in ["panel_newest", "first_public", "latest_version", "public_bound", "verified_experiment", "year_end", "best_available"]:
        if column in export:
            export[column] = pd.to_datetime(export[column], utc=True, errors="coerce").dt.date
    if {"panel_newest", "first_public"} <= set(export.columns):
        gap_days = pd.to_datetime(export["first_public"], errors="coerce") - pd.to_datetime(export["panel_newest"], errors="coerce")
        export["days_from_newest_model_to_first_public"] = gap_days.dt.days
    export.to_csv(OUT / "cutoffs_by_paper.csv", index=False)

    ok = results[results["status"] == "ok"]
    base = ok[ok["scheme"] == "panel_newest"].set_index("paper_id")
    summary_rows = []
    for scheme in SCHEMES:
        part = ok[ok["scheme"] == scheme].set_index("paper_id")
        if not len(part):
            continue
        shared = part.index.intersection(base.index)
        row: Dict[str, Any] = {
            "scheme": scheme, "label": LABELS[scheme], "papers": len(part), "median_eligible": part["eligible_variants"].median(),
            "papers_with_models_after_cutoff": int((part["panel_models_after_cutoff"] > 0).sum()),
            "median_panel_mass_available": part["panel_mass_available"].median(),
        }
        for column, kind in [("retro_adoption_gap", "retrospective"), ("hist_uniform_gap", "historical"), ("hist_breadth", "historical"), ("hist_structural", "historical")]:
            row[f"median_{column}"] = part[column].median()
            row[f"tau_{column}"] = float(kendalltau(part.loc[shared, column], base.loc[shared, column]).statistic) if len(shared) > 2 else np.nan
        row["median_retro_greedy_reduction_pct"] = part["retro_greedy_reduction_pct"].median()
        row["median_hist_greedy_reduction_pct"] = part["hist_greedy_reduction_pct"].median()
        shift = np.abs(rank_of(part.loc[shared, "retro_adoption_gap"].to_numpy()) - rank_of(base.loc[shared, "retro_adoption_gap"].to_numpy()))
        row["median_rank_shift"] = float(np.median(shift))
        row["max_rank_shift"] = float(shift.max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "temporal_summary.csv", index=False)

    # Does the gap penalize older papers? Age is measured back from the snapshot.
    age_rows = []
    snapshot = pd.to_datetime(meta.creation_date, utc=True).max()
    for scheme in ["panel_newest", "no_cutoff"]:
        part = ok[ok["scheme"] == scheme].set_index("paper_id")
        ages = (snapshot - pd.to_datetime(base["cutoff"], utc=True)).dt.days.reindex(part.index)
        for column in ["retro_adoption_gap", "hist_uniform_gap", "hist_breadth"]:
            rho = spearmanr(ages, part[column], nan_policy="omit")
            age_rows.append({"scheme": scheme, "metric": column, "spearman_with_paper_age": float(rho.statistic), "p_value": float(rho.pvalue)})
    age = pd.DataFrame(age_rows)
    age.to_csv(OUT / "metric_vs_paper_age.csv", index=False)

    drift = snapshot_drift(F, contexts)
    pd.Series(drift, dtype=object).to_csv(OUT / "adoption_drift_between_snapshots.csv", header=["value"])

    write_figure(export, ok, base)
    write_report(summary, age, export, drift)
    print((OUT / "REPORT.md").read_text(encoding="utf-8"))


def write_figure(cutoffs: pd.DataFrame, ok: pd.DataFrame, base: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    if "first_public" in cutoffs:
        part = cutoffs.dropna(subset=["first_public"]).copy()
        x = pd.to_datetime(part["panel_newest"])
        y = pd.to_datetime(part["first_public"])
        ax.scatter(x, y, s=28, color="#1f77b4", edgecolor="white", linewidth=0.4, zorder=3, label="Paper")
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([low, high], [low, high], color="#4d4d4d", linewidth=0.8, linestyle="--", label="Same date")
        ax.set_xlabel("Release of the newest model in the panel")
        ax.set_ylabel("First public date of the paper")
        ax.grid(True, color="#e6e6e6", linewidth=0.5)
        ax.legend(frameon=False, fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    ax.set_title("Newest model against first public date. Below the line, a later version added the model", fontsize=9)

    ax = axes[1]
    colors = {"verified_experiment": "#2a9d4b", "public_bound": "#1f77b4", "year_end": "#d95f02", "no_cutoff": "#7f7f7f"}
    for scheme, color in colors.items():
        part = ok[ok["scheme"] == scheme].set_index("paper_id")
        shared = part.index.intersection(base.index)
        if len(shared):
            ax.scatter(base.loc[shared, "retro_adoption_gap"], part.loc[shared, "retro_adoption_gap"], s=22, color=color, edgecolor="white",
                       linewidth=0.3, label=LABELS[scheme], zorder=3)
    limits = [ok["retro_adoption_gap"].min(), ok["retro_adoption_gap"].max()]
    ax.plot(limits, limits, color="#4d4d4d", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Adoption gap, cutoff at newest model in panel")
    ax.set_ylabel("Adoption gap under the other cutoff")
    ax.grid(True, color="#e6e6e6", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Sensitivity of the adoption gap to the cutoff", fontsize=9)
    fig.tight_layout()
    save_figure(fig, OUT, "temporal_sensitivity")
    plt.close(fig)


def write_report(summary: pd.DataFrame, age: pd.DataFrame, cutoffs: pd.DataFrame, drift: Dict[str, Any]) -> None:
    n_public = int(cutoffs["first_public"].notna().sum()) if "first_public" in cutoffs else 0
    n_verified = int(cutoffs["verified_experiment"].notna().sum()) if "verified_experiment" in cutoffs else 0
    negative = int((cutoffs["days_from_newest_model_to_first_public"] < 0).sum()) if "days_from_newest_model_to_first_public" in cutoffs else 0
    median_lag = cutoffs["days_from_newest_model_to_first_public"].median() if "days_from_newest_model_to_first_public" in cutoffs else float("nan")
    retro = summary[["label", "papers", "median_eligible", "median_retro_adoption_gap", "tau_retro_adoption_gap", "median_retro_greedy_reduction_pct",
                     "median_rank_shift", "max_rank_shift", "papers_with_models_after_cutoff"]]
    hist = summary[["label", "papers", "median_hist_uniform_gap", "tau_hist_uniform_gap", "median_hist_breadth", "tau_hist_breadth",
                    "median_hist_structural", "tau_hist_structural", "median_hist_greedy_reduction_pct"]]
    drift_block = markdown_table(pd.Series(drift, dtype=object).rename("value").reset_index().rename(columns={"index": "measure"})) if drift else (
        "Requires a second snapshot. It runs once this analysis uses a release other than June 2026.")
    text = f"""# P1.3 Temporal reconstruction, release {RELEASE}

## Summary

Cutoffs available: first public date for {n_public} of {len(cutoffs)} papers, and a date stated in the paper for {n_verified}.
The median paper appeared {median_lag:.0f} days after the newest model in its panel. In {negative} papers the first public date precedes that
model's recorded release, usually because a later version added models, so first public date is not an upper bound on experiment
time. `public_bound` uses the latest version for those papers. Both dates are in `cutoffs_by_paper.csv`.

## Retrospective analysis

Adoption comes from the snapshot, after the paper, restricted to models available at the cutoff. It answers how well the
panel represents, among what was available, the models adopted now. It does not measure adoption at the time of the paper.

{markdown_table(retro)}

## Historical analysis

Uses only what existed at the cutoff: availability and structure. A uniform target over available models replaces adoption,
so these coordinates carry no information from after the paper.

{markdown_table(hist)}

`tau` compares paper rankings with the newest-model cutoff. The stated-date row covers only the papers with one, so its tau rests on few papers.

## Do older papers look worse?

Spearman correlation between paper age and each coordinate. A positive value on the adoption gap means older panels look less
representative, which would partly reflect adoption moving to newer models after they were published.

{markdown_table(age)}

## Adoption drift between snapshots

{drift_block}

## How to read this

- The newest-model cutoff is mechanically tied to the panel: a paper that evaluated a recent model gets a larger comparison pool.
  That is the reason to compare it with cutoffs set independently of the panel.
- `public_bound` is an upper bound on when experiments ran, so its pool may include models the authors could not have used. arXiv reports
  only the first and latest versions, so for updated papers the bound can be loose.
- Repository creation dates approximate model release, so reuploads and renames can shift them.
- Stated dates come from `data/paper_experiment_dates_verified.csv`, where each row quotes the passage it relies on.
"""
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
