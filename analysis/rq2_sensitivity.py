#!/usr/bin/env python3
"""RQ2: how sensitive is the paper audit to the modelling choices?

Every alternative is applied to the historical audit, the one the paper reports: each paper against the variants
available by its cutoff. One choice changes at a time, and for each alternative we report

  ranking stability   Kendall tau between paper rankings by adoption gap, primary against alternative,
                      and the median and largest change in a paper's rank;
  conclusion          the median percentile of observed panels among same-size uniform reference panels,
                      and how many papers fall below the median reference.

Choices and alternatives:

  target       nine alternatives of P1.2 (downloads with other alpha, all-time downloads, log downloads, likes, uniform)
  geometry     topological, repository-count, adoption-weighted edge lengths (foundation cache)
  resolution   exact matches only, descendant expansion
  cutoff       the alternative cutoff rules of P1.3
  population   six taxonomy-defined populations and each paper's claim-conditioned population (appendix C.5)

Reference panels follow appendix C exactly (seed, order, 120 draws), so the primary row reproduces its percentiles,
and alternatives with the same support and size reuse the same panels.

Outputs in deliverables/<release>/RQ2_sensitivity/.
"""

from __future__ import annotations

from typing import Any, Dict, List

import common
from common import OUTPUT, deliverable, foundation, markdown_table
import rq_common as rq
import p1_2_target_sensitivity as p12

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import build_appendix_c as baseline  # noqa: E402

OUT = deliverable("RQ2_sensitivity")
CUTOFFS = ["verified_experiment", "first_public", "public_bound", "year_end", "best_available", "no_cutoff"]
CUTOFF_LABELS = {"verified_experiment": "Date stated in the paper", "first_public": "First public version",
                 "public_bound": "First version fitting the panel", "year_end": "End of publication year",
                 "best_available": "Best available date", "no_cutoff": "No cutoff"}
GEOMETRY_LABELS = {"topological": "Topological", "structural_repo_count": "Repository-count",
                   "adoption_weighted_ablation": "Adoption-weighted"}
POPULATION_LABELS = {"assistant_models": "Assistant models", "reasoning_models": "Reasoning models",
                     "code_models": "Code models", "base_models": "Base models",
                     "small_models_le_13b": "Models up to 13B", "large_models_ge_34b": "Models of 34B or more",
                     "claim_conditioned": "Claim-conditioned"}


def cutoff_mask(F, cutoff, facts, created, lookup) -> np.ndarray:
    """Variants available by the cutoff day, plus panel leaves whose own release passes (as P1.3)."""
    if cutoff is None or pd.isna(cutoff):
        return np.ones(F.n, dtype=bool)
    day = pd.Timestamp(cutoff).normalize()
    if day.tzinfo is None:
        day = day.tz_localize("UTC")
    mask = np.asarray((~pd.isna(created)) & (created.normalize() <= day), dtype=bool)
    for fact in facts:
        idx = lookup.get(str(fact["variant_id"]))
        release = fact.get("release_date")
        if idx is not None and pd.notna(release) and pd.Timestamp(release).normalize() <= day:
            mask[idx] = True
    return mask


def score(F, entries, support, n, position, target=None, lengths=None, structure=False) -> Dict[str, float]:
    """Adoption gap and its uniform-reference percentile; with `structure`, also breadth and structural imbalance.

    Breadth and structural imbalance do not depend on the adoption target or on the edge lengths used for the gap,
    so they are computed only for choices that change the support or the panel.
    """
    empty = {"adoption_gap": np.nan, "uniform_percentile": np.nan,
             "breadth": np.nan, "breadth_percentile": np.nan, "structural": np.nan, "structural_percentile": np.nan}
    if not entries or support.sum() == 0:
        return empty
    target = baseline.target_adoption(F, support) if target is None else target
    scorer = rq.GapScorer(F, target, lengths)
    observed = scorer.gap(*rq.sigma_parts(entries))
    support_idx = np.where(support)[0]
    draws = rq.uniform_references(support_idx, n, position)
    out = dict(empty)
    out.update(adoption_gap=observed, uniform_percentile=rq.percentile_no_better(observed, [scorer.gap_uniform(r) for r in draws]))
    if structure:
        pi_b = baseline.target_breadth(F, support)
        views = baseline.population_views(F, pi_b)
        weights = pi_b[support_idx]
        K = F.kernel()
        b_obs = baseline.breadth_soft(F, list(entries), pi_b, support_idx)
        d_obs = baseline.structural_linf(F, F.sigma(list(entries)), views)[0]
        b_ref, d_ref = [], []
        for r in draws:
            r = np.asarray(r, int)
            b_ref.append(float(K[np.ix_(support_idx, r)].max(axis=1) @ weights))
            sigma = np.bincount(r, minlength=F.n) / len(r)
            d_ref.append(baseline.structural_linf(F, sigma, views)[0])
        out.update(breadth=b_obs, breadth_percentile=float(np.mean(np.asarray(b_ref) <= b_obs)),
                   structural=d_obs, structural_percentile=float(np.mean(np.asarray(d_ref) >= d_obs)))
    return out


def main() -> None:
    F = foundation()
    contexts = rq.load_contexts()
    worst = rq.check_against_w1(F, contexts)
    positions = rq.paper_positions(F)
    papers = rq.scoreable(F)
    lookup = {str(v): i for i, v in enumerate(F.variant["variant_id"].astype(str))}
    meta = baseline.load_variant_metadata(F, common.Path(common.os.environ["COVERAGE_LINEAGE_FILE"]))
    created = pd.to_datetime(meta.creation_date, utc=True)
    scores = p12.build_scores(F)
    definitions = p12.target_definitions(F)
    populations = baseline.default_population_masks(F)
    claims = pd.read_csv(OUTPUT / "appendix_c" / "appendix_c5_claim_conditioned_per_paper.csv").set_index("paper_id")
    cutoffs = pd.read_csv(common.deliverable("P1_3_temporal") / "cutoffs_by_paper.csv").set_index("paper_id")

    rows: List[Dict[str, Any]] = []
    for pid in papers:
        panel = F.panels[pid]
        entries = panel["entries"]
        n = int(panel["n_resolved"])
        position = positions[pid]
        eligible = np.asarray(contexts[pid]["eligible"], bool)
        support_none = np.zeros(F.n, dtype=bool)

        def add(choice: str, alternative: str, label: str, values: Dict[str, float]) -> None:
            rows.append({"paper_id": pid, "choice": choice, "alternative": alternative, "label": label, **values})

        add("primary", "primary", "Primary specification", score(F, entries, eligible, n, position, structure=True))
        for name, definition in definitions.items():
            if name == "alpha_0.10":
                continue
            target = p12.make_target(definition, scores, eligible)
            add("target", name, definition["label"], score(F, entries, eligible, n, position, target=target))
        for name, label in GEOMETRY_LABELS.items():
            add("geometry", name, label, score(F, entries, eligible, n, position, lengths=F.geometries[name]["lengths"]))
        for regime, label in [("exact_only", "Exact matches only"), ("descendant_expansion", "Descendant expansion")]:
            alt = rq.regime_entries(entries, regime)
            add("resolution", regime, label, score(F, alt, eligible, len(alt), position, structure=True))
        for scheme in CUTOFFS:
            cutoff = None if scheme == "no_cutoff" else cutoffs.loc[pid, scheme] if pid in cutoffs.index else np.nan
            if scheme != "no_cutoff" and pd.isna(cutoff):
                add("cutoff", scheme, CUTOFF_LABELS[scheme], score(F, [], support_none, n, position))
                continue
            support = cutoff_mask(F, cutoff, contexts[pid]["facts"], created, lookup)
            add("cutoff", scheme, CUTOFF_LABELS[scheme], score(F, entries, support, n, position, structure=True))
        for name, info in populations.items():
            if name == "all_textual":
                continue
            support = eligible & np.asarray(info["mask"], bool)
            add("population", name, POPULATION_LABELS[name], score(F, entries, support, n, position, structure=True))
        chosen = str(claims.loc[pid, "selected_population_id"])
        support = eligible & np.asarray(populations[chosen]["mask"], bool)
        add("population", "claim_conditioned", POPULATION_LABELS["claim_conditioned"], score(F, entries, support, n, position, structure=True))

    per_paper = pd.DataFrame(rows)
    primary = per_paper[per_paper["choice"] == "primary"].set_index("paper_id")

    c4 = pd.read_csv(OUTPUT / "appendix_c" / "appendix_c4_complete_audit_per_paper.csv").set_index("paper_id")
    gap_check = float((primary["adoption_gap"] - c4.loc[primary.index, "observed_adoption_gap"]).abs().max())
    pct_check = float((primary["uniform_percentile"] - c4.loc[primary.index, "uniform_adoption_percentile"]).abs().max())
    breadth_check = float((primary["breadth_percentile"] - c4.loc[primary.index, "uniform_breadth_percentile"]).abs().max())
    structural_check = float((primary["structural_percentile"] - c4.loc[primary.index, "uniform_structural_percentile"]).abs().max())

    summary_rows = []
    for (choice, alternative), group in per_paper.groupby(["choice", "alternative"], sort=False):
        g = group.set_index("paper_id").dropna(subset=["adoption_gap"])
        common_ids = g.index.intersection(primary.index)
        base_rank = primary.loc[common_ids, "adoption_gap"].rank(method="average")
        alt_rank = g.loc[common_ids, "adoption_gap"].rank(method="average")
        shift = (alt_rank - base_rank).abs()
        tau = kendalltau(base_rank, alt_rank).statistic if len(common_ids) > 2 else np.nan
        summary_rows.append({
            "choice": choice, "alternative": alternative, "label": group["label"].iloc[0], "papers": len(common_ids),
            "kendall_tau": tau, "median_rank_shift": shift.median(), "max_rank_shift": shift.max(),
            "median_adoption_gap": g["adoption_gap"].median(),
            "median_uniform_percentile": g["uniform_percentile"].median(),
            "papers_below_half": int((g["uniform_percentile"] < 0.5).sum()),
            "median_breadth_percentile": g["breadth_percentile"].median(),
            "median_structural_percentile": g["structural_percentile"].median(),
        })
    summary = pd.DataFrame(summary_rows)
    per_paper.to_csv(OUT / "sensitivity_by_paper.csv", index=False)
    summary.to_csv(OUT / "sensitivity_summary.csv", index=False)

    by_choice = summary[summary["choice"] != "primary"].groupby("choice").agg(
        alternatives=("alternative", "count"), min_tau=("kendall_tau", "min"), max_tau=("kendall_tau", "max"),
        largest_shift=("max_rank_shift", "max"), min_median_percentile=("median_uniform_percentile", "min"),
        max_median_percentile=("median_uniform_percentile", "max")).reset_index()
    by_choice.to_csv(OUT / "sensitivity_by_choice.csv", index=False)

    show = summary.copy()
    report = f"""# RQ2 Sensitivity of the audit to modelling choices, release {common.RELEASE}

All alternatives are applied to the historical audit (each paper against the variants available by its cutoff),
changing one choice at a time. Checks: the fast adoption gap matches F.w1 to {worst:.1e}; the primary row matches
appendix C to {gap_check:.1e} in gap and, in uniform-reference percentiles, to {pct_check:.1e} (gap),
{breadth_check:.1e} (breadth), and {structural_check:.1e} (structural imbalance). Breadth and structural imbalance
do not depend on the target or on the gap's edge lengths, so they are computed only for the other choices.

`kendall_tau` and rank shifts compare paper rankings by adoption gap with the primary ranking, over the papers for
which the alternative is defined. `median_uniform_percentile` is the median share of 120 same-size uniform reference
panels whose gap is at least the observed one; `papers_below_half` counts papers below the median reference.

## By choice

{markdown_table(by_choice)}

## By alternative

{markdown_table(show[["choice", "label", "papers", "kendall_tau", "median_rank_shift", "max_rank_shift", "median_adoption_gap", "median_uniform_percentile", "papers_below_half", "median_breadth_percentile", "median_structural_percentile"]])}

## How to read this

- A change of target or population changes the estimand, so a lower tau there says the question changed, not that the
  method is unstable. Geometry, resolution, and cutoff are closer to measurement choices.
- Alternative cutoffs can leave some of a panel's models outside the eligible pool; their mass is then transported to
  the pool, which raises the gap.
- Descendant expansion scores a panel of every compatible variant, so its references are larger panels too.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
