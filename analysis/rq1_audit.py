#!/usr/bin/env python3
"""RQ1: how well do the panels of published evaluations cover the models available when each study was done?

Uses the historical audit of appendix C (each paper against the variants available by its cutoff) and adds:

  coverage_distribution.csv   distribution of every coordinate over the 46 panels, with panel concentration
  reference_percentiles.csv   distribution of the percentiles against the three same-size reference samplers
  structural_cells.csv        which partition and cell produce each panel's structural imbalance
  family_composition.csv      panel mass by family against adoption and breadth target mass, averaged over papers
  by_source.csv               medians by source stratum
  reference_bands.csv/.pdf    per paper, the observed gap against the 5-95% range of uniform and adoption-weighted
                              references (the same panels as appendix C)

Outputs in deliverables/<release>/RQ1_audit/.
"""

from __future__ import annotations

from typing import Any, Dict, List

import common
from common import OUTPUT, deliverable, foundation, markdown_table, save_figure
import rq_common as rq

import numpy as np
import pandas as pd

import build_appendix_c as baseline  # noqa: E402

OUT = deliverable("RQ1_audit")
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9893"
INK, MUTED = "#0b0b0b", "#52514e"


def stratum(venue: str) -> str:
    v = str(venue).lower()
    if any(k in v for k in ["acl", "emnlp", "naacl", "findings"]):
        return "ACL Anthology"
    if any(k in v for k in ["iclr", "icml", "neurips", "nips", "aaai"]):
        return "ML conferences"
    if "arxiv" in v or "preprint" in v:
        return "arXiv"
    return "Not recorded"


def main() -> None:
    F = foundation()
    contexts = rq.load_contexts()
    rq.check_against_w1(F, contexts)
    positions = rq.paper_positions(F)
    c4 = pd.read_csv(OUTPUT / "appendix_c" / "appendix_c4_complete_audit_per_paper.csv").set_index("paper_id")
    corpus = pd.read_csv(common.deliverable("P1_5_corpus") / "corpus_table.csv").set_index("paper_id")
    families = F.variant["family_id"].astype(str).to_numpy()

    rows: List[Dict[str, Any]] = []
    composition: List[pd.DataFrame] = []
    for pid in rq.scoreable(F):
        panel = F.panels[pid]
        entries = panel["entries"]
        eligible = np.asarray(contexts[pid]["eligible"], bool)
        support_idx = np.where(eligible)[0]
        n = int(panel["n_resolved"])
        sigma = F.sigma(entries)
        pi_a = baseline.target_adoption(F, eligible)
        pi_b = baseline.target_breadth(F, eligible)
        by_family = pd.DataFrame({"family": families, "panel": sigma, "adoption": pi_a, "breadth": pi_b}).groupby("family").sum()
        by_family = by_family[(by_family > 0).any(axis=1)]
        by_family["paper_id"] = pid
        composition.append(by_family.reset_index())
        panel_families = by_family[by_family["panel"] > 0]["panel"]

        scorer = rq.GapScorer(F, pi_a)
        uniform = np.array([scorer.gap_uniform(r) for r in rq.uniform_references(support_idx, n, positions[pid])])
        adoption = np.array([scorer.gap_uniform(r) for r in rq.adoption_references(support_idx, pi_a, n, positions[pid])])
        rows.append({
            "paper_id": pid, "n": n, "eligible": len(support_idx),
            "distinct_families": int((panel_families > 1e-12).sum()),
            "largest_family_share": float(panel_families.max()),
            "largest_family": str(panel_families.idxmax()),
            "uniform_p05": np.quantile(uniform, 0.05), "uniform_p50": np.median(uniform), "uniform_p95": np.quantile(uniform, 0.95),
            "adoption_p05": np.quantile(adoption, 0.05), "adoption_p50": np.median(adoption), "adoption_p95": np.quantile(adoption, 0.95),
        })
    bands = pd.DataFrame(rows).set_index("paper_id").join(c4.drop(columns=["n"]))
    bands["source"] = [stratum(corpus.loc[p, "venue"]) if p in corpus.index else "Not recorded" for p in bands.index]
    bands["year"] = [corpus.loc[p, "year"] if p in corpus.index else np.nan for p in bands.index]

    coords = {"n": "Panel size", "distinct_families": "Distinct families", "largest_family_share": "Largest family share",
              "observed_adoption_gap": "Adoption gap G_a", "observed_breadth": "Lineage breadth B",
              "observed_structural_linf": "Structural imbalance D_inf", "observed_coverage_q0": "C(0)",
              "observed_coverage_q1": "C(1)"}
    dist = pd.DataFrame([{"coordinate": label, "median": bands[col].median(), "q25": bands[col].quantile(0.25),
                          "q75": bands[col].quantile(0.75), "min": bands[col].min(), "max": bands[col].max()}
                         for col, label in coords.items()])
    dist.to_csv(OUT / "coverage_distribution.csv", index=False)

    pct_rows = []
    for sampler, label in [("adoption_weighted", "Adoption-weighted"), ("family_stratified", "Family-stratified"), ("uniform", "Uniform")]:
        for coordinate in ["adoption", "breadth", "structural"]:
            values = bands[f"{sampler}_{coordinate}_percentile"]
            pct_rows.append({"reference": label, "coordinate": coordinate, "median": values.median(),
                             "q25": values.quantile(0.25), "q75": values.quantile(0.75),
                             "papers_below_0.05": int((values < 0.05).sum()), "papers_below_0.5": int((values < 0.5).sum())})
    percentiles = pd.DataFrame(pct_rows)
    percentiles.to_csv(OUT / "reference_percentiles.csv", index=False)

    cells = (bands.groupby(["observed_structural_partition", "observed_structural_cell"]).size()
             .rename("papers").reset_index().sort_values("papers", ascending=False))
    partitions = bands["observed_structural_partition"].value_counts().rename("papers").reset_index()
    cells.to_csv(OUT / "structural_cells.csv", index=False)
    sign = (bands["observed_structural_signed_error"] > 0).sum()

    comp = pd.concat(composition)
    n_papers = comp["paper_id"].nunique()
    mean_comp = comp.groupby("family")[["panel", "adoption", "breadth"]].sum() / n_papers
    mean_comp["papers_with_family"] = comp[comp["panel"] > 1e-12].groupby("family")["paper_id"].nunique()
    mean_comp = mean_comp.fillna(0).sort_values("panel", ascending=False)
    top = mean_comp.head(8).copy()
    other = mean_comp.iloc[8:][["panel", "adoption", "breadth"]].sum()
    top.loc["all other families"] = [other["panel"], other["adoption"], other["breadth"], np.nan]
    adoption_top = mean_comp.sort_values("adoption", ascending=False).head(8)
    top.reset_index().to_csv(OUT / "family_composition.csv", index=False)
    adoption_top.reset_index().to_csv(OUT / "family_composition_by_adoption.csv", index=False)

    source = bands.groupby("source").agg(papers=("n", "size"), median_n=("n", "median"),
                                         median_gap=("observed_adoption_gap", "median"),
                                         median_uniform_gap_percentile=("uniform_adoption_percentile", "median"),
                                         median_uniform_breadth_percentile=("uniform_breadth_percentile", "median"),
                                         median_structural=("observed_structural_linf", "median")).reset_index()
    source.to_csv(OUT / "by_source.csv", index=False)
    bands.reset_index()[["paper_id", "source", "year", "n", "eligible", "distinct_families", "largest_family", "largest_family_share",
                         "observed_adoption_gap", "uniform_p05", "uniform_p50", "uniform_p95", "adoption_p05", "adoption_p50",
                         "adoption_p95", "uniform_adoption_percentile", "adoption_weighted_adoption_percentile",
                         "observed_breadth", "observed_structural_linf", "observed_structural_partition",
                         "observed_structural_cell", "observed_coverage_q1"]].to_csv(OUT / "reference_bands.csv", index=False)
    write_figure(bands)

    report = f"""# RQ1 Coverage of existing panels, release {common.RELEASE}

Historical audit: each of the {len(bands)} scoreable panels against the variants available by its cutoff.

## Distribution of the coordinates over papers

{markdown_table(dist, digits=4)}

## Percentiles against same-size references

Share of the 120 reference panels that the observed panel matches or beats (larger is better).

{markdown_table(percentiles)}

## Where the structural imbalance comes from

{markdown_table(partitions)}

Most frequent cells:

{markdown_table(cells.head(10))}

In {sign} of {len(bands)} papers the worst cell is over-represented by the panel (positive signed error).

## Panel mass by family against target mass, averaged over papers

`panel` is the mean share of panel mass, `adoption` and `breadth` the mean share of each target at the paper's cutoff.

{markdown_table(top.reset_index(), digits=3)}

Families with the most adoption mass:

{markdown_table(adoption_top.reset_index(), digits=3)}

## By source

{markdown_table(source)}
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


def write_figure(bands: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = bands.sort_values("observed_adoption_gap")
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(4.6, 5.6))
    ax.hlines(y + 0.18, order["uniform_p05"], order["uniform_p95"], color=GRAY, linewidth=1.8, label="Uniform references, 5–95%")
    ax.hlines(y - 0.18, order["adoption_p05"], order["adoption_p95"], color=ORANGE, linewidth=1.8, label="Adoption-weighted refs., 5–95%")
    ax.scatter(order["observed_adoption_gap"], y, s=20, color=BLUE, zorder=3, edgecolor="white", linewidth=0.6, label="Observed panel")
    ax.set_yticks(y)
    ax.set_yticklabels([f"n={int(v)}" for v in order["n"]], fontsize=6, color=MUTED)
    ax.set_ylim(-1, len(order))
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlabel("Adoption gap $G_a$ at the paper's cutoff (lower is better)", color=INK, fontsize=9)
    ax.set_ylabel("Papers, sorted by observed gap", color=INK, fontsize=9)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors=MUTED)
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, fontsize=7.5, frameon=False,
              handlelength=1.4, columnspacing=1.0)
    fig.tight_layout()
    save_figure(fig, OUT, "reference_bands")
    plt.close(fig)


if __name__ == "__main__":
    main()
