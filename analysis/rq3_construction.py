#!/usr/bin/env python3
"""RQ3: how far can coverage-aware construction improve a panel, under the constraints its authors faced?

Builds on the constraint ladder of f_feasibility.py (pools saved in paper_contexts.pkl) and adds two analyses:

  trade-off   the greedy construction maximizes lambda * (gap reduction) + (1 - lambda) * (breadth gain), each term
              scaled by what the pure adoption-greedy and breadth-greedy panels achieve for that paper and pool, for
              lambda from 1 (adoption only) to 0 (breadth only). Rungs: access-role-size, size-by-role composition,
              and all constraints. Reports the gap and breadth of each panel against the observed panel.
  swaps       starting from the observed panel, the best single replacements under the same constraints, keeping
              the replaced model's size bucket and role, for up to three swaps. Reports how much of the full
              greedy rebuild's gap reduction a few swaps achieve.

Outputs in deliverables/<release>/RQ3_construction/.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
from typing import Any, Dict, List

import common
from common import deliverable, markdown_table, save_figure
import rq_common as rq
import f_feasibility as ff

import numpy as np
import pandas as pd

import build_appendix_c as baseline  # noqa: E402

OUT = deliverable("RQ3_construction")
LAMBDAS = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0]
TRADEOFF_RUNGS = ["core", "cost", "adoption_floor"]
SWAP_RUNGS = ["cost", "adoption_floor"]
MAX_SWAPS = 3
RUNG_LABELS = {"core": "Access, role, max. size", "cost": "+ size-by-role composition", "adoption_floor": "All constraints"}
WORKERS = int(os.environ.get("RQ3_WORKERS", "16"))
CONTEXTS: Dict[str, Dict[str, Any]] = {}


def scores_of(F, selected, eligible) -> Dict[str, float]:
    s = baseline.score_entries(F, baseline.exact_entries(F, selected), eligible)
    return {"G": s["adoption_gap"], "B": s["breadth"], "D": s["structural_linf"], "C1": s["coverage_q1"]}


def tradeoff_job(pid: str) -> List[Dict[str, Any]]:
    F, meta, rep = ff.STATE["F"], ff.STATE["meta"], ff.STATE["rep"]
    ctx = CONTEXTS[pid]
    eligible = np.asarray(ctx["eligible"], bool)
    n = int(ctx["n"])
    facts = [SimpleNamespace(**f) for f in ctx["facts"]]
    observed = baseline.score_entries(F, ctx["entries"], eligible)
    G0, B0, D0 = observed["adoption_gap"], observed["breadth"], observed["structural_linf"]
    pi_a = baseline.target_adoption(F, eligible)
    pi_b = baseline.target_breadth(F, eligible)
    rows_population = np.where(eligible)[0]
    K = F.kernel()
    params = rep["params_b"].to_numpy(float)
    rows = []
    for rung in TRADEOFF_RUNGS:
        pool = np.asarray(ctx["pools"][rung], bool)
        if pool.sum() < n:
            continue
        adopt = ff.adoption_step(F, pi_a, n)
        breadth = ff.breadth_step(F, K, pi_b, rows_population, np.where(pool)[0])
        ends = {}
        for lam in [1.0, 0.0]:
            quota = ff.SlotQuota(meta, facts) if rung != "core" else None
            sel, _ = ff.greedy(F, n, pool, adopt if lam == 1.0 else breadth, quota)
            ends[lam] = scores_of(F, sel, eligible)
        g_range = max(G0 - ends[1.0]["G"], 1e-9)
        b_range = max(ends[0.0]["B"] - B0, 1e-12)
        for lam in LAMBDAS:
            if lam in ends and lam in (1.0, 0.0):
                quota = ff.SlotQuota(meta, facts) if rung != "core" else None
                step = adopt if lam == 1.0 else breadth
            else:
                quota = ff.SlotQuota(meta, facts) if rung != "core" else None

                def step(selected, lam=lam):
                    a = adopt(selected) / g_range
                    b = breadth(selected) / b_range
                    out = lam * a + (1.0 - lam) * np.where(np.isfinite(b), b, 0.0)
                    out[~np.isfinite(b)] = -np.inf
                    return out
            sel, relaxed = ff.greedy(F, n, pool, step, quota)
            s = scores_of(F, sel, eligible)
            rows.append({"paper_id": pid, "rung": rung, "lambda": lam, "n": n,
                         "observed_G": G0, "observed_B": B0, "observed_D": D0, **s,
                         "gap_reduction_pct": 100 * (G0 - s["G"]) / G0,
                         "breadth_gain_pct": 100 * (s["B"] - B0) / B0,
                         "structural_reduction": D0 - s["D"],
                         "param_ratio": float(np.nansum(params[sel]) / ctx["observed_params"]) if ctx["observed_params"] > 0 else np.nan,
                         "relaxed_slots": relaxed})
    return rows


def swap_job(pid: str) -> List[Dict[str, Any]]:
    F, meta, rep = ff.STATE["F"], ff.STATE["meta"], ff.STATE["rep"]
    ctx = CONTEXTS[pid]
    eligible = np.asarray(ctx["eligible"], bool)
    entries = [dict(e) for e in ctx["entries"]]
    n = len(entries)
    facts = [SimpleNamespace(**f) for f in ctx["facts"]]
    quota = ff.SlotQuota(meta, facts)
    pi_a = baseline.target_adoption(F, eligible)
    T = F.branch_mass(pi_a)
    w = 1.0 / n
    observed = baseline.score_entries(F, entries, eligible)
    out = []
    for rung in SWAP_RUNGS:
        pool = np.asarray(ctx["pools"][rung], bool)
        current = [dict(e) for e in entries]
        swapped = [False] * n
        gap = observed["adoption_gap"]
        out.append({"paper_id": pid, "rung": rung, "swaps": 0, "n": n, "G": gap, "B": observed["breadth"],
                    "D": observed["structural_linf"], "removed": "", "added": "", "added_repo": ""})
        for k in range(1, MAX_SWAPS + 1):
            sigma = F.sigma(current)
            in_panel = sigma > 1e-12
            best = None
            for j in range(n):
                if swapped[j]:
                    continue
                part = np.zeros(F.n)
                part[np.asarray(current[j]["candidate_idx"], int)] = np.asarray(current[j]["posterior"], float) / np.sum(current[j]["posterior"])
                counts = F.branch_mass(sigma - w * part)
                base = float(np.dot(F.lengths, np.abs(T - counts)))
                change = np.zeros(F.n)
                for depth in range(F.paths.shape[1]):
                    e = F.paths[:, depth]
                    change += F.lengths[e] * (np.abs(T[e] - counts[e] - w) - np.abs(T[e] - counts[e]))
                slot = quota.slots[j]
                allowed = pool & quota.allowed([slot]) & ~in_panel
                if not allowed.any():
                    continue
                c = int(np.argmin(np.where(allowed, change, np.inf)))
                new_gap = base + float(change[c])
                if best is None or new_gap < best[0]:
                    best = (new_gap, j, c)
            if best is None or best[0] >= gap - 1e-12:
                break
            gap, j, c = best
            removed = ";".join(str(F.variant.iloc[int(i)]["variant_id"]) for i in current[j]["candidate_idx"])
            current[j] = {"candidate_idx": np.array([c]), "posterior": np.array([1.0])}
            swapped[j] = True
            s = baseline.score_entries(F, current, eligible)
            out.append({"paper_id": pid, "rung": rung, "swaps": k, "n": n, "G": s["adoption_gap"], "B": s["breadth"],
                        "D": s["structural_linf"], "removed": removed, "added": str(F.variant.iloc[c]["variant_id"]),
                        "added_repo": str(rep.iloc[c]["repo_id"])})
    return out


def main() -> None:
    ff.init_state()
    CONTEXTS.update(rq.load_contexts())
    papers = [pid for pid in rq.scoreable(ff.STATE["F"]) if pid in CONTEXTS]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        tradeoff = pd.DataFrame([r for rows in pool.map(tradeoff_job, papers) for r in rows])
        swaps = pd.DataFrame([r for rows in pool.map(swap_job, papers) for r in rows])
    tradeoff.to_csv(OUT / "tradeoff_by_paper.csv", index=False)
    swaps.to_csv(OUT / "swaps_by_paper.csv", index=False)

    t = tradeoff.copy()
    t["both_improved"] = (t["G"] < t["observed_G"] - 1e-12) & (t["B"] > t["observed_B"] + 1e-12)
    t["structural_improved"] = t["structural_reduction"] > 1e-12
    trade_summary = t.groupby(["rung", "lambda"], sort=False).agg(
        papers=("paper_id", "nunique"), median_gap_reduction_pct=("gap_reduction_pct", "median"),
        median_breadth_gain_pct=("breadth_gain_pct", "median"), papers_both_improved=("both_improved", "sum"),
        papers_structural_improved=("structural_improved", "sum"), median_param_ratio=("param_ratio", "median")).reset_index()
    trade_summary.to_csv(OUT / "tradeoff_summary.csv", index=False)

    ladder = pd.read_csv(common.deliverable("F_feasibility") / "feasibility_ladder.csv")
    full = ladder.set_index(["paper_id", "rung"])["adoption_gap_reduction_pct"]
    s0 = swaps[swaps["swaps"] == 0].set_index(["paper_id", "rung"])
    swap_rows = []
    for rung in SWAP_RUNGS:
        for k in range(1, MAX_SWAPS + 1):
            reductions, fractions = [], []
            for pid in papers:
                if (pid, rung) not in s0.index:
                    continue
                g0 = s0.loc[(pid, rung), "G"]
                done = swaps[(swaps["paper_id"] == pid) & (swaps["rung"] == rung) & (swaps["swaps"] <= k)]
                gk = done.sort_values("swaps")["G"].iloc[-1]
                red = 100 * (g0 - gk) / g0
                reductions.append(red)
                greedy = full.get((pid, rung), np.nan)
                if np.isfinite(greedy) and greedy > 1e-9:
                    fractions.append(min(red / greedy, 10.0))
            swap_rows.append({"rung": rung, "swaps": k, "papers": len(reductions),
                              "median_gap_reduction_pct": float(np.median(reductions)),
                              "papers_improved": int(np.sum(np.asarray(reductions) > 1e-9)),
                              "median_share_of_greedy_rebuild": float(np.median(fractions)) if fractions else np.nan,
                              "papers_at_or_beyond_greedy": int(np.sum(np.asarray(fractions) >= 1.0))})
    swap_summary = pd.DataFrame(swap_rows)
    swap_summary.to_csv(OUT / "swaps_summary.csv", index=False)
    write_figure(trade_summary)

    report = f"""# RQ3 Coverage-aware construction, release {common.RELEASE}

## Trade-off between adoption gap and breadth

Greedy panels maximizing lambda * (scaled gap reduction) + (1 - lambda) * (scaled breadth gain), under each set of
constraints. Medians over papers, relative to the observed panel. `papers_both_improved` counts papers whose panel
has both a smaller gap and a larger breadth than the observed one.

{markdown_table(trade_summary)}

## Few swaps from the observed panel

Best replacements under the same constraints, each keeping the replaced model's size bucket and role.
`median_share_of_greedy_rebuild` compares the swap reduction with the full greedy rebuild of the ladder (capped at 10).

{markdown_table(swap_summary)}

## Notes

- Lambda scales each objective by what the pure adoption-greedy and breadth-greedy panels achieve for that paper and
  pool, so 0.5 weighs the two relative improvements equally.
- Swaps never add a variant already in the panel and stop when no swap reduces the gap.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


def write_figure(summary: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"core": "#9a9893", "cost": "#2a78d6", "adoption_floor": "#eb6834"}
    markers = {"core": "s", "cost": "o", "adoption_floor": "^"}
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.axhline(0, color="#b5b4ae", linewidth=0.8)
    ax.axvline(0, color="#b5b4ae", linewidth=0.8)
    for rung in TRADEOFF_RUNGS:
        part = summary[summary["rung"] == rung].sort_values("lambda")
        ax.plot(part["median_gap_reduction_pct"], part["median_breadth_gain_pct"], color=colors[rung], linewidth=2.0,
                marker=markers[rung], markersize=6, markeredgecolor="white", markeredgewidth=0.8, label=RUNG_LABELS[rung])
        for _, r in part.iterrows():
            if r["lambda"] in (1.0, 0.5, 0.0):
                ax.annotate(f"λ={r['lambda']:g}", (r["median_gap_reduction_pct"], r["median_breadth_gain_pct"]),
                            textcoords="offset points", xytext=(5, 4), fontsize=6.5, color="#52514e")
    ax.set_xlabel("Median adoption-gap reduction (%)", color="#0b0b0b")
    ax.set_ylabel("Median breadth gain (%)", color="#0b0b0b")
    ax.grid(color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, OUT, "tradeoff")
    plt.close(fig)


if __name__ == "__main__":
    main()
