#!/usr/bin/env python3
"""RQ4: does lineage-based coverage track behavior measured independently on benchmarks?

Two earlier attempts (P1.1, incremental out-of-sample R2 of simulated panels; VALIDATION, gap-guided swaps of the
observed panels) were fragile. Both compare panels whose inclusion probabilities differ, so under a concentrated
adoption target a panel's error is driven mostly by whether it contains the few heavily downloaded variants (GPT-2
alone holds 22% of the capability target). That channel does not involve the lineage tree at all: diagnostic C
below shows that selecting low-gap panels helps exactly as much under a tree whose leaves were randomly permuted.
This analysis removes the channel by design.

Outcome data, never seen by any construction: the frozen Open LLM Leaderboard capability scores (six benchmarks),
for the variants whose leaderboard rows resolve uniquely (complete cases, as in common.load_outcomes). Every score is
standardized over the scored population. Safety (29 variants) is too small for these tests and is not used.

A. Lineage signal (does lineage distance predict behavioral similarity?).
   A1. Semivariance by lineage relation: for pairs whose deepest shared node is the base model (same family, series,
       and size; different role), the series, the family, or none, the mean of (z_x - z_y)^2 / 2 per benchmark. With
       standardized scores, 1 means no more similar than two random models. Null: 999 permutations of score vectors
       across variants (one-sided p, smaller than null). Repeated on scale-adjusted scores (minus the mean of the
       size bucket), to separate lineage from parameter scale.
   A2. Leave-one-out prediction of each variant's scores from its closest lineage relatives (the other members of
       the deepest node it shares with anyone), against the size-bucket mean and a permuted-tree null.

B. Fixed-inclusion panel designs (do lower gaps give more accurate benchmark conclusions?).
   Panels of n models are drawn by systematic sampling with inclusion proportional to a target t over a scored world
   W, along an ordering of W. Every design gives every variant the same expected multiplicity n t(i); designs differ
   only in how the panel is spread:
     blind          random order (lineage-blind; the reference)
     lineage        random depth-first order of the lineage tree (children shuffled at every level). Each subtree
                    is a contiguous block, so |sigma(L_e) - t(L_e)| < 1/n on every edge: the design lowers the gap
                    under every edge-length geometry, without changing any inclusion probability.
     scale          size-bucket order, random within (a metadata design that ignores lineage)
     scale_lineage  size bucket, then random depth-first lineage order within the bucket
   Truth: the t-weighted mean of each benchmark over W, standardized by the t-weighted SD. Targets: adoption pi_a
   (primary), breadth pi_b, uniform, all restricted to W as in the audit.
   Worlds: each of the 46 audited papers (W = scored variants available by its cutoff, n = its resolved panel size,
   R_PAPER draws), and the full scored population with n in GLOBAL_SIZES (R_GLOBAL draws), with a family-cluster
   bootstrap of the population (B_BOOT resamples of the families, R_BOOT draws each) for intervals.

   Primary endpoint, fixed before the final run: paper worlds, adoption target, ratio of RMSE of the panel mean
   (lineage / blind), geometric mean over the six benchmarks; reported as the median over papers, the number of
   papers below one, and the family-bootstrap interval of the same ratio in the full population.
   Secondary: other targets and designs, the rate of panels missing a population mean by more than 0.5 SD, and the
   gap of each design.

C. Diagnostic of the earlier designs: among K iid draws with probability t, take the one with the smallest W1 under
   the real tree, under a tree with randomly permuted leaves (placebo), or the largest (anti), and compare the RMSE
   with a random pick.

Outputs in deliverables/<release>/RQ4_behavioral/.
"""

from __future__ import annotations

import json
import time
import zlib
from typing import Dict, List, Tuple

import common
from common import deliverable, markdown_table
import rq_common as rq

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wilcoxon

import build_appendix_c as baseline  # noqa: E402

OUT = deliverable("RQ4_behavioral")
SEED = 20261116
N_PERM = 999
R_PAPER = 2000
R_GLOBAL = 4000
GLOBAL_SIZES = [5, 10, 20, 40]
B_BOOT = 200
R_BOOT = 500
K_REJECT = 20
R_REJECT = 3000
LARGE_ERROR = 0.5
CHUNK = 1000
DESIGNS = ["blind", "lineage", "scale", "scale_lineage"]
TARGETS = ["adoption", "breadth", "uniform"]
SIZE_ORDER = ["<1B", "1B-3B", "3B-7B", "7B-13B", "13B-34B", "34B-70B", "70B-180B", "180B+", "unknown"]
RELATIONS = ["same base", "same series", "same family", "different family"]


# ----------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------

def load() -> Dict:
    F = common.foundation()
    support, outcomes = common.load_outcomes()["capability"]
    support = np.asarray(support, int)
    Y = outcomes.to_numpy(float)
    Z = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    size_rank = (F.variant["size_bucket"].astype(str).map({s: i for i, s in enumerate(SIZE_ORDER)})
                 .fillna(len(SIZE_ORDER) - 1).to_numpy(int))
    return {"F": F, "support": support, "Z": Z, "names": list(outcomes.columns), "size_rank": size_rank}


class World:
    """A scored world: standardized scores, lineage codes, size ranks, and the three targets.

    `members` are variant indices (repeated after a family bootstrap); `copy` separates the copies of a family so
    that each copy is its own branch of the tree."""

    def __init__(self, D: Dict, members: np.ndarray, copy: np.ndarray | None = None) -> None:
        F = D["F"]
        self.members = np.asarray(members, int)
        m = len(self.members)
        copy = np.zeros(m, int) if copy is None else np.asarray(copy, int)
        pos = pd.Series(np.arange(len(D["support"])), index=D["support"])
        self.Z = D["Z"][pos.loc[self.members].to_numpy()]
        paths = F.paths[self.members]
        self.codes = np.column_stack([pd.factorize(pd.Series(list(zip(paths[:, lvl], copy))))[0] for lvl in range(3)])
        self.size_rank = D["size_rank"][self.members]
        self.paths = paths
        self.m = m
        d = F.DOWNLOAD_PROB[self.members]
        adoption = d / d.sum() if d.sum() > 0 else np.full(m, 1.0 / m)
        pa = (1.0 - F.ALPHA) * adoption + F.ALPHA / m
        pb = F.PI_B[self.members]
        self.t = {"adoption": pa / pa.sum(), "breadth": pb / pb.sum(), "uniform": np.full(m, 1.0 / m)}
        self.truth = {}
        for name, t in self.t.items():
            mu = t @ self.Z
            sd = np.sqrt(t @ (self.Z - mu) ** 2)
            self.truth[name] = (mu, sd)
        self._incidence = None

    def incidence(self, lengths: np.ndarray) -> Tuple[sparse.csr_matrix, np.ndarray]:
        edges, inverse = np.unique(self.paths.ravel(), return_inverse=True)
        rows = np.repeat(np.arange(self.m), self.paths.shape[1])
        A = sparse.csr_matrix((np.ones(len(rows)), (rows, inverse)), shape=(self.m, len(edges)))
        return A, lengths[edges]


# ----------------------------------------------------------------------------------------------------------------
# A. Lineage signal
# ----------------------------------------------------------------------------------------------------------------

def within_sums(codes: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, float]:
    """Sum over unordered pairs sharing a code of (z_x - z_y)^2, per column, and the number of such pairs."""
    k = codes.max() + 1
    cnt = np.bincount(codes, minlength=k).astype(float)
    s1 = np.stack([np.bincount(codes, weights=Z[:, j], minlength=k) for j in range(Z.shape[1])], axis=1)
    s2 = np.stack([np.bincount(codes, weights=Z[:, j] ** 2, minlength=k) for j in range(Z.shape[1])], axis=1)
    return (cnt[:, None] * s2 - s1 ** 2).sum(axis=0), float((cnt * (cnt - 1) / 2).sum())


def semivariance(codes: np.ndarray, Z: np.ndarray) -> Dict[str, np.ndarray]:
    """Mean of (z_x - z_y)^2 / 2 per benchmark for the pairs of each exclusive lineage relation."""
    m = len(Z)
    total = (m * (Z ** 2).sum(axis=0) - Z.sum(axis=0) ** 2)
    n_total = m * (m - 1) / 2
    w = [within_sums(codes[:, lvl], Z) for lvl in (2, 1, 0)]  # base, series, family
    sums = [w[0][0], w[1][0] - w[0][0], w[2][0] - w[1][0], total - w[2][0]]
    pairs = [w[0][1], w[1][1] - w[0][1], w[2][1] - w[1][1], n_total - w[2][1]]
    return {rel: s / max(p, 1) / 2 for rel, s, p in zip(RELATIONS, sums, pairs)} | {
        f"pairs::{rel}": p for rel, p in zip(RELATIONS, pairs)}


def scale_adjust(Z: np.ndarray, size_rank: np.ndarray) -> np.ndarray:
    means = pd.DataFrame(Z).groupby(size_rank).transform("mean").to_numpy()
    return Z - means


def loo_group_mean(codes: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Leave-one-out mean of each element's group and whether the group has another member."""
    k = codes.max() + 1
    cnt = np.bincount(codes, minlength=k).astype(float)
    s = np.stack([np.bincount(codes, weights=Z[:, j], minlength=k) for j in range(Z.shape[1])], axis=1)
    others = cnt[codes] - 1
    pred = (s[codes] - Z) / np.maximum(others, 1)[:, None]
    return pred, others > 0


def closest_relatives(codes: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Predict each element from the other members of the deepest lineage node it shares with anyone."""
    m = len(Z)
    pred = (Z.sum(axis=0) - Z) / (m - 1)
    done = np.zeros(m, bool)
    for lvl in (2, 1, 0):
        p, ok = loo_group_mean(codes[:, lvl], Z)
        use = ok & ~done
        pred[use] = p[use]
        done |= ok
    return pred


def r2(Z: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return 1 - ((Z - pred) ** 2).sum(axis=0) / ((Z - Z.mean(axis=0)) ** 2).sum(axis=0)


def lineage_signal(D: Dict, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame]:
    W = World(D, D["support"])
    names = D["names"]
    rows = []
    for label, Z in [("raw", W.Z), ("scale-adjusted", scale_adjust(W.Z, W.size_rank))]:
        obs = semivariance(W.codes, Z)
        null = [semivariance(W.codes, Z[rng.permutation(W.m)]) for _ in range(N_PERM)]
        for rel in RELATIONS:
            nul = np.array([x[rel] for x in null])
            for j, b in enumerate(names + ["mean"]):
                o = obs[rel][j] if j < len(names) else obs[rel].mean()
                nu = nul[:, j] if j < len(names) else nul.mean(axis=1)
                rows.append({"scores": label, "relation": rel, "benchmark": b, "pairs": int(obs[f"pairs::{rel}"]),
                             "semivariance": o, "null_mean": nu.mean(), "null_p025": np.quantile(nu, 0.025),
                             "null_p975": np.quantile(nu, 0.975), "p_lower": (1 + np.sum(nu <= o)) / (N_PERM + 1)})
    semi = pd.DataFrame(rows)

    size_pred, _ = loo_group_mean(W.size_rank, W.Z)
    rel_pred = closest_relatives(W.codes, W.Z)
    resid = W.Z - size_pred
    combo = size_pred + closest_relatives(W.codes, resid)
    perm_r2 = []
    for _ in range(200):
        p = rng.permutation(W.m)
        perm_r2.append(r2(W.Z, closest_relatives(W.codes[p], W.Z)))
    perm_r2 = np.array(perm_r2)
    pred_rows = []
    for label, values in [("closest lineage relatives", r2(W.Z, rel_pred)),
                          ("size bucket", r2(W.Z, size_pred)),
                          ("size bucket + lineage relatives", r2(W.Z, combo)),
                          ("closest relatives, permuted tree (mean of 200)", perm_r2.mean(axis=0))]:
        for j, b in enumerate(names):
            pred_rows.append({"predictor": label, "benchmark": b, "loo_r2": values[j]})
        pred_rows.append({"predictor": label, "benchmark": "mean", "loo_r2": float(np.mean(values))})
    pred = pd.DataFrame(pred_rows)
    pred.loc[pred["predictor"].str.startswith("closest relatives, permuted"), "null_p975"] = list(
        np.quantile(perm_r2, 0.975, axis=0)) + [float(np.quantile(perm_r2.mean(axis=1), 0.975))]
    return semi, pred


# ----------------------------------------------------------------------------------------------------------------
# B. Fixed-inclusion designs
# ----------------------------------------------------------------------------------------------------------------

def orders(W: World, design: str, R: int, rng: np.random.Generator) -> np.ndarray:
    leaf = rng.random((R, W.m))
    if design == "blind":
        return np.argsort(leaf, axis=1)
    lineage = [rng.random((R, W.codes[:, lvl].max() + 1))[:, W.codes[:, lvl]] for lvl in (2, 1, 0)]
    size = np.broadcast_to(W.size_rank, (R, W.m))
    if design == "lineage":
        keys = [leaf, *lineage]
    elif design == "scale":
        keys = [leaf, size]
    elif design == "scale_lineage":
        keys = [leaf, *lineage, size]
    else:
        raise ValueError(design)
    return np.lexsort(keys, axis=-1)


def systematic(W: World, t: np.ndarray, order: np.ndarray, n: int, u: np.ndarray) -> np.ndarray:
    """Multiplicity of each world element (in world order) under systematic sampling along `order`."""
    c = np.cumsum(t[order], axis=1)
    c = c / c[:, -1:]
    hi = np.floor(n * c + u[:, None])
    lo = np.concatenate([np.floor(u)[:, None], hi[:, :-1]], axis=1)
    counts = hi - lo
    S = np.zeros_like(counts)
    np.put_along_axis(S, order, counts, axis=1)
    return S


def run_design(W: World, target: str, n: int, R: int, seed: List[int], with_gap: Dict | None = None) -> Dict:
    """Errors and gaps of every design for one world, target, and size. Designs share the random starts."""
    t = W.t[target]
    mu, sd = W.truth[target]
    out = {}
    for design in DESIGNS:
        errs, gaps, inc = [], [], np.zeros(W.m)
        for start in range(0, R, CHUNK):
            r = min(CHUNK, R - start)
            rng_u = np.random.default_rng(seed + [start])
            rng_o = np.random.default_rng(seed + [start, DESIGNS.index(design) + 1])
            S = systematic(W, t, orders(W, design, r, rng_o), n, rng_u.random(r))
            errs.append((S @ W.Z / n - mu) / sd)
            inc += S.sum(axis=0)
            if with_gap is not None:
                gaps.append({g: np.abs((S / n) @ A - t @ A) @ lam for g, (A, lam) in with_gap.items()})
        E = np.vstack(errs)
        res = {"E": E, "inclusion_dev": float(np.max(np.abs(inc / R - n * t)))}
        if gaps:
            res["gap"] = {g: np.concatenate([x[g] for x in gaps]) for g in with_gap}
        out[design] = res
    return out


def summarize(res: Dict) -> Dict[str, Dict[str, float]]:
    base = np.sqrt((res["blind"]["E"] ** 2).mean(axis=0))
    out = {}
    for design, r in res.items():
        rmse = np.sqrt((r["E"] ** 2).mean(axis=0))
        out[design] = {"rmse": float(rmse.mean()),
                       "ratio": float(np.exp(np.mean(np.log(rmse / base)))),
                       "large_error_rate": float(np.mean(np.abs(r["E"]) > LARGE_ERROR)),
                       **{f"rmse::{j}": float(v) for j, v in enumerate(rmse)},
                       **{f"ratio::{j}": float(v) for j, v in enumerate(rmse / base)}}
        if "gap" in r:
            for g, v in r["gap"].items():
                out[design][f"gap_{g}"] = float(v.mean())
    return out


def gap_geometries(F, W: World) -> Dict:
    geoms = {"primary": F.lengths, **{g: F.geometries[g]["lengths"] for g in ("topological",)}}
    return {g: W.incidence(lam) for g, lam in geoms.items()}


def global_designs(D: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    F = D["F"]
    W = World(D, D["support"])
    geo = gap_geometries(F, W)
    rows, bench = [], []
    for target in TARGETS:
        for n in GLOBAL_SIZES:
            res = run_design(W, target, n, R_GLOBAL, [SEED, 1, TARGETS.index(target), n], with_gap=geo)
            s = summarize(res)
            for design, v in s.items():
                rows.append({"target": target, "n": n, "design": design, "rmse": v["rmse"], "ratio": v["ratio"],
                             "large_error_rate": v["large_error_rate"], "gap_primary": v["gap_primary"],
                             "gap_topological": v["gap_topological"], "inclusion_dev": res[design]["inclusion_dev"]})
                for j, b in enumerate(D["names"]):
                    bench.append({"target": target, "n": n, "design": design, "benchmark": b,
                                  "rmse": v[f"rmse::{j}"], "ratio": v[f"ratio::{j}"]})
    return pd.DataFrame(rows), pd.DataFrame(bench)


def family_bootstrap(D: Dict) -> pd.DataFrame:
    F = D["F"]
    fam = F.paths[D["support"], 0]
    fams = np.unique(fam)
    members = {f: D["support"][fam == f] for f in fams}
    rng = np.random.default_rng([SEED, 2])
    rows = []
    for b in range(B_BOOT):
        pick = rng.choice(fams, len(fams), replace=True)
        idx = np.concatenate([members[f] for f in pick])
        copy = np.concatenate([np.full(len(members[f]), k) for k, f in enumerate(pick)])
        W = World(D, idx, copy)
        for target in TARGETS:
            for n in GLOBAL_SIZES:
                s = summarize(run_design(W, target, n, R_BOOT, [SEED, 3, b, TARGETS.index(target), n]))
                for design, v in s.items():
                    rows.append({"boot": b, "target": target, "n": n, "design": design, "ratio": v["ratio"],
                                 "large_error_rate": v["large_error_rate"]})
        if (b + 1) % 20 == 0:
            print(f"  family bootstrap {b + 1}/{B_BOOT}", flush=True)
    return pd.DataFrame(rows)


def paper_designs(D: Dict) -> pd.DataFrame:
    F = D["F"]
    contexts = rq.load_contexts()
    scored = np.zeros(F.n, bool)
    scored[D["support"]] = True
    rows = []
    for pid in rq.scoreable(F):
        members = np.where(np.asarray(contexts[pid]["eligible"], bool) & scored)[0]
        n = len(F.panels[pid]["entries"])
        if len(members) < 2 * n or n < 2:
            rows.append({"paper_id": pid, "n": n, "world": len(members), "included": False})
            continue
        W = World(D, members)
        geo = gap_geometries(F, W)
        for target in TARGETS:
            res = run_design(W, target, n, R_PAPER, [SEED, 4, zlib.crc32(pid.encode()), TARGETS.index(target)],
                             with_gap=geo)
            for design, v in summarize(res).items():
                rows.append({"paper_id": pid, "n": n, "world": W.m, "included": True, "target": target,
                             "design": design, "rmse": v["rmse"], "ratio": v["ratio"],
                             "large_error_rate": v["large_error_rate"], "gap_primary": v["gap_primary"],
                             "gap_topological": v["gap_topological"], "inclusion_dev": res[design]["inclusion_dev"]})
    return pd.DataFrame(rows)


def paper_summary(pp: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng([SEED, 5])
    d = pp[pp["included"]]
    rows = []
    for (target, design), g in d.groupby(["target", "design"]):
        if design == "blind":
            continue
        blind = d[(d["target"] == target) & (d["design"] == "blind")].set_index("paper_id")
        g = g.set_index("paper_id")
        ratio = g["ratio"].to_numpy()
        boot = [np.median(rng.choice(ratio, len(ratio))) for _ in range(5000)]
        logr = np.log(ratio)
        rows.append({"target": target, "design": design, "papers": len(g), "median_ratio": float(np.median(ratio)),
                     "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
                     "q25": float(np.quantile(ratio, 0.25)), "q75": float(np.quantile(ratio, 0.75)),
                     "papers_below_one": int((ratio < 1).sum()),
                     "wilcoxon_p": float(wilcoxon(logr).pvalue) if np.any(logr != 0) else np.nan,
                     "large_error_blind": float(blind["large_error_rate"].median()),
                     "large_error_design": float(g["large_error_rate"].median()),
                     "gap_ratio_primary": float(np.median(g["gap_primary"] / blind.loc[g.index, "gap_primary"])),
                     "gap_ratio_topological": float(np.median(g["gap_topological"] / blind.loc[g.index, "gap_topological"]))})
    order = {d: i for i, d in enumerate(DESIGNS)}
    out = pd.DataFrame(rows)
    out["_t"] = out["target"].map({t: i for i, t in enumerate(TARGETS)})
    out["_d"] = out["design"].map(order)
    return out.sort_values(["_t", "_d"]).drop(columns=["_t", "_d"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------
# C. Diagnostic of the earlier designs
# ----------------------------------------------------------------------------------------------------------------

def rejective(D: Dict) -> pd.DataFrame:
    F = D["F"]
    W = World(D, D["support"])
    rng = np.random.default_rng([SEED, 6])
    A, lam = W.incidence(F.lengths)
    placebos = []
    for _ in range(50):
        p = rng.permutation(W.m)
        Wp = World(D, D["support"])
        Wp.paths = W.paths[p]
        placebos.append(Wp.incidence(F.lengths))
    rows = []
    for target in ("adoption", "breadth"):
        t = W.t[target]
        mu, sd = W.truth[target]
        for n in (5, 10, 20):
            picks = {k: [] for k in ("random", "min gap", "min gap, permuted tree", "max gap")}
            for r in range(R_REJECT):
                draws = rng.choice(W.m, (K_REJECT, n), p=t)
                S = np.zeros((K_REJECT, W.m))
                np.add.at(S, (np.repeat(np.arange(K_REJECT), n), draws.ravel()), 1)
                E = (S @ W.Z / n - mu) / sd
                g = np.abs((S / n) @ A - t @ A) @ lam
                Ap, lp = placebos[r % len(placebos)]
                gp = np.abs((S / n) @ Ap - t @ Ap) @ lp
                picks["random"].append(E[0])
                picks["min gap"].append(E[g.argmin()])
                picks["min gap, permuted tree"].append(E[gp.argmin()])
                picks["max gap"].append(E[g.argmax()])
            base = np.sqrt((np.array(picks["random"]) ** 2).mean(axis=0))
            for k, v in picks.items():
                rmse = np.sqrt((np.array(v) ** 2).mean(axis=0))
                rows.append({"target": target, "n": n, "selection": k,
                             "ratio": float(np.exp(np.mean(np.log(rmse / base))))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    D = load()
    rng = np.random.default_rng([SEED, 0])
    print("A. lineage signal", flush=True)
    semi, pred = lineage_signal(D, rng)
    semi.to_csv(OUT / "lineage_semivariance.csv", index=False)
    pred.to_csv(OUT / "lineage_prediction.csv", index=False)

    print("B. fixed-inclusion designs: papers", flush=True)
    pp = paper_designs(D)
    pp.to_csv(OUT / "designs_by_paper.csv", index=False)
    ps = paper_summary(pp)
    ps.to_csv(OUT / "designs_paper_summary.csv", index=False)

    print("B. fixed-inclusion designs: full population", flush=True)
    glob, glob_bench = global_designs(D)
    print("B. family bootstrap", flush=True)
    boot = family_bootstrap(D)
    boot.to_csv(OUT / "designs_family_bootstrap.csv", index=False)
    ci = boot.groupby(["target", "n", "design"])["ratio"].quantile([0.025, 0.975]).unstack()
    ci.columns = ["ci_low", "ci_high"]
    share = boot.groupby(["target", "n", "design"])["ratio"].apply(lambda s: float((s < 1).mean())).rename("boot_share_below_one")
    glob = glob.merge(ci.reset_index(), on=["target", "n", "design"]).merge(share.reset_index(), on=["target", "n", "design"])
    glob.to_csv(OUT / "designs_population.csv", index=False)
    glob_bench.to_csv(OUT / "designs_population_by_benchmark.csv", index=False)

    print("C. rejective selection", flush=True)
    rej = rejective(D)
    rej.to_csv(OUT / "rejective_placebo.csv", index=False)

    max_dev = float(max(pp.loc[pp["included"], "inclusion_dev"].max(), glob["inclusion_dev"].max()))
    included = pp.drop_duplicates("paper_id")
    primary = ps[(ps["target"] == "adoption") & (ps["design"] == "lineage")].iloc[0]
    manifest = {"release": common.RELEASE, "seed": SEED, "scored_variants": int(len(D["support"])),
                "families": int(len(np.unique(D["F"].paths[D["support"], 0]))), "benchmarks": D["names"],
                "n_perm": N_PERM, "r_paper": R_PAPER, "r_global": R_GLOBAL, "b_boot": B_BOOT, "r_boot": R_BOOT,
                "papers_included": int(included["included"].sum()), "papers_total": int(len(included)),
                "max_inclusion_deviation_mc": max_dev, "runtime_s": round(time.time() - t0, 1),
                "primary": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                            for k, v in primary.items()}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    sv = semi[(semi["benchmark"] == "mean")][["scores", "relation", "pairs", "semivariance", "null_mean", "null_p975", "p_lower"]]
    pm = pred[pred["benchmark"] == "mean"][["predictor", "loo_r2"]]
    gshow = glob[glob["design"] != "blind"][["target", "n", "design", "ratio", "ci_low", "ci_high", "large_error_rate",
                                             "gap_primary"]]
    report = f"""# RQ4 behavioral validation, release {common.RELEASE}

Scored population: {len(D['support'])} variants with all six Open LLM Leaderboard capability scores, from
{manifest['families']} families. Paper worlds: {manifest['papers_included']} of {manifest['papers_total']} audited
papers have at least twice their panel size in scored variants available by their cutoff.

## Primary endpoint (paper worlds, adoption target, lineage-spread vs lineage-blind)

Median RMSE ratio {primary['median_ratio']:.3f} (95% CI over papers {primary['ci_low']:.3f}-{primary['ci_high']:.3f});
{int(primary['papers_below_one'])} of {int(primary['papers'])} papers below one; Wilcoxon p = {primary['wilcoxon_p']:.2g}.
Inclusion probabilities are equal across designs up to Monte Carlo error (largest deviation of mean multiplicity
from n t(i): {max_dev:.3f}).

## A1. Semivariance by lineage relation (mean over benchmarks; 1 = random pairs)

{markdown_table(sv)}

## A2. Leave-one-out prediction of benchmark scores (mean R2 over benchmarks)

{markdown_table(pm)}

## B. Paper worlds, all targets and designs

{markdown_table(ps)}

## B. Full scored population, family-bootstrap 95% intervals

{markdown_table(gshow)}

## C. Selecting the lowest-gap panel among {K_REJECT} iid draws (RMSE ratio to a random pick)

{markdown_table(rej)}
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
