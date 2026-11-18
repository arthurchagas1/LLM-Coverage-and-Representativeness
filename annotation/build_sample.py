#!/usr/bin/env python3
"""Draw the stratified, frozen sample of repositories for the human validation of the lineage taxonomy.

Strata (release 20261105, textual population of 414,075 repositories):
  large      families with >= 1,000 repositories   (23 families, ~96% of repositories)
  medium     families with 50-999 repositories
  small      families with 2-49 repositories
  singleton  families with a single repository
  panel      the most-downloaded repository of each variant that the 46 audited panels resolve to
             (the models whose taxonomy errors move the paper's coverage estimates most directly)

Within the four family-size strata repositories are drawn uniformly without replacement, so each item carries the
design weight N_stratum / n_stratum for population-level estimates. Panel items are excluded from the other strata
and estimated separately.

Outputs (in annotation/data/, refuse to overwrite once any annotation exists):
  items.json          what the annotation site shows: repository id, link, and raw Hub metadata only. No system label.
  system_labels.csv   the taxonomy's labels for each item, hidden from annotators, used only by agreement.py.
  vocabulary.json     known family and series slugs and the allowed size buckets and roles (autocomplete only).
  sample_manifest.json  seed, strata sizes, population sizes, weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "analysis")]
import release_config  # noqa: E402

OUT = ROOT / "annotation" / "data"
SEED = 20261117
DEFAULT_N = {"large": 60, "medium": 50, "small": 50, "singleton": 40, "panel": 40}
BATCHES = [60, 60, 120]  # nested checkpoints: agreement can be computed at 60, 120, and 240 labels
TIERS = [("singleton", 1, 1), ("small", 2, 49), ("medium", 50, 999), ("large", 1000, 10 ** 12)]
SIZE_BUCKETS = ["<1B", "1B-3B", "3B-7B", "7B-13B", "13B-34B", "34B-70B", "70B-180B", "180B+", "unknown"]
ROLES = ["base", "instruct", "chat", "reasoning", "code", "math", "domain", "seq2seq", "unknown-role"]
SHOW = ["pipeline_tag", "library_name", "architecture", "model_type", "base_models", "base_model_relation",
        "parameter_count", "context_length", "license", "languages", "tags", "created_at", "downloads_30d", "gated"]
SYSTEM = ["family_id", "series_id", "base_model_id", "variant_id", "size_label", "size_bucket", "model_role",
          "training_stage", "is_likely_official_model", "is_derivative_model", "lineage_family_evidence",
          "lineage_size_evidence", "lineage_role_evidence"]


def text(value, limit: int = 400) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        value = ", ".join(str(v) for v in value)
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return ""
    s = str(value)
    return s if len(s) <= limit else s[:limit] + " ..."


def panel_repos(lineage: pd.DataFrame) -> pd.DataFrame:
    """Most-downloaded repository of each variant that the audited panels resolve to."""
    import common  # noqa: E402

    F = common.foundation()
    variants = set()
    for panel in F.panels.values():
        for e in panel["entries"]:
            for i, p in zip(e["candidate_idx"], e["posterior"]):
                if p > 0:
                    variants.add(str(F.variant.iloc[int(i)]["variant_id"]))
    cand = lineage[lineage["variant_id"].isin(variants)].sort_values("downloads_30d", ascending=False)
    return cand.drop_duplicates("variant_id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite even if annotations exist")
    for k, v in DEFAULT_N.items():
        ap.add_argument(f"--n-{k}", type=int, default=v)
    args = ap.parse_args()
    n_by = {k: getattr(args, f"n_{k}") for k in DEFAULT_N}

    ann_dir = ROOT / "annotation" / "annotations"
    if (OUT / "items.json").exists() and any(ann_dir.glob("*.jsonl")) and not args.force:
        sys.exit("Annotations already exist; the sample is frozen. Use --force only if you know what you are doing.")
    OUT.mkdir(parents=True, exist_ok=True)

    release = release_config.current_release()
    path = release_config.ROOT / release_config.RELEASES[release]["lineage"] / "oome_models_with_lineage.parquet"
    lineage = pd.read_parquet(path, columns=["repo_id", "is_multimodal_leak", *SHOW, *SYSTEM])
    lineage = lineage[~lineage["is_multimodal_leak"].fillna(False).astype(bool)].reset_index(drop=True)
    lineage["downloads_30d"] = pd.to_numeric(lineage["downloads_30d"], errors="coerce").fillna(0)
    fam_size = lineage["family_id"].map(lineage["family_id"].value_counts())
    lineage["family_repos"] = fam_size.astype(int)

    rng = np.random.default_rng(SEED)
    panel = panel_repos(lineage)
    take_panel = panel.sample(n=min(n_by["panel"], len(panel)), random_state=int(rng.integers(2 ** 31)))
    chosen = [take_panel.assign(stratum="panel", population=len(panel))]
    rest = lineage[~lineage["repo_id"].isin(take_panel["repo_id"])]
    for name, lo, hi in TIERS:
        pool = rest[(rest["family_repos"] >= lo) & (rest["family_repos"] <= hi)]
        k = min(n_by[name], len(pool))
        chosen.append(pool.sample(n=k, random_state=int(rng.integers(2 ** 31))).assign(stratum=name, population=len(pool)))
    sample = pd.concat(chosen, ignore_index=True)
    sample["n_stratum"] = sample.groupby("stratum")["repo_id"].transform("size")
    sample["weight"] = sample["population"] / sample["n_stratum"]
    sample = sample.sample(frac=1.0, random_state=int(rng.integers(2 ** 31))).reset_index(drop=True)
    sample["item_id"] = [f"T{k + 1:03d}" for k in range(len(sample))]

    # Nested, stratified batches. A separate generator keeps the sample itself unchanged. Within each stratum the
    # items get evenly spaced positions in random order; sorting all items by position and cutting at the batch
    # sizes gives every batch (and every prefix of batches) nearly the same stratum mix as the full sample.
    brng = np.random.default_rng(SEED + 1)
    pos = np.empty(len(sample))
    for _, idx in sample.groupby("stratum").indices.items():
        ranks = brng.permutation(len(idx))
        pos[idx] = (ranks + brng.random(len(idx))) / len(idx)
    order = np.argsort(pos, kind="stable")
    bounds = np.cumsum(BATCHES)
    if bounds[-1] != len(sample):
        bounds[-1] = len(sample)
    batch = np.empty(len(sample), int)
    batch[order] = np.searchsorted(bounds, np.arange(len(sample)), side="right") + 1
    sample["batch"] = batch

    items = []
    for row in sample.itertuples(index=False):
        r = row._asdict()
        items.append({"item_id": r["item_id"], "batch": int(r["batch"]), "repo_id": r["repo_id"],
                      "url": f"https://huggingface.co/{r['repo_id']}",
                      "metadata": {k: text(r[k]) for k in SHOW}})
    (OUT / "items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    sample[["item_id", "batch", "repo_id", "stratum", "population", "n_stratum", "weight", "family_repos", *SYSTEM]].to_csv(
        OUT / "system_labels.csv", index=False)

    series = (lineage.groupby("family_id")["series_id"].unique().apply(lambda s: sorted(map(str, s))).to_dict())
    vocab = {"families": sorted(map(str, lineage["family_id"].unique())), "series": series,
             "size_buckets": SIZE_BUCKETS, "roles": ROLES}
    (OUT / "vocabulary.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    manifest = {"release": release, "seed": SEED, "population_repositories": int(len(lineage)),
                "strata": {s: {"sampled": int((sample["stratum"] == s).sum()),
                               "population": int(sample.loc[sample["stratum"] == s, "population"].iloc[0])}
                           for s in sample["stratum"].unique()},
                "tiers": {name: [lo, hi] for name, lo, hi in TIERS},
                "batches": {int(b): {s: int(((sample["batch"] == b) & (sample["stratum"] == s)).sum())
                                     for s in sample["stratum"].unique()} for b in sorted(sample["batch"].unique())}}
    (OUT / "sample_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
