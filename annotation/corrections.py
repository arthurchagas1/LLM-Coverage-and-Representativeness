#!/usr/bin/env python3
"""Correction layer for clear annotation errors. The raw files in annotations/ are never edited.

Two kinds of corrections, applied the same way to every annotator and logged one row per changed label in
results/corrections_log.csv:

1. Rules that only use what the annotation site showed (repository name and Hub metadata) and a guideline that
   leaves no room for judgment:
   R-size   The Hub reports `parameter_count` and the annotator's bucket differs from the bucket of that count
            (including `unknown`). Guideline 5: the bucket follows the actual total parameters, not the name, and
            training tokens are not parameters.
   R-role   The repository name carries an explicit tuning suffix and the annotator chose a role that contradicts it:
            `instruct` / `it` -> instruct when the label was base, unknown-role, chat, or code without a code token
            in the name; `chat` -> chat when the label was base or unknown-role; `base` / `pretrain` /
            `pretrained` -> base when the label was unknown-role, the name has no tuning token, and no base model
            is declared. Guideline 6.
   R-scope  The Hub tags the repository `image-text-to-text` or `image-to-text`: a vision-language model, outside
            the text-only population of the study (guideline 2), so the label becomes out_of_scope.
2. Item-level corrections in corrections_manual.csv (item_id, annotator, field, new_value, reason). Each one cites
   the metadata or guideline that makes the original label a clear error; ambiguous cases stay for adjudication.
   `annotator` may be `*` to apply to every annotator whose label differs.

Agreement is reported both on the raw labels (the reliability estimate) and on the corrected labels; reference
labels and taxonomy accuracy use the corrected labels.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TUNING = {"sft", "dpo", "grpo", "ppo", "rlhf", "rl", "orpo", "kto", "ft", "finetune", "finetuned", "lora", "instruct",
          "it", "chat"}
BASE_TOKENS = {"base", "pretrain", "pretrained"}
MULTIMODAL_TAGS = {"image-text-to-text", "image-to-text"}
CODE_TOKENS = {"code", "coder", "coding", "starcoder", "codellama", "codegen", "codeqwen"}


def bucket_of_count(n: float) -> str:
    b = n / 1e9
    for cut, name in [(1, "<1B"), (3, "1B-3B"), (7, "3B-7B"), (13, "7B-13B"), (34, "13B-34B"), (70, "34B-70B"),
                      (180, "70B-180B")]:
        if b < cut:
            return name
    return "180B+"


def name_tokens(repo_id: str) -> list[str]:
    name = repo_id.split("/", 1)[-1].lower()
    return [t for t in re.split(r"[^a-z0-9]+", name) if t]


def rule_corrections(ann: pd.DataFrame, items: dict) -> list[dict]:
    log = []
    for idx, r in ann.iterrows():
        meta = items[r["item_id"]]["metadata"]
        repo = items[r["item_id"]]["repo_id"]
        tags = {t.strip().lower() for t in str(meta.get("tags", "")).split("|")} | {str(meta.get("pipeline_tag", "")).lower()}
        if r.get("decision") and r.get("decision") != "out_of_scope" and tags & MULTIMODAL_TAGS:
            log.append({"row": idx, "field": "decision", "new": "out_of_scope", "rule": "R-scope",
                        "reason": "tag " + ",".join(sorted(tags & MULTIMODAL_TAGS))})
            continue
        if r.get("decision") != "in_scope":
            continue
        try:
            count = float(meta.get("parameter_count") or "nan")
        except ValueError:
            count = float("nan")
        if np.isfinite(count) and count > 0:
            want = bucket_of_count(count)
            if r["size_bucket"] and r["size_bucket"] != want:
                log.append({"row": idx, "field": "size_bucket", "new": want, "rule": "R-size",
                            "reason": f"parameter_count={int(count)}"})
        toks = name_tokens(repo)
        role = r.get("role", "")
        if role:
            if ("instruct" in toks or "it" in toks) and "chat" not in toks:
                if role in {"base", "unknown-role", "chat"} or (role == "code" and not CODE_TOKENS & set(toks)):
                    log.append({"row": idx, "field": "role", "new": "instruct", "rule": "R-role",
                                "reason": "name has instruct/it suffix"})
            elif "chat" in toks and role in {"base", "unknown-role"}:
                log.append({"row": idx, "field": "role", "new": "chat", "rule": "R-role", "reason": "name has chat"})
            elif (role == "unknown-role" and BASE_TOKENS & set(toks) and not TUNING & set(toks)
                  and not str(meta.get("base_models", "")).strip()):
                log.append({"row": idx, "field": "role", "new": "base", "rule": "R-role",
                            "reason": "name has base/pretrain, no declared base model"})
    return log


def apply_corrections(ann: pd.DataFrame, data_dir: Path, manual_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (corrected annotations, log). `ann` must already have one row per (annotator, item)."""
    items = {i["item_id"]: i for i in json.loads((data_dir / "items.json").read_text(encoding="utf-8"))}
    out = ann.copy()
    log = rule_corrections(out, items)
    if manual_path.exists():
        manual = pd.read_csv(manual_path, dtype=str, comment="#").fillna("")
        for m in manual.itertuples(index=False):
            sel = out["item_id"] == m.item_id
            if m.annotator != "*":
                sel &= out["annotator"] == m.annotator
            for idx in out.index[sel]:
                log.append({"row": idx, "field": m.field, "new": m.new_value, "rule": "manual", "reason": m.reason})
    rows = []
    for e in log:
        old = out.at[e["row"], e["field"]]
        if str(old) == e["new"]:
            continue
        rows.append({"item_id": out.at[e["row"], "item_id"], "annotator": out.at[e["row"], "annotator"],
                     "field": e["field"], "old": old, "new": e["new"], "rule": e["rule"], "reason": e["reason"]})
        out.at[e["row"], e["field"]] = e["new"]
        if e["field"] == "decision" and e["new"] != "in_scope":
            for f in ("family", "series", "size_bucket", "role", "artifact"):
                out.at[e["row"], f] = ""
    return out, pd.DataFrame(rows, columns=["item_id", "annotator", "field", "old", "new", "rule", "reason"])
