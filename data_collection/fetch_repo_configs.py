#!/usr/bin/env python3
"""Read context lengths from config.json for the repositories the feasibility analysis uses.

The listing API returns no max_position_embeddings, so context length needs one file per
repository. This fetches only one representative repository per variant, plus every repository named in
a paper panel, and caches results in data/repo_config_cache.csv so reruns only fetch what is missing.
Gated and removed repositories stay unknown: they are recorded, not imputed.

    DMKD_RELEASE=20261105 .venv/bin/python data_collection/fetch_repo_configs.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import release_config  # noqa: E402

for key, value in release_config.release_env(release_config.current_release()).items():
    os.environ.setdefault(key, value)

import pandas as pd  # noqa: E402

CACHE = ROOT / "data" / "repo_config_cache.csv"
CONTEXT_KEYS = ["max_position_embeddings", "n_positions", "n_ctx", "seq_length", "max_sequence_length", "model_max_length"]
HEADERS = {"User-Agent": "research-metadata-check"}
WORKERS = 16


def context_from(config: Any) -> Optional[int]:
    """Largest declared context, looking inside nested text configs too. Sentinel values do not count."""
    if not isinstance(config, dict):
        return None
    found = []
    for key in CONTEXT_KEYS:
        value = config.get(key)
        if isinstance(value, (int, float)) and 0 < value < 10_000_000:
            found.append(int(value))
    for nested in ["text_config", "llm_config", "language_config"]:
        inner = context_from(config.get(nested))
        if inner:
            found.append(inner)
    return max(found) if found else None


def fetch(repo: str) -> Dict[str, Any]:
    url = f"https://huggingface.co/{urllib.parse.quote(repo, safe='/')}/resolve/main/config.json"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as response:
                config = json.loads(response.read().decode("utf-8", "replace"))
            return {"repo_id": repo, "context_length": context_from(config), "status": "ok"}
        except urllib.error.HTTPError as error:
            if error.code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            return {"repo_id": repo, "context_length": None, "status": f"http_{error.code}"}
        except Exception as error:
            if attempt == 4:
                return {"repo_id": repo, "context_length": None, "status": type(error).__name__}
            time.sleep(5)
    return {"repo_id": repo, "context_length": None, "status": "rate_limited"}


def main() -> None:
    columns = ["repo_id", "variant_id", "downloads_30d", "is_likely_official_model", "is_multimodal_leak"]
    lineage = pd.read_parquet(os.environ["COVERAGE_LINEAGE_FILE"], columns=columns)
    lineage = lineage[~lineage["is_multimodal_leak"].fillna(False).astype(bool)].copy()
    lineage["official"] = lineage["is_likely_official_model"].fillna(False).astype(bool)
    lineage["downloads_30d"] = pd.to_numeric(lineage["downloads_30d"], errors="coerce").fillna(0)
    representatives = (lineage.sort_values(["official", "downloads_30d"], ascending=[False, False])
                       .groupby("variant_id", sort=False)["repo_id"].first())
    targets = set(representatives.astype(str))
    known = set(lineage["repo_id"].astype(str))
    papers = json.loads((ROOT / "data/paper_annotations/acl15_new_schema.json").read_text(encoding="utf-8"))
    for paper in papers:
        names = paper.get("coverage_model_list_open_only") or []
        names = names.split(";") if isinstance(names, str) else names
        targets |= {str(name).strip() for name in names if str(name).strip() in known}

    existing = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["repo_id", "context_length", "status"])
    # Rate limits are worth retrying on a later run, unlike a missing file.
    settled = existing[existing["status"].astype(str) != "rate_limited"]
    todo = sorted(targets - set(settled["repo_id"].astype(str)))
    print(f"targets {len(targets)}, cached {len(targets) - len(todo)}, to fetch {len(todo)}", flush=True)

    results = [] if not len(settled) else settled.to_dict("records")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for count, row in enumerate(pool.map(fetch, todo), 1):
            results.append(row)
            if count % 250 == 0 or count == len(todo):
                # Atomic, because the feasibility step may read this file during a long fetch.
                pd.DataFrame(results).to_csv(CACHE.with_suffix(".tmp"), index=False)
                CACHE.with_suffix(".tmp").replace(CACHE)
                print(f"{count}/{len(todo)}", flush=True)
    frame = pd.DataFrame(results)
    frame.to_csv(CACHE, index=False)
    here = frame[frame["repo_id"].isin(targets)]
    print(here["status"].value_counts().to_string())
    print("with context:", int(here["context_length"].notna().sum()), "of", len(here))


if __name__ == "__main__":
    main()
