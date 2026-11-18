#!/usr/bin/env python3
"""P1.5: make the sample of 50 papers auditable.

The artifacts record which papers were annotated and how each panel was reconstructed, but not
how the papers were found. This script therefore keeps three things apart: what the
artifacts document, a retrospective search frame built now with a declared query, and what
the authors still have to confirm. The frame shows how the sample compares with the visible
literature; it does not claim to be the original search.

OpenAlex responses are cached in data/corpus_frame_cache.json and stay within the anonymous
daily budget. Outputs in deliverables/<release>/P1_5_corpus/.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import common
from common import RELEASE, TABLES, deliverable, markdown_table, save_figure

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = common.ROOT
OUT = deliverable("P1_5_corpus")
ANNOTATIONS = ROOT / "data/paper_annotations/acl15_new_schema.json"
CACHE = ROOT / "data/corpus_frame_cache.json"
API = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "research-metadata-check"}

FRAME_QUERY = '("large language model" OR "large language models" OR LLM OR LLMs) AND (benchmark OR benchmarking OR evaluation OR evaluating OR leaderboard)'
FRAME_FROM, FRAME_TO = "2024-01-01", "2026-06-03"
FRAME_FILTER = f"title_and_abstract.search:{FRAME_QUERY},from_publication_date:{FRAME_FROM},to_publication_date:{FRAME_TO}"
SAMPLE_PER_STRATUM = 200

STRATUM_PATTERNS = {
    "ACL Anthology": r"association for computational linguistics|findings of|empirical methods|north american chapter|\bacl\b|\bemnlp\b|\bnaacl\b",
    "AAAI, ICLR, NeurIPS, ICML": r"learning representations|neural information processing|international conference on machine learning|\baaai\b|artificial intelligence.*proceedings|\biclr\b|\bicml\b|neurips",
    "arXiv": r"arxiv",
}
THEMES = [
    ("Safety and trust", r"safety|jailbreak|alignment|harm|bias|fairness|toxic|privacy|trust|attack|red.?team"),
    ("Code", r"\bcode\b|coding|programming|software"),
    ("Mathematics and reasoning", r"math|reason|logic|proof|formal|lean"),
    ("Language and culture", r"multilingual|arabic|chinese|african|afro|translation|cross.?lingual|\bmt\b|cultur"),
    ("Domain-specific", r"medical|medicine|financ|legal|scien|earth|enterprise|journalis|news|anomaly|time.?series|geolog|educat"),
    ("Agents and interaction", r"agent|multi.?turn|interactive|memory|tool|mobile|planning|wiki"),
    ("Long context and summarization", r"long.?context|long.?term|summar"),
]


def venue_group(venue: Any) -> str:
    text = str(venue or "").lower()
    if re.search(r"\b(acl|emnlp|naacl|eacl|findings)\b", text):
        return "ACL Anthology"
    if re.search(r"\b(aaai|iclr|neurips|icml)\b", text):
        return "AAAI, ICLR, NeurIPS, ICML"
    if re.search(r"arxiv|preprint", text):
        return "arXiv or preprint"
    return "Unspecified"


def venue_name(venue: Any) -> str:
    text = str(venue or "").lower()
    for name in ["findings", "emnlp", "naacl", "acl", "aaai", "iclr", "neurips", "icml"]:
        if re.search(rf"\b{name}\b", text):
            return "ACL Findings (all)" if name == "findings" else name.upper()
    return "arXiv or preprint" if re.search(r"arxiv|preprint", text) else "Unspecified"


def theme_of(paper: Dict[str, Any]) -> str:
    text = " ".join(str(paper.get(k, "")) for k in ["benchmark_type", "claim_scope", "title"]).lower()
    for theme, pattern in THEMES:
        if re.search(pattern, text):
            return theme
    return "General capability"


class OpenAlex:
    """Cached, polite client. Stops calling once the anonymous budget or the network refuses."""

    def __init__(self) -> None:
        self.cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
        # Offline (the default of run_all_experiments.py): answer from the cache only, so reruns are deterministic.
        self.available = os.environ.get("DMKD_OFFLINE", "0") != "1"
        self.calls = 0

    def get(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = json.dumps(params, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        if not self.available:
            return None
        query = dict(params)
        if os.environ.get("OPENALEX_API_KEY"):
            query["api_key"] = os.environ["OPENALEX_API_KEY"]
        url = API + "?" + urllib.parse.urlencode(query)
        for attempt in range(8):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.calls += 1
                self.cache[key] = payload
                if self.calls % 10 == 0:
                    self.save()
                time.sleep(0.25)
                return payload
            except urllib.error.HTTPError as error:
                if error.code == 429:
                    # The API names how long to wait while its search cluster is under load.
                    try:
                        wait = float(json.loads(error.read().decode("utf-8")).get("retryAfter", 60))
                    except Exception:
                        wait = 60.0
                    if wait > 600 or attempt == 7:
                        self.available = False
                        return None
                    time.sleep(wait + 2)
                    continue
                if error.code == 503 and attempt == 0:
                    # Search is paused for anonymous users; a whole run of retries would not help.
                    self.available = False
                    return None
                if error.code >= 500 and attempt < 7:
                    time.sleep(5)
                    continue
                return None
            except Exception:
                if attempt == 7:
                    return None
                time.sleep(5)
        return None

    def save(self) -> None:
        CACHE.write_text(json.dumps(self.cache), encoding="utf-8")


def normalize(title: Any) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(title).lower()).strip()


def locate(client: OpenAlex, paper: Dict[str, Any]) -> Dict[str, Any]:
    """Find the paper in OpenAlex by title, and test whether the frame query retrieves it."""
    query = re.sub(r"[^\w\s-]", " ", str(paper.get("title", ""))).strip()
    payload = client.get({"filter": f"title.search:{query}", "per-page": 5,
                          "select": "id,display_name,publication_date,cited_by_count,type,primary_location"})
    row: Dict[str, Any] = {"paper_id": paper["paper_id"], "openalex_found": False}
    if not payload:
        return row
    best, best_score = None, 0.0
    for work in payload.get("results", []):
        score = difflib.SequenceMatcher(None, normalize(paper.get("title")), normalize(work.get("display_name"))).ratio()
        if score > best_score:
            best, best_score = work, score
    if best is None or best_score < 0.9:
        return row
    source = ((best.get("primary_location") or {}).get("source") or {}).get("display_name")
    row.update(openalex_found=True, openalex_id=best["id"].rsplit("/", 1)[-1], openalex_date=best.get("publication_date"),
               cited_by=best.get("cited_by_count"), openalex_source=source)
    check = client.get({"filter": f"ids.openalex:{row['openalex_id']},{FRAME_FILTER}", "per-page": 1, "select": "id"})
    if check is not None:
        row["in_retrospective_frame"] = bool(check.get("meta", {}).get("count", 0))
    return row


def build_frame(client: OpenAlex) -> Dict[str, Any]:
    """Frame size overall and by stratum, by year, and citation samples per stratum."""
    out: Dict[str, Any] = {"available": False}
    total = client.get({"filter": FRAME_FILTER, "per-page": 1, "select": "id"})
    if not total:
        return out
    out["available"] = True
    out["total"] = total["meta"]["count"]
    by_year = client.get({"filter": FRAME_FILTER, "group_by": "publication_year"})
    out["by_year"] = {g["key"]: g["count"] for g in (by_year or {}).get("group_by", [])}
    groups = client.get({"filter": FRAME_FILTER, "group_by": "primary_location.source.id", "per-page": 200})
    strata: Dict[str, Dict[str, Any]] = {name: {"count": 0, "sources": []} for name in STRATUM_PATTERNS}
    for group in (groups or {}).get("group_by", []):
        name = str(group.get("key_display_name") or "")
        for stratum, pattern in STRATUM_PATTERNS.items():
            if re.search(pattern, name, re.I):
                strata[stratum]["count"] += group["count"]
                strata[stratum]["sources"].append(str(group["key"]).rsplit("/", 1)[-1])
                break
    out["strata"] = strata
    samples: Dict[str, List[int]] = {}
    for stratum, data in strata.items():
        if not data["sources"]:
            continue
        ids = "|".join(data["sources"][:40])
        payload = client.get({"filter": f"{FRAME_FILTER},primary_location.source.id:{ids}", "sample": SAMPLE_PER_STRATUM, "seed": 20261105,
                              "per-page": SAMPLE_PER_STRATUM, "select": "cited_by_count,publication_year"})
        if payload:
            samples[stratum] = [w.get("cited_by_count", 0) for w in payload.get("results", [])]
    out["citation_samples"] = samples
    return out


def write_flowchart(flow: List[tuple]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 1.15 * len(flow) + 0.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(flow))
    ax.axis("off")
    for i, (title, detail, known) in enumerate(flow):
        y = len(flow) - i - 1
        box = FancyBboxPatch((0.6, y + 0.14), 8.8, 0.72, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.0,
                             edgecolor="#1f77b4" if known else "#d95f02", facecolor="#eef4fa" if known else "#fdf0e6",
                             linestyle="-" if known else "--")
        ax.add_patch(box)
        ax.text(5, y + 0.6, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#222222")
        ax.text(5, y + 0.34, detail, ha="center", va="center", fontsize=8, color="#444444")
        if i < len(flow) - 1:
            ax.annotate("", xy=(5, y + 0.12), xytext=(5, y - 0.12), arrowprops=dict(arrowstyle="<-", color="#666666", lw=1.0))
    ax.set_title("Sample flow. Dashed boxes are not documented or are reconstructed after the fact", fontsize=10)
    save_figure(fig, OUT, "sample_flowchart")
    plt.close(fig)


def main() -> None:
    papers = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    frame = pd.DataFrame([{
        "paper_id": p["paper_id"], "title": p.get("title"), "year": p.get("year"), "venue": p.get("venue"),
        "stratum": venue_group(p.get("venue")), "venue_short": venue_name(p.get("venue")), "theme": theme_of(p),
        "source_url": p.get("source_url") or "", "source": p.get("source") or "", "source_type": p.get("source_type") or "",
        "include_final_sample": p.get("include_final_sample"), "needs_manual_review": p.get("needs_manual_review"),
        "normalization_confidence": p.get("normalization_confidence"), "has_selection_notes": bool(p.get("source_selection_notes")),
        "n_main_models": pd.to_numeric(p.get("main_n_models_total"), errors="coerce"),
        "n_open_models": pd.to_numeric(p.get("main_n_open_weight_models"), errors="coerce"),
        "n_closed_models": pd.to_numeric(p.get("main_n_closed_models"), errors="coerce"),
        "has_open_coverage_list": bool(p.get("coverage_model_list_open_only")),
        "benchmark_type": p.get("benchmark_type"), "claim_scope": p.get("claim_scope"),
    } for p in papers])

    client = OpenAlex()
    located = pd.DataFrame([locate(client, p) for p in papers])
    frame_info = build_frame(client)
    client.save()
    frame = frame.merge(located, on="paper_id", how="left")
    dates_path = ROOT / "data/paper_dates.csv"
    if dates_path.exists():
        dates = pd.read_csv(dates_path)[["paper_id", "arxiv_id", "first_public_date", "arxiv_match"]]
        frame = frame.merge(dates, on="paper_id", how="left")
    frame.to_csv(OUT / "corpus_table.csv", index=False)

    composition = frame.groupby(["stratum", "venue_short", "year"], dropna=False).size().rename("papers").reset_index()
    composition.to_csv(OUT / "composition_by_venue_year.csv", index=False)
    theme_table = frame.pivot_table(index="theme", columns="stratum", values="paper_id", aggfunc="count", fill_value=0)
    theme_table["Total"] = theme_table.sum(axis=1)
    theme_table = theme_table.sort_values("Total", ascending=False).reset_index()
    theme_table.to_csv(OUT / "composition_by_theme.csv", index=False)
    stratum_table = frame.groupby("stratum").agg(
        papers=("paper_id", "count"), median_models=("n_main_models", "median"), median_open=("n_open_models", "median"),
        median_closed=("n_closed_models", "median"), no_open_panel=("has_open_coverage_list", lambda s: int((~s).sum())),
        needs_manual_review=("needs_manual_review", lambda s: int(s.astype(str).eq("True").sum()))).reset_index()
    stratum_table.to_csv(OUT / "composition_by_stratum.csv", index=False)

    comparison_rows = []
    if frame_info.get("available"):
        for stratum, data in frame_info["strata"].items():
            sample_group = "arXiv or preprint" if stratum == "arXiv" else stratum
            ours = frame[frame["stratum"] == sample_group]
            cited = pd.to_numeric(ours.get("cited_by"), errors="coerce").dropna()
            others = frame_info.get("citation_samples", {}).get(stratum, [])
            p_value = float(mannwhitneyu(cited, others, alternative="two-sided").pvalue) if len(cited) >= 3 and len(others) >= 3 else np.nan
            comparison_rows.append({
                "stratum": stratum, "frame_papers": data["count"], "sample_papers": len(ours),
                "sampling_fraction_pct": 100.0 * len(ours) / data["count"] if data["count"] else np.nan,
                "sample_found_in_openalex": int(ours["openalex_found"].fillna(False).astype(bool).sum()),
                "sample_retrieved_by_query": int(ours.get("in_retrospective_frame", pd.Series(dtype=float)).fillna(False).astype(bool).sum()),
                "median_citations_sample": float(cited.median()) if len(cited) else np.nan,
                "median_citations_frame_sample": float(np.median(others)) if len(others) else np.nan,
                "frame_citation_sample_size": len(others), "citations_mannwhitney_p": p_value,
            })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT / "sample_vs_frame.csv", index=False)

    n_total = len(frame)
    no_panel = frame[~frame["has_open_coverage_list"]]
    mentions, resolved = None, None
    audit_path = TABLES / "corrected_resolution_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        mentions = len(audit)
        resolved = int((audit["status"].astype(str) != "unresolved").sum())
    counts = frame["stratum"].value_counts().to_dict()
    flow = []
    if frame_info.get("available"):
        flow.append((f"Retrospective search, not the original one: {frame_info['total']:,} OpenAlex records",
                     f"{FRAME_FROM} to {FRAME_TO}, query in SELECTION_PROTOCOL.md", False))
    flow += [("Original search and screening: not recorded", "Databases, queries, dates, and screened counts are unknown", False),
             (f"{n_total} papers annotated", "; ".join(f"{k}: {v}" for k, v in counts.items()), True),
             (f"{n_total - len(no_panel)} with an open-weight panel to score", f"{len(no_panel)} removed: no open-weight coverage list", True)]
    if mentions:
        flow.append((f"{mentions} model mentions, {resolved} resolved", f"{mentions - resolved} unresolved, kept in the audit", True))
    write_flowchart(flow)

    write_protocol(frame, frame_info, comparison, theme_table, stratum_table, no_panel)
    print((OUT / "SELECTION_PROTOCOL.md").read_text(encoding="utf-8")[:3500])
    print("OpenAlex calls this run:", client.calls)


def write_protocol(frame: pd.DataFrame, info: Dict[str, Any], comparison: pd.DataFrame, themes: pd.DataFrame,
                   strata: pd.DataFrame, no_panel: pd.DataFrame) -> None:
    source_types = Counter(frame["source_type"].replace("", "not recorded"))
    venues = frame["venue_short"].value_counts().to_dict()
    aaai = int((frame["venue_short"] == "AAAI").sum())
    review = int(frame["needs_manual_review"].astype(str).eq("True").sum())
    first_person = int(frame["has_selection_notes"].sum())
    urls = int((frame["source_url"] != "").sum())
    found = int(frame["openalex_found"].fillna(False).astype(bool).sum())
    retrieved = int(frame.get("in_retrospective_frame", pd.Series(dtype=float)).fillna(False).astype(bool).sum())
    frame_block = "OpenAlex was unavailable in this run, so the frame was not built. Rerun this step to build it."
    draft_frame = "[Retrospective frame check to be added once it is built.]"
    if info.get("available"):
        draft_frame = (f"As a retrospective check, we built a search frame in OpenAlex with a declared query, which retrieves "
                       f"{retrieved} of the {found} sample papers indexed there; Table X compares the two.")
        frame_block = f"""The frame holds {info['total']:,} records, and the query retrieves {retrieved} of the {found} sample papers found in OpenAlex.

{markdown_table(comparison)}

Records per year: {info.get('by_year')}.

OpenAlex indexes conference proceedings unevenly, ICLR especially, so stratum sizes are approximate and likely undercount
that stratum. The citation comparison uses a random sample of each stratum. Citations are also recency-confounded."""
    text = f"""# Selection protocol for the 50 audited papers, release {RELEASE}

Each statement below is marked **documented** (recorded in the project), **reconstructed** (built now, after the sample
existed), or **to confirm** (only the authors can say).

## Search strategy

- **To confirm:** the databases, queries, and dates of the original search. No artifact records them, nor how many papers were screened.
- **Documented:** annotation files date from June 2026 at the latest (`data/paper_annotations/`), and the Hub snapshot is from 2026-06-03.
- **Reconstructed:** a retrospective frame, using OpenAlex, with this query and range:

```
{FRAME_QUERY}
publication date {FRAME_FROM} to {FRAME_TO}
```

The frame tests how the sample sits within the visible literature. It should be described as a retrospective check, never as the search that produced the sample.

## Inclusion and exclusion criteria

**Documented** in Technical Supplement C.3 and the annotation schema:

- Included: the study's central contribution includes an empirical evaluation of identifiable language-model systems, with enough information to reconstruct its main panel (`is_main_text_llm_eval` is true for all {len(frame)}).
- Duplicates across publication sources removed.
- arXiv stratum restricted to recent preprints available at corpus construction.
- `include_final_sample` values: {frame['include_final_sample'].value_counts().to_dict()}.

**To confirm:** whether these criteria were set before selecting papers or described afterward, and why a paper marked `no` is still in the annotation file.

## Dates

- Publication years: {frame['year'].value_counts().sort_index().to_dict()}.
- **To confirm:** the corpus cutoff date.

## Composition

Strata as the venue fields show them:

{markdown_table(strata)}

Venues: {venues}.

Themes, classified with keyword rules over benchmark type, claim scope, and title (reconstructed, with no human check):

{markdown_table(themes)}

Per-paper detail is in `corpus_table.csv`, with counts by venue and year in `composition_by_venue_year.csv`.

## Flow

See `sample_flowchart.png`. Solid boxes are documented, and dashed boxes are unrecorded or reconstructed.

Papers without an open-weight panel to score ({len(no_panel)}): {', '.join(no_panel['paper_id'])}.

## Sample against the retrospective frame

{frame_block}

## Possible biases

- **Not a probability sample.** Fixed quotas by source mean the audit describes these papers, and generalizing to "the literature" needs care. Percentiles and medians across papers are descriptive.
- **Venue mix.** The second stratum is described as AAAI, ICLR, NeurIPS, and ICML, but holds {aaai} AAAI papers. ICLR supplies {venues.get('ICLR', 0)} of its papers. The Technical Supplement's description should change, or the stratum be rebalanced.
- **Text availability.** {source_types.most_common(1)[0][1]} records have source type "{source_types.most_common(1)[0][0]}". Other values: {dict(source_types)}. Selection may favor papers whose text was at hand.
- **Open-weight requirement.** Scoring needs named open-weight models, so closed-only evaluations are out of scope by design, which can make open-weight panels look typical.
- **Recent-work tilt.** Most papers are from 2024 onward, with models evolving quickly. See P1.3 for how the cutoffs affect this.
- **Citation level.** See the frame comparison above; a selection effect would show as higher citations than the frame, within strata.

## Annotation process

- **Documented:** {review} of {len(frame)} records have `needs_manual_review = True`, and normalization confidence is {frame['normalization_confidence'].value_counts().to_dict()}. Nothing records whether those reviews took place.
- **To confirm:** who or what produced the annotations. {first_person} records hold selection notes written in the first person ("I used Table 3 as the authoritative main model list"), and source types such as `user_pasted_paper_text` suggest text pasted into an assistant. If an LLM produced or assisted them, the paper should say so and report what verification followed.
- No inter-annotator check of panel extraction exists, for the mentions or the lists.
- {urls} of {len(frame)} records have a source URL; {found} were located in OpenAlex by title. `corpus_table.csv` carries these links, so the missing URLs can be filled in.

## What the authors need to supply

1. Original search source, queries, and dates, or a statement that the sample was purposive and hand-assembled.
2. Whether the criteria came before or after selection, and the corpus cutoff date.
3. Who annotated, with which tool, and which of the {review} flagged records were reviewed.
4. How to describe the second stratum, given no AAAI papers appear in it.

## Draft text for the paper, once the items above are confirmed

> We audit a purposive, source-stratified sample of {len(frame)} LLM evaluation papers published between {frame['year'].dropna().astype(str).min()} and {frame['year'].dropna().astype(str).max()}: [strata as confirmed]. Papers qualified when their central contribution included an empirical evaluation of identifiable language-model systems with enough detail to reconstruct the evaluated panel. Because selection was not probabilistic, cross-paper statistics describe this sample and do not estimate prevalence across the literature. {draft_frame}
"""
    (OUT / "SELECTION_PROTOCOL.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
