#!/usr/bin/env python3
"""Find each audited paper on arXiv: first public date, plus full text for the date search.

Writes data/paper_dates.csv and one text file per paper in data/paper_fulltext/. Results are
cached, so reruns only fetch what is missing. arXiv asks for one request every three seconds.
"""

from __future__ import annotations

import difflib
import json
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "paper_dates.csv"
TEXT_DIR = ROOT / "data" / "paper_fulltext"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
HEADERS = {"User-Agent": "research-metadata-check"}
MINIMUM_TITLE_SIMILARITY = 0.90


def normalize(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(title).lower()).strip()


def get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def search_arxiv(title: str):
    """Best title match, searching the exact phrase first and then the title words."""
    words = [w for w in normalize(title).split() if len(w) > 2][:12]
    queries = [f'ti:"{title}"', " AND ".join(f"ti:{w}" for w in words)]
    best = None
    for query in queries:
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": query, "start": 0, "max_results": 8})
        root = ET.fromstring(get(url))
        time.sleep(3.2)
        for entry in root.findall("a:entry", ATOM):
            candidate = entry.findtext("a:title", "", ATOM).replace("\n", " ")
            score = difflib.SequenceMatcher(None, normalize(title), normalize(candidate)).ratio()
            if best is None or score > best["title_similarity"]:
                best = {"arxiv_id": entry.findtext("a:id", "", ATOM).rsplit("/", 1)[-1],
                        "arxiv_title": " ".join(candidate.split()), "title_similarity": round(score, 3),
                        "first_public_date": entry.findtext("a:published", "", ATOM)[:10],
                        "last_updated_date": entry.findtext("a:updated", "", ATOM)[:10]}
        if best and best["title_similarity"] >= MINIMUM_TITLE_SIMILARITY:
            break
    return best


def fetch_text(arxiv_id: str, paper_id: str) -> bool:
    target = TEXT_DIR / f"{paper_id}.txt"
    if target.exists():
        return True
    with tempfile.TemporaryDirectory(dir=TEXT_DIR) as folder:
        pdf = Path(folder) / "paper.pdf"
        pdf.write_bytes(get(f"https://arxiv.org/pdf/{arxiv_id}", timeout=120))
        time.sleep(3.2)
        subprocess.run(["pdftotext", "-layout", str(pdf), str(target)], check=True, timeout=120)
    return target.exists()


def main() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    papers = json.loads((ROOT / "data/paper_annotations/acl15_new_schema.json").read_text(encoding="utf-8"))
    existing = pd.read_csv(OUT_CSV).set_index("paper_id") if OUT_CSV.exists() else pd.DataFrame()
    rows = []
    for index, paper in enumerate(papers):
        pid = paper["paper_id"]
        row = {"paper_id": pid, "title": paper.get("title"), "year": paper.get("year"), "venue": paper.get("venue"),
               "source_url": paper.get("source_url")}
        if pid in existing.index and pd.notna(existing.loc[pid].get("arxiv_id")):
            row.update(existing.loc[pid].to_dict())
        else:
            try:
                row.update(search_arxiv(str(paper.get("title"))) or {})
            except Exception as error:
                row["lookup_error"] = f"{type(error).__name__}: {error}"
        similarity = row.get("title_similarity")
        matched = pd.notna(similarity) and float(similarity) >= MINIMUM_TITLE_SIMILARITY
        row["arxiv_match"] = bool(matched)
        if matched:
            try:
                row["fulltext_available"] = fetch_text(str(row["arxiv_id"]), pid)
            except Exception as error:
                row["fulltext_available"] = False
                row["lookup_error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"{index + 1:2d}/{len(papers)} {pid:45s} sim={similarity} date={row.get('first_public_date')}", flush=True)
    frame = pd.DataFrame(rows)
    print("matched:", int(frame["arxiv_match"].sum()), "of", len(frame))


if __name__ == "__main__":
    main()
