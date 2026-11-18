"""Where each data release lives. A release is one Hub snapshot and everything built from it.

Every pipeline step reads its folders from the environment variables set by `release_env`, so the
same code runs on any release without path edits.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent

RELEASES: Dict[str, Dict[str, str]] = {
    # Hub snapshot collected on 2026-11-05 UTC. This is the release reported in the paper. Its taxonomy was built
    # before the `custom_code` role fix (see data_collection/p4_families.py), so the fix is off here.
    "20261105": {
        "snapshot": "data/hf_snapshots_20261105",
        "clean": "data/oome_clean_20261105",
        "lineage": "data/oome_lineages_20261105",
        "ignore_custom_code_tag": "0",
    },
    # Same snapshot and cleaned population, with the role fix found by the human validation of the taxonomy.
    "20261105_rolefix": {
        "snapshot": "data/hf_snapshots_20261105",
        "clean": "data/oome_clean_20261105",
        "lineage": "data/oome_lineages_20261105_rolefix",
        "ignore_custom_code_tag": "1",
    },
}
DEFAULT_RELEASE = "20261105"


def current_release() -> str:
    return os.environ.get("DMKD_RELEASE", DEFAULT_RELEASE)


def release_env(release: str) -> Dict[str, str]:
    """Environment variables that point every step at this release."""
    folders = RELEASES[release]
    return {
        "DMKD_RELEASE": release,
        "HF_SNAPSHOT_DIR": str(ROOT / folders["snapshot"]),
        "OOME_CLEAN_DIR": str(ROOT / folders["clean"]),
        "OOME_LINEAGE_DIR": str(ROOT / folders["lineage"]),
        "OOME_ROLE_IGNORE_CUSTOM_CODE": folders["ignore_custom_code_tag"],
        "COVERAGE_DATA_ROOT": str(ROOT),
        "COVERAGE_PROJECT_ROOT": str(ROOT),
        "COVERAGE_LINEAGE_FILE": str(ROOT / folders["lineage"] / "oome_models_with_lineage.parquet"),
        "COVERAGE_OUTPUT_DIR": str(ROOT / "outputs" / release),
        "DMKD_DELIVERABLES": str(ROOT / "deliverables" / release),
    }


def output_dir(release: str | None = None) -> Path:
    return ROOT / "outputs" / (release or current_release())


def deliverables_dir(task: str, release: str | None = None) -> Path:
    path = ROOT / "deliverables" / (release or current_release()) / task
    path.mkdir(parents=True, exist_ok=True)
    return path
