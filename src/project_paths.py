"""Portable project- and data-root discovery.

Every path in this project is derived from the location of this file or from
the environment, so no machine, account, or author layout is baked into the
code. The project root is the parent of ``src/``. The data root -- the checkout
holding the population parquet and the paper annotations -- is looked up in
this order:

  1. the ``COVERAGE_DATA_ROOT`` environment variable;
  2. the project root itself;
  3. the project root's parent;
  4. sibling data checkouts named in ``SIBLING_DATA_ROOTS``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = PROJECT_ROOT

OUTPUTS = Path(os.environ.get("COVERAGE_OUTPUT_DIR") or PROJECT_ROOT / "outputs_corrected")
CACHE = OUTPUTS / "cache"
DIAGNOSTICS = OUTPUTS / "diagnostics"
COUNTERFACTUAL_PANELS = OUTPUTS / "counterfactual_panels"
FOUNDATION_PKL = CACHE / "foundation.pkl"

LINEAGE_RELATIVE = Path("data/oome_lineages_20261105/oome_models_with_lineage.parquet")
ANNOTATIONS_RELATIVE = Path("data/paper_annotations/acl15_new_schema.json")
SIBLING_DATA_ROOTS: list[str] = []  # DMKD27 is self-contained: never fall back to older checkouts.


def _data_root_candidates() -> Iterator[Path]:
    yield PROJECT_ROOT
    yield PROJECT_ROOT.parent
    for name in SIBLING_DATA_ROOTS:
        yield PROJECT_ROOT.parent / name


def data_root() -> Path:
    """Locate the checkout holding the population parquet."""
    override = os.environ.get("COVERAGE_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for candidate in _data_root_candidates():
        if (candidate / LINEAGE_RELATIVE).exists():
            return candidate.resolve()
    tried = ", ".join(str(c) for c in _data_root_candidates())
    raise FileNotFoundError(
        "Could not locate the data root holding "
        f"{LINEAGE_RELATIVE}. Set COVERAGE_DATA_ROOT. Tried: {tried}"
    )


def lineage_file() -> Path:
    override = os.environ.get("COVERAGE_LINEAGE_FILE")
    return Path(override) if override else data_root() / LINEAGE_RELATIVE


def annotation_file() -> Path:
    return data_root() / ANNOTATIONS_RELATIVE


def portable(path: Path, root: Path = PROJECT_ROOT) -> str:
    """Render a path for manifests without leaking the machine layout."""
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(Path(root).resolve())
    except ValueError:
        return resolved.name
    return str(relative) if str(relative) != "." else "."
