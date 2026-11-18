#!/usr/bin/env python3
"""Execute a notebook headlessly from the repository root, leaving the committed notebook untouched.

    python scripts/run_notebook.py notebooks/paper_figures.ipynb

The executed copy, with its outputs, goes to outputs/<release>/<notebook name>. The figures themselves are written by
the notebook (figures/<release>/).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    source = ROOT / sys.argv[1]
    release = os.environ.get("DMKD_RELEASE", "20261105")
    target = ROOT / "outputs" / release / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(source, as_version=4)
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(notebook, {"metadata": {"path": str(ROOT)}})
    nbformat.write(notebook, target)
    print(f"executed {source.relative_to(ROOT)} -> {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
