#!/usr/bin/env python3
"""Draw the manuscript figures that the validation pipeline does not produce.

Figure 1 of the AAAI PDF had no generating code in the project: it was embedded as
a vector file. This script redraws it from the regenerated validation table, so the
figure now follows the data. Requires outputs_corrected/tables from the validation step.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUTS = Path(os.environ.get("COVERAGE_OUTPUT_DIR") or ROOT / "outputs_corrected")
TABLE = OUTPUTS / "tables/external_grouped_cv_incremental_r2.csv"
OUT = OUTPUTS / "figures"

# Display order and labels used in the paper, top to bottom.
OUTCOME_LABELS = {
    "IFEval": "IFEval",
    "BBH": "BBH",
    "MATH Lvl 5": "MATH Level 5",
    "GPQA": "GPQA",
    "MUSR": "MUSR",
    "MMLU-PRO": "MMLU-PRO",
    "salad_attack_safety": "SALAD attack safety",
    "salad_base_safety": "SALAD base safety",
}
MARK = "#1f77b4"
REFERENCE_LINE = "#4d4d4d"
GRID = "#b0b0b0"


def figure1() -> list[Path]:
    """Adoption-gap incremental out-of-sample fit per outcome, with bootstrap intervals."""
    table = pd.read_csv(TABLE)
    gap = table[table["predictor"] == "adoption_gap"].set_index("outcome")
    missing = sorted(set(OUTCOME_LABELS) - set(gap.index))
    if missing:
        raise ValueError(f"Outcomes absent from {TABLE.name}: {missing}")
    gap = gap.loc[list(OUTCOME_LABELS)]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    positions = range(len(gap))
    ax.errorbar(
        gap["incremental_oos_r2"], positions,
        xerr=[gap["incremental_oos_r2"] - gap["bootstrap_95_low"],
              gap["bootstrap_95_high"] - gap["incremental_oos_r2"]],
        fmt="o", color=MARK, markersize=5, linewidth=1.5, capsize=3, zorder=3,
    )
    ax.axvline(0, color=REFERENCE_LINE, linewidth=0.8, zorder=2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)
    ax.set_yticks(list(positions), [OUTCOME_LABELS[o] for o in gap.index])
    ax.invert_yaxis()
    ax.set_xlabel(r"Incremental out-of-sample fit, $\Delta R^2_{\mathrm{OOS}}$")
    ax.set_title("Adoption gap contribution to population-mean recovery")
    fig.tight_layout()

    paths = [OUT / "figure1_adoption_gap_incremental_r2.pdf", OUT / "figure1_adoption_gap_incremental_r2.png"]
    # Fixed metadata keeps the PDF byte-stable across reruns.
    fig.savefig(paths[0], metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(paths[1], dpi=200)
    plt.close(fig)
    return paths


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in figure1():
        print(path)


if __name__ == "__main__":
    main()
