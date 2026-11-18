"""Assert the standalone math reproduces the source notebook's metric table."""

from __future__ import annotations

import numpy as np

from coverage_geometry import load_foundation


def main() -> None:
    F = load_foundation()
    gt = F.data["gt_paper_metrics"].set_index("paper_id")
    gt_geom = F.data["gt_geometry_scores"]

    checks = ["adoption_gap", "breadth", "structural_linf",
              "coverage_q0", "coverage_q1", "coverage_delta_1_0",
              "posterior_effective_variants"]
    max_abs = {c: 0.0 for c in checks}
    n_scoreable = 0
    for pid, panel in F.panels.items():
        entries = panel["entries"]
        if not entries:
            continue
        n_scoreable += 1
        mb = F.metric_bundle(entries)
        for c in checks:
            if pid in gt.index and c in gt.columns and np.isfinite(gt.loc[pid, c]):
                max_abs[c] = max(max_abs[c], abs(mb[c] - float(gt.loc[pid, c])))

    print(f"scoreable papers checked: {n_scoreable}")
    ok = True
    for c in checks:
        flag = "OK" if max_abs[c] < 1e-9 else "FAIL"
        ok &= max_abs[c] < 1e-9
        print(f"  {c:32s} max|delta|={max_abs[c]:.2e}  {flag}")

    # geometry W1 reproduction across the four geometries
    geom_max = 0.0
    pivot = gt_geom.pivot(index="paper_id", columns="geometry", values="adoption_gap")
    for pid, panel in F.panels.items():
        if not panel["entries"]:
            continue
        sigma = F.sigma(panel["entries"])
        for geom in pivot.columns:
            ref = pivot.loc[pid, geom]
            if np.isfinite(ref):
                got = F.w1(F.PI_A, sigma, geom=("primary" if geom == "structural_variant_count" else geom))
                geom_max = max(geom_max, abs(got - float(ref)))
    print(f"  geometry W1 (4 schemes)           max|delta|={geom_max:.2e}  "
          f"{'OK' if geom_max < 1e-9 else 'FAIL'}")
    ok &= geom_max < 1e-9

    assert ok, "standalone math diverges from source ground truth"
    print("\nALL FOUNDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
