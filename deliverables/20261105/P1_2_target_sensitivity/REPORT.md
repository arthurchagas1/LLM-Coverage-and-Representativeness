# P1.2 Sensitivity of the adoption target, release 20261105

## Summary

Baseline target: downloads with alpha 0.10, effective support 20, median gap 1.820.

Across download-based alpha values from 0 to 0.50, paper rankings keep a Kendall tau of at least
0.884 with the baseline. Across the other targets (log downloads, likes, uniform, and all-time
downloads when available), the lowest tau is 0.275 and the largest single rank shift is
30 positions.

## Metrics and rankings

| target | effective_support | median_adoption_gap | median_reduction_core_pct | papers_improved_core | kendall_tau_vs_baseline | median_rank_shift | max_rank_shift | best_10_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_downloads | 16.207 | 1.821 | 44.651 | 46 | 0.973 | 0 | 5 | 10 |
| alpha_0.01 | 16.535 | 1.821 | 44.356 | 46 | 0.973 | 0 | 5 | 10 |
| alpha_0.05 | 17.951 | 1.822 | 43.174 | 46 | 0.979 | 0 | 4 | 10 |
| alpha_0.10 | 19.991 | 1.82 | 41.634 | 46 | 1 | 0 | 0 | 10 |
| alpha_0.20 | 25.269 | 1.814 | 38.578 | 46 | 0.981 | 0 | 4 | 9 |
| alpha_0.50 | 64.083 | 1.82 | 27.003 | 46 | 0.884 | 1.5 | 10 | 9 |
| log_downloads | 3399.478 | 1.829 | 12.572 | 46 | 0.592 | 5.5 | 24 | 5 |
| likes | 147.145 | 1.61 | 17.273 | 46 | 0.275 | 11.5 | 30 | 4 |
| uniform | 4233.5 | 1.841 | 12.051 | 46 | 0.583 | 6 | 24 | 5 |
| all_time_downloads | 39.192 | 1.775 | 31.654 | 46 | 0.816 | 3 | 19 | 8 |

## Error improvement against each target's own population mean

Mean delta R2 across benchmarks, by separation scheme. `positive` and `ci_above_zero` refer to the leave-size-out scheme,
with family-cluster intervals from 200 replicates.

| target | size | strategy | positive | ci_above_zero |
| --- | --- | --- | --- | --- |
| all_time_downloads | 0.017 | 0.05 | 5 | 0 |
| alpha_0.01 | 0.004 | 0.042 | 4 | 0 |
| alpha_0.05 | 1.9e-04 | 0.005 | 3 | 0 |
| alpha_0.10 | -0.003 | 0.004 | 4 | 0 |
| alpha_0.20 | 0.002 | 0.043 | 4 | 0 |
| alpha_0.50 | 0.013 | 0.061 | 8 | 0 |
| likes | 0.085 | 0.157 | 8 | 6 |
| log_downloads | 9.2e-04 | 0.019 | 6 | 0 |
| raw_downloads | 0.001 | 0.042 | 3 | 1 |
| uniform | 0.011 | 0.005 | 6 | 2 |

## How to read this

- The tree is the same under every target, so the gaps share one distance scale.
- Each target changes the estimand: the population mean being recovered changes with it, so error improvements describe
  how well the gap tracks recovery of that target's own mean, not of one fixed mean.
- A panel's rank can change because the target moved, not because the panel did. Use `rank_correlation_between_targets.csv`
  to see which targets are interchangeable.
- All-time downloads exist only in snapshots collected from 2026-11 on.

## Files

`paper_metrics_by_target.csv`, `stability_by_target.csv`, `rank_correlation_between_targets.csv`, `error_improvement_by_target.csv`,
`error_improvement_summary.csv`, and the figures `metric_stability`, `rank_stability`, and `error_improvement`.
