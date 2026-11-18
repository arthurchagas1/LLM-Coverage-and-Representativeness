# P1.4 Value of the Wasserstein guarantee, release 20261105

## Summary

- **Validity is not informative here.** With ell computed on the same population as the panels, the ratio cannot exceed one,
  so `max_ratio` below one verifies arithmetic. Its maximum here is 0.397.
- **Tightness.** The median bound is 38 times the realized error (range
  6 to 83 across benchmarks).
- **Non-vacuity.** The bound beats the trivial limit, the largest deviation of any single model from the population mean, in a
  median of 0.0% of simulated panels, and it falls below one population SD in
  0.0%.
- **Which pairs set ell.** It comes from one extreme pair, and only a median 0.02% of pairs have
  over half that slope. The 95th percentile of error per unit of gap is a median 0.083 of ell.
- **Close models, different behavior.** Among pairs sharing a base model, a median 13.8% differ by
  over one population SD, which makes the smoothness assumption weakest where the geometry treats models as nearly interchangeable.

## Bound against realized error, simulated panels

| outcome | ell | ell_q90 | max_ratio | median_ratio | median_bound_over_error | median_bound_points | median_error_points | trivial_limit_points | panels_bound_below_trivial_pct | panels_bound_below_one_sd_pct | empirical_slope_over_ell |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IFEval | 2.992 | 0.249 | 0.058 | 0.012 | 83.397 | 438.962 | 4.898 | 47.108 | 0 | 0 | 0.043 |
| BBH | 1.098 | 0.213 | 0.172 | 0.03 | 32.953 | 161.055 | 4.565 | 44.077 | 0 | 0 | 0.099 |
| MATH Lvl 5 | 1.215 | 0.184 | 0.173 | 0.022 | 45.785 | 178.323 | 3.902 | 47.439 | 0 | 0 | 0.068 |
| GPQA | 0.43 | 0.072 | 0.138 | 0.023 | 43.021 | 63.078 | 1.44 | 21.061 | 0 | 0 | 0.067 |
| MUSR | 0.621 | 0.071 | 0.07 | 0.018 | 54.061 | 91.11 | 1.632 | 26.345 | 0 | 0 | 0.047 |
| MMLU-PRO | 1.015 | 0.195 | 0.18 | 0.032 | 31.449 | 148.859 | 4.454 | 50.456 | 0 | 0 | 0.107 |
| salad_base_safety | 0.377 | 0.242 | 0.397 | 0.176 | 5.697 | 46.318 | 8.295 | 27.079 | 6.833 | 0 | 0.352 |
| salad_attack_safety | 1.406 | 0.249 | 0.149 | 0.04 | 25.188 | 172.948 | 4.887 | 56.943 | 0 | 0 | 0.104 |

## Audited papers

| outcome | papers | median_overlap_mass | median_error_points | median_bound_points | median_ratio | max_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| BBH | 42 | 0.75 | 9.803 | 197.611 | 0.051 | 0.179 |
| GPQA | 42 | 0.75 | 2.549 | 77.395 | 0.033 | 0.184 |
| IFEval | 42 | 0.75 | 13.392 | 538.596 | 0.025 | 0.066 |
| MATH Lvl 5 | 42 | 0.75 | 8.903 | 218.798 | 0.04 | 0.162 |
| MMLU-PRO | 42 | 0.75 | 9.671 | 182.647 | 0.053 | 0.173 |
| MUSR | 42 | 0.75 | 2.217 | 111.79 | 0.02 | 0.121 |
| salad_attack_safety | 29 | 0.5 | 4.647 | 212.268 | 0.021 | 0.224 |
| salad_base_safety | 29 | 0.5 | 6.734 | 56.848 | 0.153 | 0.397 |

## Behavior by lineage relation, median across benchmarks

| relation | median_difference_sd | pairs_differing_over_one_sd_pct | pairs_differing_over_two_sd_pct |
| --- | --- | --- | --- |
| same_base | 0.42 | 13.761 | 3.675 |
| same_series | 0.853 | 44.247 | 15.935 |
| same_family | 1.032 | 51.178 | 21.336 |
| different_family | 0.961 | 48.685 | 19.848 |

## Largest behavioral differences between lineage-close variants

| outcome | variant_a | variant_b | relation | score_a_points | score_b_points | difference_sd |
| --- | --- | --- | --- | --- | --- | --- |
| MUSR | variant::qwen::qwen2::78b::instruct | variant::qwen::qwen2::3x1.6b::instruct | same_series | 36.499 | 0.944 | 7.446 |
| MUSR | variant::qwen::qwen2::78b::instruct | variant::qwen::qwen2::4b::unknown-role | same_series | 36.499 | 2.103 | 7.204 |
| MUSR | variant::qwen::qwen2::0.5b::instruct | variant::qwen::qwen2::78b::instruct | same_series | 2.269 | 36.499 | 7.169 |
| MUSR | variant::qwen::qwen2::0.5b::unknown-role | variant::qwen::qwen2::78b::instruct | same_series | 2.354 | 36.499 | 7.151 |
| MUSR | variant::qwen::qwen2::78b::instruct | variant::qwen::qwen2::15b::unknown-role | same_series | 36.499 | 2.409 | 7.14 |
| MUSR | variant::qwen::qwen2::78b::instruct | variant::qwen::qwen2::7b::math | same_series | 36.499 | 2.617 | 7.096 |
| MUSR | variant::qwen::qwen2::78b::instruct | variant::qwen::qwen2::5b::unknown-role | same_series | 36.499 | 2.624 | 7.094 |
| MUSR | variant::qwen::qwen2::78b::instruct | variant::qwen::qwen2::1b::base | same_series | 36.499 | 2.734 | 7.071 |
| MUSR | variant::qwen::qwen2::78b::instruct | variant::qwen::qwen2::900m::base | same_series | 36.499 | 2.966 | 7.023 |
| MUSR | variant::qwen::qwen2::78b::instruct | variant::qwen::qwen2::3b::domain | same_series | 36.499 | 3.195 | 6.975 |

Variants here are taxonomy leaves, so a large difference can also reflect a taxonomy error (two unrelated models in one
leaf) rather than real behavioral divergence. These rows are worth inspecting by hand before citing them.

## Discussion for the paper

The guarantee explains why the adoption gap is the right quantity to compute: it is the exact worst case over all
lineage-smooth properties, so a panel with a small gap cannot be badly wrong about any property that varies smoothly on the tree.
It does not supply a useful numeric error bar. Worst-case ell is driven by a few extreme pairs, the resulting bound sits well
above realized errors, and it is often no better than limits any panel obeys. The honest claim is qualitative and relative, in
that lower gap tightens the worst case and, as P1.1 measures, tends to go with lower error, rather than a promise about error size.
The quantile slopes show the gap between the worst case and typical behavior, but `ell_q90` and `ell_q99` are descriptive and carry no
guarantee: the coverage they reach on these panels is an empirical observation, not a certificate.

## Files

`bound_tightness_by_benchmark.csv`, `behavior_by_lineage_relation.csv`, `lineage_close_pairs_with_large_differences.csv`,
`simulated_panel_bounds.csv`, `audited_panel_bounds.csv`, and the figures `bound_tightness` and `behavior_by_lineage_relation`.
