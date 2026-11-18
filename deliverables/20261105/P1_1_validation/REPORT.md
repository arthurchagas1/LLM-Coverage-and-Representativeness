# P1.1 Validation under model reuse, release 20261105

## Summary

Mean incremental fit from the adoption gap, with family-cluster 95% intervals. Capability and safety are shown apart because
the safety population, with 29 variants, can send R2 far below zero and dominate a joint mean.

| Benchmarks | Separation | Mean delta R2 | 95% interval | Positive benchmarks | Interval above zero |
| --- | --- | --- | --- | --- | --- |
| all | size | -0.003 | [-0.004, 0.180] | 4/8 | 0/8 |
| all | strategy | 0.004 | [-0.087, 0.923] | 5/8 | 0/8 |
| all | family_disjoint | -4.211 | [-708.811, -0.176] | 0/8 | 0/8 |
| capability | size | -0.006 | [-0.005, 0.229] | 3/6 | 0/6 |
| capability | strategy | -0.014 | [-0.116, 1.212] | 3/6 | 0/6 |
| capability | family_disjoint | -0.649 | [-35.030, 0.211] | 0/6 | 0/6 |
| safety | size | 0.009 | [-0.046, 0.207] | 1/2 | 0/2 |
| safety | strategy | 0.057 | [-0.150, 0.426] | 2/2 | 0/2 |
| safety | family_disjoint | -14.899 | [-2834.428, -0.042] | 0/2 | 0/2 |

Within a panel size, the highest-gap third of capability panels has a median 1.58 times the mean error of
the lowest-gap third (`error_by_gap_tercile.csv`).

The panel-bootstrap intervals in `robustness_by_benchmark.csv` reproduce the original method. Compare them with the family
interval on the same row to see how much the dependence between panels matters.

R2 is unstable in the family-disjoint and strategy schemes: a test set with little variance sends it far below zero, and a
very poor control model pushes delta R2 above one. Check `r2_base` before reading delta R2 there, and prefer the rank
association below, which needs no fitted model.

## Rank association between gap and error, within panel size

Positive means higher-gap panels have larger errors. `disjoint_families` uses panels from one side of a family split at a time.

| group | outcome | panels | within_size_spearman | family_low | family_high |
| --- | --- | --- | --- | --- | --- |
| capability | IFEval | all_models | 0.134 | -0.005 | 0.469 |
| capability | IFEval | disjoint_families | 0.123 | -0.184 | 0.36 |
| capability | BBH | all_models | 0.282 | 0.126 | 0.523 |
| capability | BBH | disjoint_families | 0.185 | -0.045 | 0.378 |
| capability | MATH Lvl 5 | all_models | 0.11 | -0.035 | 0.493 |
| capability | MATH Lvl 5 | disjoint_families | 0.008 | -0.241 | 0.296 |
| capability | GPQA | all_models | 0.269 | 0.078 | 0.488 |
| capability | GPQA | disjoint_families | 0.18 | -0.032 | 0.363 |
| capability | MUSR | all_models | 0.229 | 0.106 | 0.433 |
| capability | MUSR | disjoint_families | 0.142 | -0.034 | 0.308 |
| capability | MMLU-PRO | all_models | 0.233 | 0.062 | 0.526 |
| capability | MMLU-PRO | disjoint_families | 0.142 | -0.095 | 0.372 |
| safety | salad_base_safety | all_models | 0.608 | 0.265 | 0.902 |
| safety | salad_base_safety | disjoint_families | 0.551 | 0.028 | 0.795 |
| safety | salad_attack_safety | all_models | 0.231 | -0.061 | 0.815 |
| safety | salad_attack_safety | disjoint_families | 0.009 | -0.213 | 0.639 |

## Sampling noise of the design

Mean delta R2 when only the random panel draws change (100 redraws, same population). A difference between two runs
smaller than this range says nothing about the coordinate.

| scope | scheme | predictor | mean_of_means | sd_of_means | p2_5 | p97_5 | median_benchmarks_positive | min_benchmarks_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | size | adoption_gap | 0.007 | 0.01 | -0.014 | 0.026 | 4 | 1 |
| all | size | breadth | 0.004 | 0.005 | -0.007 | 0.015 | 4 | 1 |
| all | size | structural_linf | 0.023 | 0.007 | 0.01 | 0.037 | 6 | 4 |
| all | size | coverage_q0 | -0.138 | 0.046 | -0.267 | -0.074 | 0 | 0 |
| all | size | coverage_q1 | -0.017 | 0.009 | -0.036 | -0.002 | 2 | 0 |
| all | size | coverage_delta_1_0 | -0.014 | 0.008 | -0.032 | -0.001 | 2 | 0 |
| all | size | full_profile | -0.089 | 0.044 | -0.192 | -0.024 | 1 | 0 |
| all | strategy | adoption_gap | 0.054 | 0.04 | -0.053 | 0.127 | 6 | 1 |
| all | strategy | breadth | 0.006 | 0.014 | -0.025 | 0.031 | 3 | 1 |
| all | strategy | structural_linf | 0.082 | 0.037 | 0.016 | 0.155 | 6 | 4 |
| all | strategy | coverage_q0 | -0.174 | 0.05 | -0.28 | -0.083 | 0 | 0 |
| all | strategy | coverage_q1 | -0.017 | 0.021 | -0.063 | 0.015 | 4 | 1 |
| all | strategy | coverage_delta_1_0 | -0.021 | 0.021 | -0.065 | 0.014 | 3 | 1 |
| all | strategy | full_profile | -0.042 | 0.066 | -0.165 | 0.058 | 3.5 | 0 |
| capability | size | adoption_gap | 0.01 | 0.013 | -0.016 | 0.034 | 3.5 | 0 |
| capability | size | breadth | 0.001 | 0.005 | -0.006 | 0.012 | 3 | 0 |
| capability | size | structural_linf | 0.032 | 0.01 | 0.014 | 0.052 | 6 | 4 |
| capability | size | coverage_q0 | -0.138 | 0.062 | -0.296 | -0.063 | 0 | 0 |
| capability | size | coverage_q1 | -0.023 | 0.011 | -0.049 | -0.003 | 1 | 0 |
| capability | size | coverage_delta_1_0 | -0.02 | 0.011 | -0.046 | -0.002 | 1 | 0 |
| capability | size | full_profile | -0.09 | 0.056 | -0.223 | -0.016 | 1 | 0 |
| capability | strategy | adoption_gap | 0.069 | 0.053 | -0.068 | 0.164 | 5 | 1 |
| capability | strategy | breadth | -0.007 | 0.018 | -0.048 | 0.022 | 2 | 0 |
| capability | strategy | structural_linf | 0.122 | 0.049 | 0.031 | 0.214 | 6 | 4 |
| capability | strategy | coverage_q0 | -0.154 | 0.043 | -0.232 | -0.07 | 0 | 0 |
| capability | strategy | coverage_q1 | -0.031 | 0.03 | -0.093 | 0.018 | 2.5 | 0 |
| capability | strategy | coverage_delta_1_0 | -0.033 | 0.029 | -0.093 | 0.014 | 2 | 0 |
| capability | strategy | full_profile | -0.009 | 0.06 | -0.133 | 0.1 | 3 | 0 |
| safety | size | adoption_gap | -0.004 | 0.015 | -0.032 | 0.024 | 1 | 0 |
| safety | size | breadth | 0.011 | 0.015 | -0.019 | 0.044 | 1 | 0 |
| safety | size | structural_linf | -0.003 | 0.009 | -0.022 | 0.014 | 1 | 0 |
| safety | size | coverage_q0 | -0.14 | 0.052 | -0.242 | -0.055 | 0 | 0 |
| safety | size | coverage_q1 | -7.4e-04 | 0.008 | -0.019 | 0.014 | 1 | 0 |
| safety | size | coverage_delta_1_0 | 0.006 | 0.008 | -0.007 | 0.024 | 1.5 | 0 |
| safety | size | full_profile | -0.084 | 0.062 | -0.242 | 0.015 | 0 | 0 |
| safety | strategy | adoption_gap | 0.009 | 0.021 | -0.028 | 0.059 | 1 | 0 |
| safety | strategy | breadth | 0.045 | 0.021 | 0.007 | 0.085 | 1 | 1 |
| safety | strategy | structural_linf | -0.036 | 0.019 | -0.074 | -0.006 | 0 | 0 |
| safety | strategy | coverage_q0 | -0.234 | 0.157 | -0.628 | 0.017 | 0 | 0 |
| safety | strategy | coverage_q1 | 0.025 | 0.031 | -0.034 | 0.089 | 1 | 0 |
| safety | strategy | coverage_delta_1_0 | 0.017 | 0.024 | -0.027 | 0.072 | 1 | 0 |
| safety | strategy | full_profile | -0.14 | 0.184 | -0.616 | 0.118 | 1 | 0 |

## All coordinates

| scope | scheme | predictor | mean_delta_r2 | mean_delta_r2_low | mean_delta_r2_high | benchmarks_positive | benchmarks_ci_above_zero | mean_delta_mae | mean_delta_mae_low | mean_delta_mae_high | benchmarks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | size | Adoption gap | -0.003 | -0.004 | 0.18 | 4 | 0 | 0.001 | -0.002 | 0.032 | 8 |
| all | size | Lineage breadth | 0.002 | -0.156 | 0.048 | 4 | 0 | 5.5e-04 | -0.025 | 0.007 | 8 |
| all | size | Structural imbalance | 0.027 | -0.006 | 0.109 | 8 | 0 | 0.007 | -0.002 | 0.022 | 8 |
| all | size | C(0) | -0.102 | -0.567 | -0.043 | 0 | 0 | -0.016 | -0.097 | -0.007 | 8 |
| all | size | C(1) | -0.024 | -0.047 | 0.071 | 2 | 0 | -0.003 | -0.01 | 0.014 | 8 |
| all | size | C(1) - C(0) | -0.021 | -0.023 | 0.083 | 2 | 0 | -0.002 | -0.004 | 0.012 | 8 |
| all | size | Full profile | -0.057 | -0.363 | 0.136 | 1 | 0 | -0.005 | -0.062 | 0.02 | 8 |
| all | strategy | Adoption gap | 0.004 | -0.087 | 0.923 | 5 | 0 | 0.005 | -0.021 | 0.108 | 8 |
| all | strategy | Lineage breadth | 0.019 | -0.067 | 0.218 | 5 | 0 | 0.004 | -0.012 | 0.028 | 8 |
| all | strategy | Structural imbalance | 0.061 | -0.101 | 0.687 | 5 | 0 | 0.009 | -0.017 | 0.086 | 8 |
| all | strategy | C(0) | -0.116 | -0.337 | -0.001 | 1 | 0 | -0.015 | -0.036 | 0.006 | 8 |
| all | strategy | C(1) | -0.023 | -0.093 | 0.462 | 2 | 0 | 0.002 | -0.023 | 0.049 | 8 |
| all | strategy | C(1) - C(0) | -0.03 | -0.085 | 0.453 | 2 | 0 | -0.001 | -0.018 | 0.046 | 8 |
| all | strategy | Full profile | -0.011 | -0.28 | 0.936 | 4 | 0 | 0.003 | -0.041 | 0.114 | 8 |
| all | family_disjoint | Adoption gap | -4.211 | -708.811 | -0.176 | 0 | 0 | -0.204 | -8.039 | -0.013 | 8 |
| all | family_disjoint | Lineage breadth | -1.382 | -4743.511 | 0.007 | 0 | 0 | -0.053 | -8.534 | 0.005 | 8 |
| all | family_disjoint | Structural imbalance | -0.038 | -10072.463 | 0.154 | 2 | 0 | -0.004 | -6.972 | 0.027 | 8 |
| all | family_disjoint | C(0) | 0.008 | -0.484 | 0.168 | 5 | 0 | 9.9e-04 | -0.044 | 0.041 | 8 |
| all | family_disjoint | C(1) | -7.31 | -479.788 | -0.028 | 0 | 0 | -0.284 | -8.117 | -0.006 | 8 |
| all | family_disjoint | C(1) - C(0) | -1.1 | -9.163 | 0.084 | 0 | 0 | -0.093 | -0.651 | 0.012 | 8 |
| all | family_disjoint | Full profile | -11.643 | -20222.242 | -2.944 | 0 | 0 | -0.368 | -17.139 | -0.15 | 8 |
| capability | size | Adoption gap | -0.006 | -0.005 | 0.229 | 3 | 0 | -1.5e-04 | -0.004 | 0.034 | 6 |
| capability | size | Lineage breadth | -0.002 | -0.203 | 0.054 | 3 | 0 | -1.3e-04 | -0.028 | 0.007 | 6 |
| capability | size | Structural imbalance | 0.029 | -0.011 | 0.138 | 6 | 0 | 0.009 | -0.003 | 0.023 | 6 |
| capability | size | C(0) | -0.094 | -0.685 | -0.029 | 0 | 0 | -0.014 | -0.095 | -0.005 | 6 |
| capability | size | C(1) | -0.039 | -0.035 | 0.085 | 0 | 0 | -0.006 | -0.007 | 0.013 | 6 |
| capability | size | C(1) - C(0) | -0.037 | -0.03 | 0.084 | 0 | 0 | -0.006 | -0.006 | 0.014 | 6 |
| capability | size | Full profile | -0.076 | -0.455 | 0.17 | 0 | 0 | -0.01 | -0.073 | 0.024 | 6 |
| capability | strategy | Adoption gap | -0.014 | -0.116 | 1.212 | 3 | 0 | 0.005 | -0.028 | 0.136 | 6 |
| capability | strategy | Lineage breadth | 0.008 | -0.087 | 0.274 | 4 | 0 | 0.002 | -0.015 | 0.032 | 6 |
| capability | strategy | Structural imbalance | 0.105 | -0.133 | 0.929 | 5 | 0 | 0.016 | -0.022 | 0.112 | 6 |
| capability | strategy | C(0) | -0.086 | -0.387 | -0.013 | 1 | 0 | -0.009 | -0.03 | -6.7e-04 | 6 |
| capability | strategy | C(1) | -0.063 | -0.118 | 0.578 | 0 | 0 | -0.009 | -0.028 | 0.057 | 6 |
| capability | strategy | C(1) - C(0) | -0.064 | -0.117 | 0.588 | 0 | 0 | -0.009 | -0.027 | 0.058 | 6 |
| capability | strategy | Full profile | -0.005 | -0.344 | 1.235 | 3 | 0 | 0.003 | -0.047 | 0.142 | 6 |
| capability | family_disjoint | Adoption gap | -0.649 | -35.03 | 0.211 | 0 | 0 | -0.06 | -1.153 | 0.03 | 6 |
| capability | family_disjoint | Lineage breadth | -1.52 | -306.96 | 0.013 | 0 | 0 | -0.035 | -1.82 | 0.004 | 6 |
| capability | family_disjoint | Structural imbalance | -0.008 | -0.254 | 0.201 | 2 | 0 | 7.1e-04 | -0.034 | 0.027 | 6 |
| capability | family_disjoint | C(0) | 0.018 | -0.086 | 0.133 | 5 | 0 | 0.003 | -0.015 | 0.017 | 6 |
| capability | family_disjoint | C(1) | -0.248 | -18.601 | 0.128 | 0 | 0 | -0.027 | -0.835 | 0.019 | 6 |
| capability | family_disjoint | C(1) - C(0) | -0.249 | -10.751 | 0.119 | 0 | 0 | -0.027 | -0.664 | 0.016 | 6 |
| capability | family_disjoint | Full profile | -1.817 | -289.924 | -0.024 | 0 | 0 | -0.099 | -2.297 | 0.003 | 6 |
| safety | size | Adoption gap | 0.009 | -0.046 | 0.207 | 1 | 0 | 0.005 | -0.006 | 0.074 | 2 |
| safety | size | Lineage breadth | 0.015 | -0.066 | 0.146 | 1 | 0 | 0.003 | -0.053 | 0.022 | 2 |
| safety | size | Structural imbalance | 0.02 | -0.023 | 0.176 | 2 | 0 | 0.004 | -0.007 | 0.039 | 2 |
| safety | size | C(0) | -0.128 | -0.832 | 0.004 | 0 | 0 | -0.025 | -0.236 | 0.004 | 2 |
| safety | size | C(1) | 0.021 | -0.203 | 0.093 | 2 | 0 | 0.009 | -0.033 | 0.039 | 2 |
| safety | size | C(1) - C(0) | 0.028 | -0.015 | 0.223 | 2 | 0 | 0.009 | -0.008 | 0.03 | 2 |
| safety | size | Full profile | 0.003 | -0.496 | 0.394 | 1 | 0 | 0.009 | -0.122 | 0.081 | 2 |
| safety | strategy | Adoption gap | 0.057 | -0.15 | 0.426 | 2 | 0 | 0.008 | -0.041 | 0.082 | 2 |
| safety | strategy | Lineage breadth | 0.052 | -0.097 | 0.193 | 1 | 0 | 0.013 | -0.03 | 0.05 | 2 |
| safety | strategy | Structural imbalance | -0.069 | -0.115 | 0.26 | 0 | 0 | -0.015 | -0.025 | 0.075 | 2 |
| safety | strategy | C(0) | -0.204 | -0.423 | 0.157 | 0 | 0 | -0.033 | -0.084 | 0.051 | 2 |
| safety | strategy | C(1) | 0.094 | -0.124 | 0.388 | 2 | 0 | 0.035 | -0.042 | 0.073 | 2 |
| safety | strategy | C(1) - C(0) | 0.071 | -0.058 | 0.314 | 2 | 0 | 0.023 | -0.022 | 0.047 | 2 |
| safety | strategy | Full profile | -0.028 | -0.621 | 0.65 | 1 | 0 | 0.003 | -0.156 | 0.137 | 2 |
| safety | family_disjoint | Adoption gap | -14.899 | -2834.428 | -0.042 | 0 | 0 | -0.636 | -30.397 | -0.005 | 2 |
| safety | family_disjoint | Lineage breadth | -0.971 | -15420.347 | 0.629 | 0 | 0 | -0.106 | -31.615 | 0.062 | 2 |
| safety | family_disjoint | Structural imbalance | -0.13 | -40289.879 | 0.475 | 0 | 0 | -0.018 | -27.872 | 0.098 | 2 |
| safety | family_disjoint | C(0) | -0.023 | -1.911 | 0.644 | 0 | 0 | -0.004 | -0.18 | 0.157 | 2 |
| safety | family_disjoint | C(1) | -28.495 | -1919.057 | 0.126 | 0 | 0 | -1.054 | -32.412 | 0.017 | 2 |
| safety | family_disjoint | C(1) - C(0) | -3.656 | -16.476 | 0.701 | 0 | 0 | -0.288 | -1.237 | 0.125 | 2 |
| safety | family_disjoint | Full profile | -41.12 | -80877.691 | -1.647 | 0 | 0 | -1.174 | -66.651 | -0.125 | 2 |

## Adoption gap by benchmark

| outcome | scheme | delta_r2 | delta_r2_family_low | delta_r2_family_high | delta_mae | delta_mae_benchmark_points |
| --- | --- | --- | --- | --- | --- | --- |
| IFEval | size | -0.021 | -0.014 | 0.238 | -0.003 | -0.049 |
| BBH | size | 0.004 | -0.024 | 0.293 | 0.005 | 0.079 |
| MATH Lvl 5 | size | -0.057 | -0.03 | 0.281 | -0.01 | -0.124 |
| GPQA | size | 0.003 | -0.026 | 0.259 | 8.6e-04 | 0.004 |
| MUSR | size | 0.044 | -0.003 | 0.192 | 0.004 | 0.02 |
| MMLU-PRO | size | -0.01 | -0.029 | 0.298 | 0.001 | 0.023 |
| IFEval | strategy | -0.002 | -0.166 | 1.078 | 0.005 | 0.089 |
| BBH | strategy | 0.058 | -0.23 | 1.33 | 0.023 | 0.377 |
| MATH Lvl 5 | strategy | -0.215 | -0.231 | 2.129 | -0.032 | -0.412 |
| GPQA | strategy | -0.038 | -0.229 | 1.196 | 0.01 | 0.051 |
| MUSR | strategy | 0.026 | -0.089 | 0.874 | 0.003 | 0.014 |
| MMLU-PRO | strategy | 0.087 | -0.26 | 1.706 | 0.018 | 0.284 |
| IFEval | family_disjoint | -0.325 | -29.424 | 0.17 | -0.044 | -0.802 |
| BBH | family_disjoint | -0.645 | -41.369 | 0.448 | -0.063 | -1.03 |
| MATH Lvl 5 | family_disjoint | -1.083 | -17.408 | 0.547 | -0.094 | -1.212 |
| GPQA | family_disjoint | -0.402 | -36.885 | 0.467 | -0.039 | -0.199 |
| MUSR | family_disjoint | -0.735 | -34.609 | 0.518 | -0.056 | -0.267 |
| MMLU-PRO | family_disjoint | -0.705 | -33.721 | 0.395 | -0.066 | -1.042 |
| salad_base_safety | size | -0.003 | -0.091 | 0.193 | -0.001 | -0.014 |
| salad_attack_safety | size | 0.02 | -0.037 | 0.275 | 0.01 | 0.139 |
| salad_base_safety | strategy | 0.034 | -0.114 | 0.353 | 8.0e-04 | 0.011 |
| salad_attack_safety | strategy | 0.08 | -0.26 | 0.524 | 0.015 | 0.206 |
| salad_base_safety | family_disjoint | -15.065 | -1905.822 | 0.277 | -0.751 | -10.004 |
| salad_attack_safety | family_disjoint | -14.732 | -2914.167 | 0.495 | -0.521 | -7.072 |

## Influence of large families (leave one family out, size scheme)

| group | dropped_family | variants_dropped | adoption_mass_dropped | mean_delta_r2 | benchmarks_positive |
| --- | --- | --- | --- | --- | --- |
| capability | gpt-neox | 19 | 0.032 | 0.045 | 5 |
| capability | gpt2 | 13 | 0.244 | 0.088 | 6 |
| capability | llama | 165 | 0.168 | 0.068 | 6 |
| capability | mistral | 77 | 0.055 | 0.009 | 5 |
| capability | qwen | 146 | 0.356 | 0.016 | 6 |
| capability | yi | 19 | 0.038 | 0.025 | 5 |
| safety | gemma | 2 | 0.009 | -0.002 | 1 |
| safety | internlm | 4 | 0.031 | 0.017 | 1 |
| safety | llama | 9 | 0.218 | 0.015 | 2 |
| safety | mistral | 2 | 0.663 | 0.015 | 2 |
| safety | qwen | 9 | 0.063 | -0.026 | 0 |
| safety | yi | 2 | 0.008 | 0.01 | 1 |

## Design

- Panels: 50 per size-by-strategy cell, with the original seeds. Family-disjoint uses 25 per cell and side over 10 random splits.
- Family-cluster bootstrap: 500 replicates. Each draws families with replacement, reweights the population, recomputes
  targets and benchmark means, redraws every panel, and reruns all three separations.
- `delta_mae` is the control model's MAE minus the augmented model's MAE, in standardized error, so positive means better.
  `delta_mae_benchmark_points` converts it to points of that benchmark.
- The safety population is small, with few families, so its intervals are wide and its family-disjoint splits are thin.
