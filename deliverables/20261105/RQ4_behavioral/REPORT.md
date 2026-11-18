# RQ4 behavioral validation, release 20261105

Scored population: 624 variants with all six Open LLM Leaderboard capability scores, from
36 families. Paper worlds: 44 of 46 audited
papers have at least twice their panel size in scored variants available by their cutoff.

## Primary endpoint (paper worlds, adoption target, lineage-spread vs lineage-blind)

Median RMSE ratio 0.932 (95% CI over papers 0.916-0.944);
44 of 44 papers below one; Wilcoxon p = 1.1e-13.
Inclusion probabilities are equal across designs up to Monte Carlo error (largest deviation of mean multiplicity
from n t(i): 0.034).

## A1. Semivariance by lineage relation (mean over benchmarks; 1 = random pairs)

| scores | relation | pairs | semivariance | null_mean | null_p975 | p_lower |
| --- | --- | --- | --- | --- | --- | --- |
| raw | same base | 585 | 0.277 | 1.003 | 1.109 | 0.001 |
| raw | same series | 10091 | 0.865 | 1.004 | 1.085 | 0.001 |
| raw | same family | 18244 | 1.057 | 1.001 | 1.072 | 0.934 |
| raw | different family | 165456 | 1.006 | 1.002 | 1.012 | 0.827 |
| scale-adjusted | same base | 585 | 0.277 | 0.619 | 0.688 | 0.001 |
| scale-adjusted | same series | 10091 | 0.469 | 0.62 | 0.675 | 0.001 |
| scale-adjusted | same family | 18244 | 0.592 | 0.619 | 0.67 | 0.159 |
| scale-adjusted | different family | 165456 | 0.632 | 0.619 | 0.625 | 1 |

## A2. Leave-one-out prediction of benchmark scores (mean R2 over benchmarks)

| predictor | loo_r2 |
| --- | --- |
| closest lineage relatives | 0.489 |
| size bucket | 0.365 |
| size bucket + lineage relatives | 0.506 |
| closest relatives, permuted tree (mean of 200) | -0.438 |

## B. Paper worlds, all targets and designs

| target | design | papers | median_ratio | ci_low | ci_high | q25 | q75 | papers_below_one | wilcoxon_p | large_error_blind | large_error_design | gap_ratio_primary | gap_ratio_topological |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adoption | lineage | 44 | 0.932 | 0.916 | 0.944 | 0.906 | 0.956 | 44 | 1.1e-13 | 0.067 | 0.056 | 0.942 | 0.929 |
| adoption | scale | 44 | 0.774 | 0.761 | 0.786 | 0.745 | 0.835 | 43 | 2.3e-13 | 0.067 | 0.039 | 0.995 | 0.995 |
| adoption | scale_lineage | 44 | 0.769 | 0.74 | 0.783 | 0.723 | 0.826 | 44 | 1.1e-13 | 0.067 | 0.036 | 0.988 | 0.986 |
| breadth | lineage | 44 | 0.946 | 0.927 | 0.971 | 0.923 | 0.989 | 36 | 6.4e-07 | 0.086 | 0.071 | 0.938 | 0.947 |
| breadth | scale | 44 | 0.898 | 0.876 | 0.913 | 0.865 | 0.948 | 37 | 1.1e-09 | 0.086 | 0.066 | 0.991 | 0.993 |
| breadth | scale_lineage | 44 | 0.886 | 0.875 | 0.922 | 0.848 | 0.987 | 35 | 2.0e-08 | 0.086 | 0.066 | 0.97 | 0.975 |
| uniform | lineage | 44 | 0.94 | 0.916 | 0.951 | 0.899 | 0.956 | 42 | 2.2e-12 | 0.139 | 0.119 | 0.949 | 0.93 |
| uniform | scale | 44 | 0.852 | 0.822 | 0.863 | 0.818 | 0.883 | 44 | 1.1e-13 | 0.139 | 0.085 | 0.998 | 0.997 |
| uniform | scale_lineage | 44 | 0.845 | 0.813 | 0.869 | 0.796 | 0.888 | 44 | 1.1e-13 | 0.139 | 0.081 | 0.993 | 0.991 |

## B. Full scored population, family-bootstrap 95% intervals

| target | n | design | ratio | ci_low | ci_high | large_error_rate | gap_primary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adoption | 5 | lineage | 0.982 | 0.852 | 1.027 | 0.175 | 1.117 |
| adoption | 5 | scale | 0.75 | 0.636 | 0.881 | 0.079 | 1.181 |
| adoption | 5 | scale_lineage | 0.727 | 0.625 | 0.889 | 0.069 | 1.186 |
| adoption | 10 | lineage | 0.942 | 0.836 | 1.013 | 0.03 | 0.916 |
| adoption | 10 | scale | 0.771 | 0.633 | 0.856 | 0.012 | 0.968 |
| adoption | 10 | scale_lineage | 0.786 | 0.62 | 0.849 | 0.014 | 0.974 |
| adoption | 20 | lineage | 0.889 | 0.802 | 0.973 | 7.9e-04 | 0.673 |
| adoption | 20 | scale | 0.724 | 0.658 | 0.852 | 4.2e-05 | 0.727 |
| adoption | 20 | scale_lineage | 0.691 | 0.626 | 0.851 | 4.2e-05 | 0.706 |
| adoption | 40 | lineage | 0.873 | 0.762 | 0.926 | 0 | 0.485 |
| adoption | 40 | scale | 0.758 | 0.668 | 0.845 | 0 | 0.541 |
| adoption | 40 | scale_lineage | 0.724 | 0.623 | 0.83 | 0 | 0.518 |
| breadth | 5 | lineage | 0.98 | 0.896 | 1.056 | 0.205 | 1.03 |
| breadth | 5 | scale | 0.795 | 0.721 | 0.951 | 0.134 | 1.06 |
| breadth | 5 | scale_lineage | 0.803 | 0.734 | 0.957 | 0.132 | 1.04 |
| breadth | 10 | lineage | 0.945 | 0.863 | 1.038 | 0.05 | 0.73 |
| breadth | 10 | scale | 0.827 | 0.736 | 0.94 | 0.028 | 0.821 |
| breadth | 10 | scale_lineage | 0.83 | 0.739 | 0.937 | 0.03 | 0.772 |
| breadth | 20 | lineage | 0.947 | 0.854 | 1.004 | 0.003 | 0.562 |
| breadth | 20 | scale | 0.867 | 0.75 | 0.929 | 0.001 | 0.606 |
| breadth | 20 | scale_lineage | 0.869 | 0.751 | 0.941 | 9.6e-04 | 0.598 |
| breadth | 40 | lineage | 0.875 | 0.769 | 0.974 | 0 | 0.388 |
| breadth | 40 | scale | 0.874 | 0.754 | 0.928 | 0 | 0.453 |
| breadth | 40 | scale_lineage | 0.867 | 0.724 | 0.931 | 0 | 0.416 |
| uniform | 5 | lineage | 0.944 | 0.905 | 1.039 | 0.251 | 1.644 |
| uniform | 5 | scale | 0.849 | 0.787 | 0.93 | 0.197 | 1.703 |
| uniform | 5 | scale_lineage | 0.847 | 0.79 | 0.96 | 0.2 | 1.711 |
| uniform | 10 | lineage | 0.936 | 0.871 | 1.017 | 0.089 | 1.502 |
| uniform | 10 | scale | 0.811 | 0.736 | 0.903 | 0.052 | 1.583 |
| uniform | 10 | scale_lineage | 0.799 | 0.704 | 0.905 | 0.048 | 1.559 |
| uniform | 20 | lineage | 0.904 | 0.816 | 0.999 | 0.013 | 1.343 |
| uniform | 20 | scale | 0.786 | 0.718 | 0.879 | 0.005 | 1.44 |
| uniform | 20 | scale_lineage | 0.758 | 0.697 | 0.849 | 0.004 | 1.405 |
| uniform | 40 | lineage | 0.867 | 0.751 | 0.959 | 1.7e-04 | 1.134 |
| uniform | 40 | scale | 0.804 | 0.707 | 0.872 | 4.2e-05 | 1.261 |
| uniform | 40 | scale_lineage | 0.739 | 0.676 | 0.818 | 4.2e-05 | 1.204 |

## C. Selecting the lowest-gap panel among 20 iid draws (RMSE ratio to a random pick)

| target | n | selection | ratio |
| --- | --- | --- | --- |
| adoption | 5 | random | 1 |
| adoption | 5 | min gap | 0.775 |
| adoption | 5 | min gap, permuted tree | 0.776 |
| adoption | 5 | max gap | 1.101 |
| adoption | 10 | random | 1 |
| adoption | 10 | min gap | 0.743 |
| adoption | 10 | min gap, permuted tree | 0.76 |
| adoption | 10 | max gap | 1.249 |
| adoption | 20 | random | 1 |
| adoption | 20 | min gap | 0.735 |
| adoption | 20 | min gap, permuted tree | 0.768 |
| adoption | 20 | max gap | 1.317 |
| breadth | 5 | random | 1 |
| breadth | 5 | min gap | 0.762 |
| breadth | 5 | min gap, permuted tree | 0.807 |
| breadth | 5 | max gap | 1.373 |
| breadth | 10 | random | 1 |
| breadth | 10 | min gap | 0.766 |
| breadth | 10 | min gap, permuted tree | 0.837 |
| breadth | 10 | max gap | 1.292 |
| breadth | 20 | random | 1 |
| breadth | 20 | min gap | 0.801 |
| breadth | 20 | min gap, permuted tree | 0.853 |
| breadth | 20 | max gap | 1.233 |
