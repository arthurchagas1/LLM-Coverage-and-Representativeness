# Lineage taxonomy annotation: agreement and accuracy

Items in the sample: 240. Checkpoint used below: first 120 items (batches 1-2).

## Progress

| annotator | items | median_seconds | batch_1 | batch_2 | batch_3 |
| --- | --- | --- | --- | --- | --- |
| 1 | 120 | 26.747 | 60/60 | 60/60 | 0/120 |
| 3 | 14 | 60.117 | 14/60 | 0/60 | 0/120 |
| annotator2 | 120 | 24.148 | 60/60 | 60/60 | 0/120 |

## Agreement at each checkpoint

Batches are nested and stratified, so the first 60 and 120 items have the same stratum mix as the full sample. A
checkpoint counts only the annotators who finished all of its items.

| checkpoint | annotators_complete | field | items | alpha_raw | alpha | alpha_low | alpha_high | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 | 2 | decision | 60 | 0.101 | 0.101 | -0.145 | 0.376 | not reliable |
| 60 | 2 | family | 44 | 0.811 | 0.906 | 0.806 | 0.976 | validated |
| 60 | 2 | series | 44 | 0.654 | 0.885 | 0.787 | 0.976 | validated |
| 60 | 2 | size_bucket | 44 | 0.815 | 1.000 | 1.000 | 1.000 | validated |
| 60 | 2 | size_bucket (interval on rank) | 44 | 1.000 | 1.000 | 1.000 | 1.000 | validated |
| 60 | 2 | role | 44 | 0.322 | 0.639 | 0.436 | 0.809 | not reliable |
| 60 | 2 | artifact | 44 | 0.215 | 0.381 | 0.146 | 0.574 | not reliable |
| 120 | 2 | decision | 120 | -0.019 | 0.086 | -0.110 | 0.281 | not reliable |
| 120 | 2 | family | 86 | 0.807 | 0.914 | 0.841 | 0.963 | validated |
| 120 | 2 | series | 86 | 0.686 | 0.905 | 0.843 | 0.964 | validated |
| 120 | 2 | size_bucket | 86 | 0.854 | 1.000 | 1.000 | 1.000 | validated |
| 120 | 2 | size_bucket (interval on rank) | 84 | 0.965 | 1.000 | 1.000 | 1.000 | validated |
| 120 | 2 | role | 86 | 0.392 | 0.611 | 0.471 | 0.741 | not reliable |
| 120 | 2 | artifact | 86 | 0.500 | 0.592 | 0.445 | 0.723 | not reliable |
| 240 | 0 |  | 0 |  |  |  |  | not reached |

## Agreement between annotators, checkpoint 120 (annotators: 1, annotator2)

Hierarchy fields are compared on items that all annotators marked in scope. Reading fixed in advance:
alpha >= 0.80 validated, 0.667-0.80 tentative, below 0.667 not reliable.

### Raw labels (inter-annotator reliability)

| field | items | percent_agreement | fleiss_kappa | krippendorff_alpha | alpha_low | alpha_high | verdict | gwet_ac1 | ac1_low | ac1_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decision | 120 | 0.750 | -0.023 | -0.019 | -0.166 | 0.146 | not reliable | 0.715 | 0.572 | 0.804 |
| family | 88 | 0.818 | 0.806 | 0.807 | 0.722 | 0.881 | validated | 0.816 | 0.733 | 0.895 |
| series | 88 | 0.693 | 0.684 | 0.686 | 0.579 | 0.777 | tentative | 0.690 | 0.584 | 0.792 |
| size_bucket | 88 | 0.875 | 0.853 | 0.854 | 0.772 | 0.933 | validated | 0.860 | 0.780 | 0.936 |
| size_bucket (interval on rank) | 74 | 0.959 |  | 0.965 | 0.909 | 1.000 | validated |  |  |  |
| role | 88 | 0.580 | 0.389 | 0.392 | 0.223 | 0.537 | not reliable | 0.534 | 0.409 | 0.633 |
| artifact | 88 | 0.659 | 0.497 | 0.500 | 0.358 | 0.632 | not reliable | 0.590 | 0.456 | 0.701 |

### After correcting clear errors

Corrections (`corrections.py`, `corrections_manual.csv`; every change in `corrections_log.csv`), labels changed:

| rule | field | labels_changed |
| --- | --- | --- |
| R-role | role | 11 |
| R-scope | decision | 2 |
| R-size | size_bucket | 31 |
| manual | artifact | 13 |
| manual | decision | 2 |
| manual | family | 14 |
| manual | role | 4 |
| manual | series | 27 |

| field | items | percent_agreement | fleiss_kappa | krippendorff_alpha | alpha_low | alpha_high | verdict | gwet_ac1 | ac1_low | ac1_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decision | 120 | 0.750 | 0.082 | 0.086 | -0.127 | 0.271 | not reliable | 0.711 | 0.604 | 0.807 |
| family | 86 | 0.919 | 0.913 | 0.914 | 0.851 | 0.974 | validated | 0.917 | 0.857 | 0.964 |
| series | 86 | 0.907 | 0.904 | 0.905 | 0.833 | 0.964 | validated | 0.906 | 0.834 | 0.964 |
| size_bucket | 86 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | validated | 1.000 | 1.000 | 1.000 |
| size_bucket (interval on rank) | 84 | 1.000 |  | 1.000 | 1.000 | 1.000 | validated |  |  |  |
| role | 86 | 0.721 | 0.609 | 0.611 | 0.466 | 0.725 | not reliable | 0.689 | 0.565 | 0.784 |
| artifact | 86 | 0.721 | 0.590 | 0.592 | 0.445 | 0.713 | not reliable | 0.664 | 0.536 | 0.776 |

### Cohen's kappa by pair

| field | annotator_a | annotator_b | cohen_kappa |
| --- | --- | --- | --- |
| decision | 1 | annotator2 | 0.124 |
| family | 1 | annotator2 | 0.913 |
| series | 1 | annotator2 | 0.904 |
| size_bucket | 1 | annotator2 | 1.000 |
| role | 1 | annotator2 | 0.616 |
| artifact | 1 | annotator2 | 0.599 |

## Disagreements

60 items disagree on scope or on a hierarchy field. They are listed in `adjudication.csv`
(fill the adjudicated_* columns, then rerun).
Items still without a reference label: 30.

## Taxonomy accuracy against the reference labels

Design-weighted over the four family-size strata (weights = share of the population's repositories):

| field | design_weighted_accuracy | low | high | strata_with_data |
| --- | --- | --- | --- | --- |
| scope | 0.995 | 0.989 | 1.000 | 4 |
| family | 0.905 | 0.782 | 1.000 | 4 |
| series | 0.850 | 0.703 | 0.997 | 4 |
| size_bucket | 0.897 | 0.774 | 1.000 | 4 |
| role | 0.907 | 0.762 | 1.000 | 4 |
| variant | 0.759 | 0.530 | 0.987 | 4 |
| role_fixed | 0.921 | 0.777 | 1.000 | 4 |
| variant_fixed | 0.763 | 0.535 | 0.992 | 4 |

By stratum, with Wilson 95% intervals:

| stratum | field | items | accuracy | low | high |
| --- | --- | --- | --- | --- | --- |
| large | scope | 21 | 1.000 | 0.845 | 1.000 |
| large | family | 21 | 0.905 | 0.711 | 0.973 |
| large | series | 21 | 0.857 | 0.654 | 0.950 |
| large | size_bucket | 21 | 0.905 | 0.711 | 0.973 |
| large | role | 13 | 0.923 | 0.667 | 0.986 |
| large | variant | 13 | 0.769 | 0.497 | 0.918 |
| large | role_fixed | 13 | 0.923 | 0.667 | 0.986 |
| large | variant_fixed | 13 | 0.769 | 0.497 | 0.918 |
| medium | scope | 14 | 0.857 | 0.601 | 0.960 |
| medium | family | 10 | 0.900 | 0.596 | 0.982 |
| medium | series | 10 | 0.600 | 0.313 | 0.832 |
| medium | size_bucket | 12 | 0.667 | 0.391 | 0.862 |
| medium | role | 7 | 0.571 | 0.250 | 0.842 |
| medium | variant | 5 | 0.600 | 0.231 | 0.882 |
| medium | role_fixed | 7 | 0.857 | 0.487 | 0.974 |
| medium | variant_fixed | 5 | 0.600 | 0.231 | 0.882 |
| small | scope | 23 | 0.913 | 0.732 | 0.976 |
| small | family | 17 | 0.941 | 0.730 | 0.990 |
| small | series | 17 | 0.941 | 0.730 | 0.990 |
| small | size_bucket | 21 | 0.905 | 0.711 | 0.973 |
| small | role | 17 | 0.529 | 0.310 | 0.738 |
| small | variant | 14 | 0.357 | 0.163 | 0.612 |
| small | role_fixed | 17 | 0.941 | 0.730 | 0.990 |
| small | variant_fixed | 14 | 0.714 | 0.454 | 0.883 |
| singleton | scope | 13 | 1.000 | 0.772 | 1.000 |
| singleton | family | 12 | 1.000 | 0.757 | 1.000 |
| singleton | series | 11 | 0.909 | 0.623 | 0.984 |
| singleton | size_bucket | 13 | 0.923 | 0.667 | 0.986 |
| singleton | role | 10 | 0.300 | 0.108 | 0.603 |
| singleton | variant | 9 | 0.222 | 0.063 | 0.547 |
| singleton | role_fixed | 10 | 0.900 | 0.596 | 0.982 |
| singleton | variant_fixed | 9 | 0.778 | 0.453 | 0.937 |
| panel | scope | 19 | 1.000 | 0.832 | 1.000 |
| panel | family | 19 | 0.947 | 0.754 | 0.991 |
| panel | series | 19 | 0.895 | 0.686 | 0.971 |
| panel | size_bucket | 19 | 0.947 | 0.754 | 0.991 |
| panel | role | 15 | 0.733 | 0.480 | 0.891 |
| panel | variant | 15 | 0.600 | 0.357 | 0.802 |
| panel | role_fixed | 15 | 0.800 | 0.548 | 0.930 |
| panel | variant_fixed | 15 | 0.667 | 0.417 | 0.848 |

Errors are listed in `taxonomy_errors.csv` (49 field errors).
