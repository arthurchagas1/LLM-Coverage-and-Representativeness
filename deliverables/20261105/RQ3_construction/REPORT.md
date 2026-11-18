# RQ3 Coverage-aware construction, release 20261105

## Trade-off between adoption gap and breadth

Greedy panels maximizing lambda * (scaled gap reduction) + (1 - lambda) * (scaled breadth gain), under each set of
constraints. Medians over papers, relative to the observed panel. `papers_both_improved` counts papers whose panel
has both a smaller gap and a larger breadth than the observed one.

| rung | lambda | papers | median_gap_reduction_pct | median_breadth_gain_pct | papers_both_improved | papers_structural_improved | median_param_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| core | 1 | 46 | 41.634 | 1.525 | 31 | 19 | 0.065 |
| core | 0.9 | 46 | 41.632 | 1.733 | 32 | 19 | 0.065 |
| core | 0.75 | 46 | 41.632 | 1.949 | 32 | 19 | 0.065 |
| core | 0.5 | 46 | 28.545 | 199.882 | 44 | 29 | 0.21 |
| core | 0.25 | 46 | -8.865 | 499.772 | 1 | 32 | 0.452 |
| core | 0.1 | 46 | -9.792 | 529.514 | 0 | 32 | 0.452 |
| core | 0 | 46 | -9.795 | 529.514 | 0 | 33 | 0.292 |
| cost | 1 | 46 | 19.28 | 1.872 | 32 | 15 | 1.002 |
| cost | 0.9 | 46 | 19.28 | 2.045 | 33 | 15 | 1.003 |
| cost | 0.75 | 46 | 19.28 | 7.069 | 38 | 15 | 1.019 |
| cost | 0.5 | 46 | 14.716 | 97.822 | 43 | 15 | 1.041 |
| cost | 0.25 | 46 | 1.51 | 292.645 | 28 | 15 | 1.02 |
| cost | 0.1 | 46 | -4.631 | 361.874 | 10 | 15 | 1.024 |
| cost | 0 | 46 | -7.736 | 367.71 | 3 | 16 | 1.032 |
| adoption_floor | 1 | 46 | 5.642 | 0.741 | 28 | 12 | 1 |
| adoption_floor | 0.9 | 46 | 5.641 | 1.797 | 31 | 12 | 1 |
| adoption_floor | 0.75 | 46 | 5.594 | 4.233 | 35 | 12 | 1.001 |
| adoption_floor | 0.5 | 46 | 4.75 | 31.811 | 38 | 13 | 1.001 |
| adoption_floor | 0.25 | 46 | 2.611 | 42.412 | 36 | 13 | 1 |
| adoption_floor | 0.1 | 46 | 0.386 | 114.839 | 27 | 12 | 1.001 |
| adoption_floor | 0 | 46 | -4.263 | 121.967 | 3 | 13 | 1.002 |

## Few swaps from the observed panel

Best replacements under the same constraints, each keeping the replaced model's size bucket and role.
`median_share_of_greedy_rebuild` compares the swap reduction with the full greedy rebuild of the ladder (capped at 10).

| rung | swaps | papers | median_gap_reduction_pct | papers_improved | median_share_of_greedy_rebuild | papers_at_or_beyond_greedy |
| --- | --- | --- | --- | --- | --- | --- |
| cost | 1 | 46 | 5.31 | 45 | 0.395 | 1 |
| cost | 2 | 46 | 9.278 | 45 | 0.674 | 8 |
| cost | 3 | 46 | 12.468 | 45 | 0.822 | 12 |
| adoption_floor | 1 | 46 | 2.214 | 43 | 0.579 | 7 |
| adoption_floor | 2 | 46 | 3.673 | 43 | 0.91 | 13 |
| adoption_floor | 3 | 46 | 4.156 | 43 | 0.977 | 18 |

## Notes

- Lambda scales each objective by what the pure adoption-greedy and breadth-greedy panels achieve for that paper and
  pool, so 0.5 weighs the two relative improvements equally.
- Swaps never add a variant already in the panel and stop when no swap reduces the gap.
