# RQ2 Sensitivity of the audit to modelling choices, release 20261105

All alternatives are applied to the historical audit (each paper against the variants available by its cutoff),
changing one choice at a time. Checks: the fast adoption gap matches F.w1 to 8.9e-16; the primary row matches
appendix C to 8.9e-16 in gap and, in uniform-reference percentiles, to 1.1e-16 (gap),
6.6e-17 (breadth), and 8.3e-17 (structural imbalance). Breadth and structural imbalance
do not depend on the target or on the gap's edge lengths, so they are computed only for the other choices.

`kendall_tau` and rank shifts compare paper rankings by adoption gap with the primary ranking, over the papers for
which the alternative is defined. `median_uniform_percentile` is the median share of 120 same-size uniform reference
panels whose gap is at least the observed one; `papers_below_half` counts papers below the median reference.

## By choice

| choice | alternatives | min_tau | max_tau | largest_shift | min_median_percentile | max_median_percentile |
| --- | --- | --- | --- | --- | --- | --- |
| cutoff | 6 | 0.766 | 1 | 22 | 0.292 | 0.712 |
| geometry | 3 | 0.49 | 0.927 | 35 | 0.362 | 1 |
| population | 7 | 0.092 | 0.921 | 39 | 0.138 | 0.979 |
| resolution | 2 | 0.816 | 0.86 | 30 | 0.221 | 0.379 |
| target | 9 | 0.271 | 0.979 | 30 | 0.125 | 0.988 |

## By alternative

| choice | label | papers | kendall_tau | median_rank_shift | max_rank_shift | median_adoption_gap | median_uniform_percentile | papers_below_half | median_breadth_percentile | median_structural_percentile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary | Primary specification | 46 | 1 | 0 | 0 | 1.82 | 0.429 | 27 | 0.008 | 0.008 |
| target | Raw downloads (alpha 0) | 46 | 0.971 | 0 | 5 | 1.821 | 0.429 | 25 |  |  |
| target | Downloads, alpha 0.01 | 46 | 0.971 | 0 | 5 | 1.821 | 0.429 | 25 |  |  |
| target | Downloads, alpha 0.05 | 46 | 0.977 | 0 | 4 | 1.821 | 0.433 | 25 |  |  |
| target | Downloads, alpha 0.20 | 46 | 0.979 | 0 | 4 | 1.814 | 0.417 | 27 |  |  |
| target | Downloads, alpha 0.50 | 46 | 0.886 | 1 | 10 | 1.82 | 0.367 | 29 |  |  |
| target | Log downloads | 46 | 0.596 | 5 | 24 | 1.829 | 0.162 | 34 |  |  |
| target | Likes, alpha 0.10 | 46 | 0.271 | 11.5 | 30 | 1.609 | 0.988 | 8 |  |  |
| target | Uniform | 46 | 0.588 | 6 | 24 | 1.838 | 0.125 | 35 |  |  |
| target | All-time downloads, alpha 0.10 | 46 | 0.814 | 3 | 19 | 1.768 | 0.55 | 22 |  |  |
| geometry | Topological | 46 | 0.927 | 1 | 9 | 1.788 | 0.362 | 27 |  |  |
| geometry | Repository-count | 46 | 0.49 | 6 | 35 | 1.876 | 0.992 | 2 |  |  |
| geometry | Adoption-weighted | 46 | 0.554 | 4 | 23 | 2.2 | 1 | 5 |  |  |
| resolution | Exact matches only | 44 | 0.86 | 2 | 22 | 1.827 | 0.379 | 27 | 0.008 | 0.008 |
| resolution | Descendant expansion | 46 | 0.816 | 2.5 | 30 | 1.832 | 0.221 | 33 | 0 | 0 |
| cutoff | Date stated in the paper | 4 | 1 | 0 | 0 | 1.677 | 0.712 | 2 | 0.017 | 0 |
| cutoff | First public version | 41 | 0.859 | 1 | 15 | 1.841 | 0.292 | 25 | 0.017 | 0.017 |
| cutoff | First version fitting the panel | 41 | 0.915 | 1 | 5 | 1.827 | 0.375 | 25 | 0.017 | 0.017 |
| cutoff | End of publication year | 45 | 0.79 | 3 | 15 | 1.812 | 0.5 | 22 | 0.017 | 0.017 |
| cutoff | Best available date | 46 | 0.905 | 1 | 8 | 1.816 | 0.438 | 26 | 0.021 | 0.017 |
| cutoff | No cutoff | 46 | 0.766 | 3 | 22 | 1.821 | 0.533 | 20 | 0.008 | 0.008 |
| population | Assistant models | 46 | 0.511 | 6.5 | 30 | 1.527 | 0.979 | 8 | 0.021 | 0.033 |
| population | Reasoning models | 46 | 0.336 | 7.5 | 30 | 1.671 | 0.213 | 29 | 0 | 0 |
| population | Code models | 46 | 0.484 | 5.5 | 39 | 1.798 | 0.663 | 18 | 0 | 0 |
| population | Base models | 46 | 0.793 | 3 | 17 | 1.917 | 0.138 | 37 | 0 | 0 |
| population | Models up to 13B | 46 | 0.921 | 1 | 8 | 1.823 | 0.367 | 27 | 0 | 0.004 |
| population | Models of 34B or more | 46 | 0.092 | 14.5 | 37 | 1.675 | 0.5 | 23 | 0.013 | 0.079 |
| population | Claim-conditioned | 46 | 0.407 | 9 | 33 | 1.69 | 0.696 | 18 | 0.013 | 0 |

## How to read this

- A change of target or population changes the estimand, so a lower tau there says the question changed, not that the
  method is unstable. Geometry, resolution, and cutoff are closer to measurement choices.
- Alternative cutoffs can leave some of a panel's models outside the eligible pool; their mass is then transported to
  the pool, which raises the gap.
- Descendant expansion scores a panel of every compatible variant, so its references are larger panels too.
