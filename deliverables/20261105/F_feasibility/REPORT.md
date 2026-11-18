# Feasible panels, release 20261105

## Summary

3 of 46 papers are problematic under the original rule, meaning F_P excludes at least one of their
own models. Reasons: {'missing_size_bucket': 3}. After repair, 46 of 46 panels sit
inside the rebuilt core F_P. 51 mentions use the sourced corrections file and
269 use their own repository record.

Under the core constraints, the adoption-greedy alternative improves the gap in 46 of
46 papers (median reduction 41.6%), at a median cost
ratio of 0.06 in parameters. Keeping the panel's size-by-role composition, it improves
45 of 46 (median 19.3%),
at a median cost ratio of 1.00.

## Results by stratum and constraint

| stratum | rung | comparable | applied | median_pool | adoption_improved | median_adoption_reduction_pct | breadth_improved | structural_improved | C1_improved | median_cost_ratio | low_adoption_members | relaxed_slots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| problematic | historical | 3 | 3 | 1279 | 3 | 56.9 | 3 | 2 | 3 | 0.02 | 3 | 0 |
| problematic | core | 3 | 3 | 733 | 3 | 53.69 | 3 | 1 | 3 | 0.03 | 4 | 0 |
| problematic | cost | 3 | 3 | 733 | 3 | 23 | 3 | 0 | 1 | 1.16 | 13 | 0 |
| problematic | license | 3 | 1 | 477 | 3 | 21.19 | 3 | 0 | 1 | 1.16 | 12 | 0 |
| problematic | context | 3 | 3 | 279 | 3 | 12.69 | 3 | 0 | 2 | 1.12 | 10 | 0 |
| problematic | adoption_floor | 3 | 3 | 185 | 3 | 14.43 | 3 | 0 | 2 | 1.12 | 7 | 1 |
| valid | historical | 43 | 43 | 4744 | 43 | 47.48 | 43 | 43 | 43 | 0.07 | 30 | 0 |
| valid | core | 43 | 43 | 1339 | 43 | 40.77 | 43 | 32 | 34 | 0.07 | 105 | 0 |
| valid | cost | 43 | 43 | 1339 | 42 | 18.91 | 42 | 16 | 25 | 1 | 181 | 0 |
| valid | license | 43 | 30 | 984 | 42 | 18.41 | 42 | 16 | 26 | 1.01 | 184 | 0 |
| valid | context | 43 | 43 | 324 | 42 | 7.18 | 41 | 15 | 25 | 1 | 134 | 0 |
| valid | adoption_floor | 43 | 43 | 114 | 39 | 5.58 | 40 | 14 | 30 | 1 | 68 | 0 |
| all | historical | 46 | 46 | 4233.5 | 46 | 48.77 | 46 | 45 | 46 | 0.05 | 33 | 0 |
| all | core | 46 | 46 | 1092.5 | 46 | 41.63 | 46 | 33 | 37 | 0.06 | 109 | 0 |
| all | cost | 46 | 46 | 1092.5 | 45 | 19.28 | 45 | 16 | 26 | 1 | 194 | 0 |
| all | license | 46 | 31 | 966.5 | 45 | 18.99 | 45 | 16 | 27 | 1.02 | 196 | 0 |
| all | context | 46 | 46 | 323.5 | 45 | 7.19 | 44 | 15 | 27 | 1 | 144 | 0 |
| all | adoption_floor | 46 | 46 | 115 | 42 | 5.64 | 43 | 14 | 32 | 1 | 75 | 1 |

`applied` counts papers where the constraint was verifiable: license in 31 of 46 and context in
46 of 46. Rungs without enough candidates: none.

## How to read this

- Membership of the original panel in F_P is largely true by construction once its metadata are known, because the
  envelope comes from the panel's own models. It shows the comparison is well posed, not that the panels were feasible
  in practice. `repaired_core_eligible` should therefore be read as a data-quality check.
- `adoption_gap` improvement under `core` is also near mechanical: greedy optimizes that coordinate directly over a pool
  of hundreds of candidates, with no constraint keeping the panel's scale. The `cost` rung and later rungs are the
  informative comparison.
- Cost is a proxy based on parameters. No monetary or measured inference cost exists for these papers.
- License and context limits use the panel's own envelope because the papers do not state requirements.
- Facts for models outside the correction list come from inferred metadata, which no human has validated here.

## Files

- `feasibility_by_paper.csv`: one row per paper, with eligibility, exclusion reasons, and panel facts.
- `feasibility_ladder.csv`: one row per paper and rung, with pool, gains, cost ratio, and status.
- `feasibility_strata.csv`: valid, problematic, and all papers.
- `exclusions_by_mention.csv`: every mention the original rule excluded, with repaired facts and evidence.
- `selected_members.csv`: the alternative panels, with a representative repository for each variant.
