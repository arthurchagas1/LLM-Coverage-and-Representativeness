# P1.3 Temporal reconstruction, release 20261105

## Summary

Cutoffs available: first public date for 41 of 46 papers, and a date stated in the paper for 4.
The median paper appeared 20 days after the newest model in its panel. In 15 papers the first public date precedes that
model's recorded release, usually because a later version added models, so first public date is not an upper bound on experiment
time. `public_bound` uses the latest version for those papers. Both dates are in `cutoffs_by_paper.csv`.

## Retrospective analysis

Adoption comes from the snapshot, after the paper, restricted to models available at the cutoff. It answers how well the
panel represents, among what was available, the models adopted now. It does not measure adoption at the time of the paper.

| label | papers | median_eligible | median_retro_adoption_gap | tau_retro_adoption_gap | median_retro_greedy_reduction_pct | median_rank_shift | max_rank_shift | papers_with_models_after_cutoff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Newest model in panel | 46 | 4233.5 | 1.82 | 1 | 48.77 | 0 | 0 | 0 |
| Date stated in the paper | 4 | 4340.5 | 1.678 | 1 | 53.03 | 0 | 0 | 1 |
| First public date | 41 | 3748 | 1.841 | 0.846 | 47.779 | 1 | 19 | 15 |
| First version that fits the panel | 41 | 5301 | 1.829 | 0.905 | 46.636 | 1 | 9 | 3 |
| End of publication year | 45 | 8358 | 1.813 | 0.788 | 41.749 | 3 | 16 | 0 |
| Best available date | 46 | 5218 | 1.816 | 0.898 | 46.778 | 2 | 10 | 4 |
| No cutoff | 46 | 12398 | 1.83 | 0.768 | 38.945 | 3 | 22 | 0 |

## Historical analysis

Uses only what existed at the cutoff: availability and structure. A uniform target over available models replaces adoption,
so these coordinates carry no information from after the paper.

| label | papers | median_hist_uniform_gap | tau_hist_uniform_gap | median_hist_breadth | tau_hist_breadth | median_hist_structural | tau_hist_structural | median_hist_greedy_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Newest model in panel | 46 | 1.841 | 1 | 0.006 | 1 | 0.73 | 1 | 12.067 |
| Date stated in the paper | 4 | 1.74 | 0.667 | 0.006 | 1 | 0.587 | 1 | 13.265 |
| First public date | 41 | 1.847 | 0.824 | 0.006 | 0.9 | 0.722 | 0.968 | 11.336 |
| First version that fits the panel | 41 | 1.832 | 0.871 | 0.006 | 0.927 | 0.722 | 0.971 | 11 |
| End of publication year | 45 | 1.829 | 0.745 | 0.006 | 0.842 | 0.695 | 0.925 | 8.32 |
| Best available date | 46 | 1.834 | 0.876 | 0.006 | 0.929 | 0.725 | 0.971 | 11.241 |
| No cutoff | 46 | 1.851 | 0.747 | 0.005 | 0.843 | 0.702 | 0.943 | 7.359 |

`tau` compares paper rankings with the newest-model cutoff. The stated-date row covers only the papers with one, so its tau rests on few papers.

## Do older papers look worse?

Spearman correlation between paper age and each coordinate. A positive value on the adoption gap means older panels look less
representative, which would partly reflect adoption moving to newer models after they were published.

| scheme | metric | spearman_with_paper_age | p_value |
| --- | --- | --- | --- |
| panel_newest | retro_adoption_gap | 0.513 | 2.7e-04 |
| panel_newest | hist_uniform_gap | 0.215 | 0.151 |
| panel_newest | hist_breadth | 0.148 | 0.327 |
| no_cutoff | retro_adoption_gap | 0.494 | 4.8e-04 |
| no_cutoff | hist_uniform_gap | 0.231 | 0.123 |
| no_cutoff | hist_breadth | 0.011 | 0.941 |

## Adoption drift between snapshots

| measure | value |
| --- | --- |
| repositories_in_both | 349123 |
| repositories_new_since_june | 64952 |
| new_repositories_download_share | 0.09 |
| repo_spearman_june_vs_now | 0.473 |
| variant_spearman_june_vs_now | 0.801 |
| top_100_variant_overlap | 66 |
| total_variation_june_vs_now | 0.324 |
| paper_rank_kendall_june_vs_now_adoption | 0.872 |

## How to read this

- The newest-model cutoff is mechanically tied to the panel: a paper that evaluated a recent model gets a larger comparison pool.
  That is the reason to compare it with cutoffs set independently of the panel.
- `public_bound` is an upper bound on when experiments ran, so its pool may include models the authors could not have used. arXiv reports
  only the first and latest versions, so for updated papers the bound can be loose.
- Repository creation dates approximate model release, so reuploads and renames can shift them.
- Stated dates come from `data/paper_experiment_dates_verified.csv`, where each row quotes the passage it relies on.
