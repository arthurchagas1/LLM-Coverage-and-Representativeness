# Selection protocol for the 50 audited papers, release 20261105

Each statement below is marked **documented** (recorded in the project), **reconstructed** (built now, after the sample
existed), or **to confirm** (only the authors can say).

## Search strategy

- **To confirm:** the databases, queries, and dates of the original search. No artifact records them, nor how many papers were screened.
- **Documented:** annotation files date from June 2026 at the latest (`data/paper_annotations/`), and the Hub snapshot is from 2026-06-03.
- **Reconstructed:** a retrospective frame, using OpenAlex, with this query and range:

```
("large language model" OR "large language models" OR LLM OR LLMs) AND (benchmark OR benchmarking OR evaluation OR evaluating OR leaderboard)
publication date 2024-01-01 to 2026-06-03
```

The frame tests how the sample sits within the visible literature. It should be described as a retrospective check, never as the search that produced the sample.

## Inclusion and exclusion criteria

**Documented** in Technical Supplement C.3 and the annotation schema:

- Included: the study's central contribution includes an empirical evaluation of identifiable language-model systems, with enough information to reconstruct its main panel (`is_main_text_llm_eval` is true for all 50).
- Duplicates across publication sources removed.
- arXiv stratum restricted to recent preprints available at corpus construction.
- `include_final_sample` values: {'yes_main': 44, 'yes_secondary': 5, 'no': 1}.

**To confirm:** whether these criteria were set before selecting papers or described afterward, and why a paper marked `no` is still in the annotation file.

## Dates

- Publication years: {2024.0: 17, 2025.0: 24, 2026.0: 8}.
- **To confirm:** the corpus cutoff date.

## Composition

Strata as the venue fields show them:

| stratum | papers | median_models | median_open | median_closed | no_open_panel | needs_manual_review |
| --- | --- | --- | --- | --- | --- | --- |
| AAAI, ICLR, NeurIPS, ICML | 19 | 16 | 11 | 6 | 2 | 10 |
| ACL Anthology | 20 | 10 | 6.5 | 2.5 | 1 | 18 |
| Unspecified | 1 | 6 | 5 | 1 | 0 | 0 |
| arXiv or preprint | 10 | 11 | 6.5 | 3 | 1 | 3 |

Venues: {'ICLR': 14, 'arXiv or preprint': 10, 'ACL Findings (all)': 9, 'ACL': 6, 'EMNLP': 4, 'ICML': 3, 'NEURIPS': 2, 'Unspecified': 1, 'NAACL': 1}.

Themes, classified with keyword rules over benchmark type, claim scope, and title (reconstructed, with no human check):

| theme | AAAI, ICLR, NeurIPS, ICML | ACL Anthology | Unspecified | arXiv or preprint | Total |
| --- | --- | --- | --- | --- | --- |
| Agents and interaction | 4 | 4 | 0 | 2 | 10 |
| Safety and trust | 3 | 4 | 1 | 2 | 10 |
| Mathematics and reasoning | 5 | 0 | 0 | 3 | 8 |
| Language and culture | 1 | 5 | 0 | 1 | 7 |
| General capability | 2 | 1 | 0 | 2 | 5 |
| Domain-specific | 1 | 3 | 0 | 0 | 4 |
| Long context and summarization | 1 | 3 | 0 | 0 | 4 |
| Code | 2 | 0 | 0 | 0 | 2 |

Per-paper detail is in `corpus_table.csv`, with counts by venue and year in `composition_by_venue_year.csv`.

## Flow

See `sample_flowchart.png`. Solid boxes are documented, and dashed boxes are unrecorded or reconstructed.

Papers without an open-weight panel to score (4): arxiv2025_evalita_llm, emnlp_2024_urs, neurips2024_019, openreview_007.

## Sample against the retrospective frame

OpenAlex was unavailable in this run, so the frame was not built. Rerun this step to build it.

## Possible biases

- **Not a probability sample.** Fixed quotas by source mean the audit describes these papers, and generalizing to "the literature" needs care. Percentiles and medians across papers are descriptive.
- **Venue mix.** The second stratum is described as AAAI, ICLR, NeurIPS, and ICML, but holds 0 AAAI papers. ICLR supplies 14 of its papers. The Technical Supplement's description should change, or the stratum be rebalanced.
- **Text availability.** 24 records have source type "not recorded". Other values: {'not recorded': 24, 'user_pasted_paper_text': 5, 'user_uploaded_paper_text': 10, 'user_provided_paper_text': 11}. Selection may favor papers whose text was at hand.
- **Open-weight requirement.** Scoring needs named open-weight models, so closed-only evaluations are out of scope by design, which can make open-weight panels look typical.
- **Recent-work tilt.** Most papers are from 2024 onward, with models evolving quickly. See P1.3 for how the cutoffs affect this.
- **Citation level.** See the frame comparison above; a selection effect would show as higher citations than the frame, within strata.

## Annotation process

- **Documented:** 31 of 50 records have `needs_manual_review = True`, and normalization confidence is {'medium': 33, 'high': 15, 'low': 2}. Nothing records whether those reviews took place.
- **To confirm:** who or what produced the annotations. 26 records hold selection notes written in the first person ("I used Table 3 as the authoritative main model list"), and source types such as `user_pasted_paper_text` suggest text pasted into an assistant. If an LLM produced or assisted them, the paper should say so and report what verification followed.
- No inter-annotator check of panel extraction exists, for the mentions or the lists.
- 24 of 50 records have a source URL; 2 were located in OpenAlex by title. `corpus_table.csv` carries these links, so the missing URLs can be filled in.

## What the authors need to supply

1. Original search source, queries, and dates, or a statement that the sample was purposive and hand-assembled.
2. Whether the criteria came before or after selection, and the corpus cutoff date.
3. Who annotated, with which tool, and which of the 31 flagged records were reviewed.
4. How to describe the second stratum, given no AAAI papers appear in it.

## Draft text for the paper, once the items above are confirmed

> We audit a purposive, source-stratified sample of 50 LLM evaluation papers published between 2024.0 and 2026.0: [strata as confirmed]. Papers qualified when their central contribution included an empirical evaluation of identifiable language-model systems with enough detail to reconstruct the evaluated panel. Because selection was not probabilistic, cross-paper statistics describe this sample and do not estimate prevalence across the literature. [Retrospective frame check to be added once it is built.]
