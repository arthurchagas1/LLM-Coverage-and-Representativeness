# Data

Everything the pipeline reads, for release `20261105` (Hugging Face Hub snapshot of 2026-11-05), from the raw
snapshot of the whole Hub to the small hand-made inputs. All of it is in git, reduced where needed to fit.

## What is in git

| Path | Rows | What it is | Produced by |
| --- | --- | --- | --- |
| `oome_clean_20261105/oome_core_text_llm.parquet` | 425,223 | The textual population: every Hub repository that passes the cleaning filter, before the lineage step (which sets aside 11,148 multimodal components, leaving 414,075). | `data_collection/clean.py` on the raw snapshot |
| `oome_clean_20261105/oome_cleaning_manifest.json`, `oome_cleaning_summary.txt` | | Filter parameters, counts by subpopulation, and rejection reasons for all 3,213,895 repositories | same |
| `hf_snapshots_20261105/*.flat.part-NN.parquet` | 3,213,895 | The raw snapshot: every publicly indexable model repository on the Hub on 2026-11-05 (02:08 to 02:47 UTC), with the 35 columns the cleaning step reads, in 11 parts below 50 MB. Input of `--with-clean`. | `data_collection/coleta.py`, reduced by `repo_build/reduce_snapshot.py` |
| `hf_snapshots_20261105/*.manifest.json` | | Collection parameters (API fields, sort order, start and end times, zero errors) and summary counts | `data_collection/coleta.py` |
| `hf_snapshots_20261105/SHA256SUMS`, `ORIGINAL_SHA256` | | Checksums of the parts and the manifest, and of the full 41-column snapshot they come from | |
| `june2026_core_downloads.parquet` | 384,028 | `repo_id` and 30-day downloads of the textual population of an earlier snapshot (2026-06-03). Used only to measure adoption drift between snapshots (P1.3). | earlier collection |
| `paper_annotations/acl15_new_schema.json` | 50 | The audited papers: venue, year, claim scope, and the evaluated models, with open-weight, closed, auxiliary, and excluded lists and the reasons for exclusion. | manual extraction |
| `feasibility_model_corrections.csv` | 37 | Facts that a panel model's Hub record lacks (size, role, release date, gating, license, context), each with its source and URL. | manual, sourced |
| `paper_dates.csv` | 50 | arXiv match of each paper, with first and latest version dates. | `data_collection/fetch_paper_dates.py` (network) |
| `paper_experiment_dates_verified.csv` | 4 | Experiment dates that a paper states, each with the quote. | manual reading of the full texts |
| `repo_config_cache.csv` | 15,079 | Context length read from `config.json` of one representative repository per variant and of every panel repository; gated and removed repositories stay unknown. | `data_collection/fetch_repo_configs.py` (network) |
| `corpus_frame_cache.json` | | OpenAlex responses for the retrospective search frame of P1.5. | `analysis/p1_5_corpus.py` (network) |
| `../external_data/open_llm_leaderboard.parquet` | 4,576 | Frozen Open LLM Leaderboard v2 results (six benchmarks), used only as held-out behavior. | Hugging Face `open-llm-leaderboard/contents` |
| `../external_data/salad_bench_leaderboard.xlsx` | | Frozen SALAD-Bench safety leaderboard, used only as held-out behavior. | SALAD-Bench |
| `../annotation/` | 240 items | The human validation of the taxonomy: sample, guidelines, raw labels, corrections, and results. See `annotation/README.md`. | annotators |

The network caches are frozen: by default `run_all_experiments.py` runs offline and reads them as they are, so reruns
give the same results. `--network` refreshes them, after which results may change.

### How the files were reduced

**Raw snapshot.** The collector's file has 41 columns and 821 MB. The parts in git keep all 3,213,895 rows, in the
original order, and the 35 columns that `data_collection/clean.py` reads. Six columns that no step reads were dropped:
`last_modified`, `inference`, `spaces`, `num_spaces`, `safetensors_json`, and `gguf_json`. The parts are compressed
with zstd, 359 MB in total. Rerunning the cleaning step on them reproduces the cleaned population below.

**Cleaned population.** `oome_core_text_llm.parquet` keeps all 425,223 rows and the 58 columns that later steps read.
Ten columns were dropped because no step after cleaning reads them: the raw JSON of the model card, config,
safetensors and GGUF metadata (`card_data_json`, `config_json`, `safetensors_json`, `gguf_json`), the file list
(`siblings`), the commit hash (`sha`), `last_modified`, `spaces`, `num_spaces`, and `inference`. This takes it from
998 MB to 18.6 MB. Running the lineage step on the reduced file reproduces the published lineage exactly.

### Columns of the population file

| Group | Columns |
| --- | --- |
| Identity | `repo_id`, `author`, `created_at` |
| Adoption | `downloads_30d`, `downloads_all_time`, `likes`, `trending_score`, `children_model_count`, and their `*_clean` versions; `adoption_proxy_log` |
| Access | `private`, `disabled`, `gated`, `license` |
| Hub metadata | `pipeline_tag`, `library_name`, `tags` (pipe-separated), `languages`, `datasets`, `architecture`, `model_type`, `base_models`, `base_model_relation`, `parameter_count`, `parameter_count_clean`, `context_length` |
| Files | `num_siblings`, `has_config_json`, `has_tokenizer`, `has_safetensors`, `has_pytorch_bin`, `has_gguf`, `has_weight_files` |
| Cleaning decision | `keep_broad`, `is_core_text_llm`, `is_multimodal_language`, `is_noncore_language_like`, `cleaning_reasons`, `cleaning_warnings`, `llm_relevance_score`, `llm_relevance_evidence`, `core_relevance_score`, `core_relevance_evidence`, `duplicate_repo_id`, `duplicate_sha`, `is_quantized`, `is_adapter`, `is_checkpoint`, `is_merge`, `is_embedding_like`, `is_classifier_like`, `is_encoder_only_like`, `is_non_language_or_modal` |
| Collector guesses | `model_family_guess`, `artifact_type_guess` |

## Rerunning the cleaning step

```bash
python run_all_experiments.py --with-clean      # cleaning (about 65 min), then every later step
```

The cleaning step then rewrites `oome_clean_20261105/` with all the columns it produces (about 90 MB, so do not commit
it; `git restore data/oome_clean_20261105` brings back the reduced file) and the four auxiliary subpopulations (broad,
multimodal, non-core, rejected), which `.gitignore` keeps out of git.

## Terms

The Hub metadata are public listings of the Hugging Face Hub API. The leaderboard files are frozen copies of public
leaderboards, redistributed for reproducibility under their original terms. The full texts of the audited papers,
which `fetch_paper_dates.py` downloads from arXiv to find stated experiment dates, are not redistributed.
