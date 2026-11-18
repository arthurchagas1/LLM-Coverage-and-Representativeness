# Which Models Count? LLM Coverage for Auditing Model-Panel Representativeness

Code, data, and results for the paper. The paper audits how well the open-weight models evaluated in LLM papers
represent the open-weight ecosystem, using a lineage taxonomy of every textual generative model on the Hugging Face
Hub. It answers four research questions:

- **RQ1** How well do existing evaluation panels cover the ecosystem?
- **RQ2** How sensitive is that audit to modelling choices (target, geometry, resolution, cutoff, population)?
- **RQ3** How much can coverage-aware construction improve a panel under realistic constraints?
- **RQ4** Is lineage coverage behaviorally meaningful: do relatives behave alike, and do panels spread over the tree
  recover population means better?

All results come from release **`20261105`**, built from a Hub snapshot of 3,213,895 repositories collected on
2026-11-05. One command rebuilds every table and figure from the data in this repository.

## Quick start

Python 3.12 on Linux or macOS, about 16 GB of RAM (the scalability benchmark peaks at 12 GB), and about 3 GB of disk
(the clone is about 400 MB).

```bash
git clone https://github.com/arthurchagas1/LLM-Coverage-and-Representativeness.git
cd LLM-Coverage-and-Representativeness
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt          # exact environment; requirements.txt lists the direct dependencies

python run_all_experiments.py                 # every step, offline, about 1.5 h on one core
python scripts/compare_results.py             # regenerated results vs. the committed ones
```

`run_all_experiments.py` runs the steps below in order, logs each one to `logs/<release>/<step>.log`, stops at the
first failure, and can resume (`--resume`), restart from a step (`--from rq1`), or run some steps only
(`--only rq4 annotation`). `--list` shows the steps.

`compare_results.py` compares every regenerated CSV, JSON, and report with the committed version (numbers to a
relative tolerance of 1e-6). Everything is seeded, so all results reproduce exactly except the wall-clock timings of
the scalability benchmark (P1.6), which depend on the machine and are reported separately.

## Pipeline

| Step | Script | Output | Time |
| --- | --- | --- | --- |
| `clean` (only with `--with-clean`) | `data_collection/clean.py` | `data/oome_clean_20261105/` from the raw snapshot | 65 min |
| `lineage` | `data_collection/p4_families.py` | `data/oome_lineages_20261105/`: family, series, base model, size, role, and variant of every repository | 20 min |
| `validation` | `src/build_foundation.py` (runs `coverage_metric_validation_source.py`) | name resolution, targets, tree geometry, reference simulations; `outputs/20261105/tables/`, `cache/foundation.pkl` | 13 min |
| `foundation_check` | `src/validate_foundation.py` | the cached geometry reproduces the source metrics | < 1 min |
| `figures` | `make_figures.py` | `outputs/20261105/figures/` | < 1 min |
| `configs`, `paper_dates` (only with `--network`) | `data_collection/fetch_*.py` | refresh `data/repo_config_cache.csv`, `data/paper_dates.csv` | 30 min |
| `diag_*` | `src/build_*.py` | per-paper, resolver, counterfactual, and temporal diagnostics in `outputs/20261105/` | 6 min |
| `feasibility` | `analysis/f_feasibility.py` | `F_feasibility`: cutoffs, eligible sets, constraint ladder | < 1 min |
| `appendix_c` | `build_appendix_c.py` | `outputs/20261105/appendix_c/`: reference panels, eligibility and claim-conditioned populations | 3 min |
| `p1_1` to `p1_6` | `analysis/p1_*.py` | `P1_*` folders: reuse validation, target sensitivity, temporal reconstruction, bound tightness, corpus, scalability | 40 min |
| `rq1` to `rq4` | `analysis/rq*.py` | `RQ1_audit`, `RQ2_sensitivity`, `RQ3_construction`, `RQ4_behavioral` | 9 min |
| `paper_figures` | `notebooks/paper_figures.ipynb` | `figures/20261105/` | < 1 min |
| `annotation` | `annotation/agreement.py` | `annotation/results/` | < 1 min |

Folders named in the Output column are under `deliverables/20261105/` unless a path is given. Each has a
`REPORT.md` that states what was computed and the main numbers.

## Where each result of the paper comes from

| Paper | Files |
| --- | --- |
| Population and filtering (Sec. 3.1, Supplement A) | `data/oome_clean_20261105/oome_cleaning_summary.txt`; `data/oome_lineages_20261105/oome_lineage_summary.txt` after the `lineage` step |
| Name resolution and sourced corrections (Supplement B) | `outputs/20261105/tables/corrected_resolution_audit.csv`, `paper_resolution_summary.csv`; `data/feasibility_model_corrections.csv` |
| Geometry checks (Supplement C) | `outputs/20261105/tables/axiomatic_and_ambiguity_tests.csv`, `logs/20261105/foundation_check.log` |
| Corpus and cutoffs (Sec. 4, Supplement D) | `deliverables/20261105/P1_5_corpus/`, `P1_3_temporal/`, `F_feasibility/feasibility_by_paper.csv` |
| RQ1 (Sec. 5.1, Supplement E) | `deliverables/20261105/RQ1_audit/`, `outputs/20261105/appendix_c/` |
| RQ2 (Sec. 5.2, Supplement F) | `deliverables/20261105/RQ2_sensitivity/`, `P1_2_target_sensitivity/`, `P1_3_temporal/` |
| RQ3 (Sec. 5.3, Supplement E) | `deliverables/20261105/RQ3_construction/`, `F_feasibility/` |
| RQ4 (Sec. 5.4, Supplement H) | `deliverables/20261105/RQ4_behavioral/` (`latex_tables.tex` holds the tables) |
| Scalability (Supplement G) | `deliverables/20261105/P1_6_scalability/` |
| Figures | `figures/20261105/rq1_family_flow.pdf`, `rq2_rank_bump.pdf`, `rq3_swaps.pdf`, `rq4_behavior.pdf` |
| Human validation of the taxonomy | `annotation/results/REPORT.md` |

`P1_1_validation` and `P1_4_bound` hold an earlier behavioral analysis (association of the adoption gap with recovery
error, and tightness of the mean-recovery bound). RQ4 replaces it in the paper; the folders stay for completeness.

## Repository layout

```
run_all_experiments.py   single entry point; release_config.py says which folders belong to each release
data_collection/         Hub collection (coleta.py), cleaning, lineage taxonomy, network caches
coverage_metric_validation_source.py, model_resolver.py
                         population, name resolution, targets, geometry, and reference simulations
src/                     foundation cache and diagnostics
analysis/                F_P, P1.1-P1.6, RQ1-RQ4 (shared code in common.py and rq_common.py)
build_appendix_c.py, make_figures.py
notebooks/               paper_figures.ipynb (figures of the paper)
annotation/              human validation of the taxonomy: sample, annotation site, guidelines, labels, agreement
scripts/                 compare_results.py, run_notebook.py
data/                    raw snapshot, population, paper annotations, sourced corrections, caches (data/README.md)
external_data/           frozen Open LLM Leaderboard and SALAD-Bench results (held-out behavior)
deliverables/20261105/   results as reported, one folder per analysis
outputs/20261105/        intermediate tables of the validation and appendix steps
figures/20261105/        figures of the paper
```

## Data

`data/README.md` describes every file. Git holds all the data, from the raw Hub snapshot to the results:

- the raw snapshot (3,213,895 repositories) with the 35 columns the cleaning step reads, in parts below 50 MB
  (359 MB instead of 821 MB);
- the cleaned population (425,223 repositories) with the 58 columns later steps read (18.6 MB instead of 998 MB),
  which is where the default run starts;
- the audited papers, sourced corrections, frozen network caches, leaderboards, and human annotations.

To rerun the cleaning filter itself as well:

```bash
python run_all_experiments.py --with-clean       # cleaning (about 65 min), then every later step
```

### A new snapshot

A release binds one Hub snapshot to everything derived from it. To build a new one:

```bash
HF_SNAPSHOT_DIR=data/hf_snapshots_YYYYMMDD python data_collection/coleta.py   # about 40 min, network
# add the release to release_config.py, then
python run_all_experiments.py --release YYYYMMDD --with-clean --network
```

## Human validation of the lineage taxonomy

`annotation/` holds a stratified sample of 240 repositories (by family size, plus the repositories of the audited
panels), the annotation guidelines, a small annotation website (`annotation/server.py`, Python standard library
only), the raw labels of each annotator (`annotation/annotations/*.jsonl`, never edited), a logged layer of
corrections of clear errors, and the agreement and accuracy analysis. `annotation/results/REPORT.md` gives the
current state: Krippendorff's alpha and Gwet's AC1 between annotators on the raw labels, and the taxonomy's accuracy
against the reference labels, design-weighted over strata. See `annotation/README.md`.

### Release with the role fix

The annotation found one systematic taxonomy error: the Hub tag `custom_code` was read as a code-model signal and
labeled 3.1% of repositories (2.6% of downloads) with role `code`. Release `20261105`, which the paper reports, keeps
that behavior so that its results reproduce. Release `20261105_rolefix` applies the fix to the same snapshot:

```bash
python run_all_experiments.py --release 20261105_rolefix
```

Its results go to `deliverables/20261105_rolefix/` and can be compared with the paper's.

## Determinism

Every random draw has a fixed seed (validation 20260612, RQ4 20261116, annotation sample 20261117) and
`run_all_experiments.py` sets `PYTHONHASHSEED=0`. By default the run is offline: the steps that call the network are
skipped and their frozen caches in `data/` are read. `--network` refreshes them (Hub configs, arXiv dates, OpenAlex);
results may then change, since those services change.

## Known limitations

- **Adoption is a snapshot.** Download counts are those of the snapshot, so adoption at a paper's date cannot be
  recovered. P1.3 separates historical from retrospective analysis and measures drift against an earlier snapshot.
- **Family can follow architecture.** With no name or declared base model to go on, the configuration's model type
  decides, so some independent pretraining efforts land in the family whose architecture they reuse. The human
  validation measures how often.
- **Singleton families** weigh as much as large ones in the breadth target (uniform branching).
- **Context length is partial.** It comes from `config.json` of one representative repository per variant, so gated
  and removed repositories stay unknown.
- **The breadth-greedy construction stores a dense similarity matrix**, which limits it to about 32,000 variants (P1.6).
- **Panel extraction.** The evaluated models of the 50 papers were extracted by hand into
  `data/paper_annotations/acl15_new_schema.json`; P1.5 documents the sample and its possible biases.

## Notes

Some code comments in `data_collection/` and `coverage_metric_validation_source.py` are in Portuguese.

## License

Code: MIT (see `LICENSE`). Third-party data in `external_data/` and the Hub metadata keep their original terms.
