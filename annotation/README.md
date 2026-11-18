# Human validation of the lineage taxonomy

Release 20261105. Run every command from the repository root (the site itself uses only the Python standard library).

## 1. Sample (already drawn; frozen once annotation starts)

```bash
python annotation/build_sample.py
```

240 repositories, seed 20261117, in five strata:

| Stratum | Rule | Sampled | Repositories in population |
| --- | --- | --- | --- |
| large | family with >= 1,000 repositories (23 families) | 60 | 396,118 |
| medium | family with 50-999 repositories | 50 | 12,705 |
| small | family with 2-49 repositories | 50 | 4,420 |
| singleton | family with one repository | 40 | 792 |
| panel | most-downloaded repository of each variant in the 46 audited panels | 40 | 355 variants |

The items are split into three nested, stratified batches of 60, 60, and 120, each with the same stratum mix as the
full sample, so agreement can already be computed after 60 and 120 labels.

`data/items.json` holds only what annotators see (repository, link, raw Hub metadata). The taxonomy's own labels
are in `data/system_labels.csv`, which annotators must not open. `data/sample_manifest.json` records the design.

## 2. Annotation site

```bash
python annotation/server.py            # http://127.0.0.1:18765
```

From a laptop, tunnel to the server and open http://localhost:18765:

```bash
ssh -L 18765:localhost:18765 <user>@<server>
```

Each annotator types a name (always the same one, to resume) and labels scope, family, series, parameter size,
functional role, artifact type, and confidence, with an evidence URL and notes. Everyone labels batch 1 first, then batch 2, then
batch 3 (the order is shuffled only within a batch); the header shows progress per batch. Read `GUIDELINES.md` (also under the
"Guidelines" button) before starting. Items appear in a different random order for each annotator; saves are
appended to `annotations/<name>.jsonl` and never overwritten. Ctrl+Enter saves and moves on.

At least two annotators must label all 240 items independently. A median of 2-3 minutes per item means about
8-12 hours per annotator; the site records the time spent per item.

## 3. Agreement, adjudication, accuracy

```bash
python annotation/agreement.py
```

By default it uses the largest checkpoint (60, 120, or 240 items) that at least two annotators have finished, and
it reports agreement at every checkpoint already reached (`results/agreement_by_checkpoint.csv`). To force one:

```bash
python annotation/agreement.py --checkpoint 60
```

With 60 items the alpha intervals are about ±0.08 wide, so 60 is a first check and 240 the final estimate.

Writes `results/REPORT.md` and CSVs:

- **Agreement** per field: percent agreement, Cohen's kappa per pair, Fleiss' kappa, Krippendorff's alpha with 95%
  bootstrap intervals (size also with an interval alpha on the bucket rank). Pre-specified reading: alpha >= 0.80
  validated, 0.667-0.80 tentative, below 0.667 not reliable. The alpha implementation reproduces Krippendorff's
  published example (0.743 nominal, 0.849 interval).
- **Adjudication**: items with disagreements go to `results/adjudication.csv`. A third person fills the
  `adjudicated_*` columns using the guidelines, then the script is rerun. A filled file is never overwritten.
- **Taxonomy accuracy** against the reference labels (unanimous or adjudicated): scope, family, series, size,
  role, and the full variant, by stratum with Wilson intervals and design-weighted over the four family-size strata;
  errors listed in `results/taxonomy_errors.csv`.

**Corrections of clear errors** (`corrections.py`, `corrections_manual.csv`): rules on the metadata the site shows
(size from `parameter_count`, role from an explicit name suffix, vision-language tags out of scope) plus a short
item list, each row citing the guideline it applies. The raw files are never edited; the report gives agreement
on the raw labels (the reliability to publish) and after corrections, and every changed label is in
`results/corrections_log.csv`. After the size rule the size agreement is 1.0 by construction wherever the Hub reports
`parameter_count`; quote the raw size alpha as the human reliability.

**Taxonomy bug found by the annotation**: the Hub tag `custom_code` triggered the `code` role in
`data_collection/p4_families.py` (fixed 2026-11-17). `data/system_labels.csv` keeps the release role in
`model_role` and the role after the fix in `model_role_fixed`; the report shows accuracy for both. Release `20261105`,
which the paper reports, keeps the old roles; release `20261105_rolefix` applies the fix
(`python run_all_experiments.py --release 20261105_rolefix`).

The next step, once the accuracy is known, is to propagate the measured error rates into the coverage estimates
(perturb the taxonomy at the observed rates by stratum and recompute RQ1-RQ3).
