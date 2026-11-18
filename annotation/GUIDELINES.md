# Annotation guidelines: lineage taxonomy

## Purpose and independence

We measure how accurately the automatic taxonomy assigns each Hugging Face repository to a family, series,
parameter size, and functional role. You label each repository from public evidence; the site never shows the
system's answer, and you must not look at it (`annotation/data/system_labels.csv`), at the code, or at another
annotator's labels. Do not discuss individual items with other annotators until everyone has finished.

Record the best-supported answer even when it is not in the suggestion list, and record doubt in Confidence and Notes
instead of guessing.

Items come in three batches (60, 60, 120). Finish each batch before moving on: agreement is first computed when
two annotators have completed batch 1.

## Procedure for each item

1. Open the repository (link at the top). Read the model card, `config.json`, the declared base model, and, when
   needed, the upstream official release page.
2. **Scope.**
   - *In scope*: a text-generative language model (causal or seq2seq) with weights.
   - *Out of scope*: embedding model, classifier, encoder-only model, vision/speech-only model, dataset, or a repo
     with no model weights.
   - *Insufficient information*: could be in scope, but public evidence does not identify its lineage.
   Hierarchy fields are only filled for in-scope items.
3. **Family**: the lineage that produced the pretrained weights (`llama`, `qwen`, `mistral`, `gemma`, `gpt2`, `t5`,
   ...). A fine-tune inherits its base model's family. An architecture name alone is not enough when the card
   names a different lineage. Use an existing slug from the suggestions whenever it fits; otherwise type a new
   lowercase slug.
4. **Series**: the versioned generation inside the family (`llama3.1`, `qwen2.5`, `gemma2`, `deepseek-r1`, ...).
   Do not merge different generations. A fine-tune inherits the series of its base model. The series never
   contains a parameter size, a size tier (`-l`, `-small`, `tiny`), a training hyperparameter, or a checkpoint
   step: `gsa-2.7B` is series `gsa`, and `MobileMoE-L` is `mobilemoe`.
   When a project has no versioned generations, the series equals the family.
5. **Parameter size**: total parameters of the checkpoint (for mixture-of-experts, total, with active parameters in
   Notes). Pick the bucket; fill the exact size if known (`7b`, `8x7b`, `125m`). Use `unknown` only when no source
   states or implies the size. **Each bucket includes its lower bound and excludes its upper bound**: 3B goes in
   `3B-7B`, 7B in `7B-13B`, 13B in `13B-34B`, 70B in `70B-180B`. Do not confuse training tokens (`100B` tokens)
   with parameters.
6. **Functional role**, most specific supported: `base`, `instruct`, `chat`, `reasoning`, `code`, `math`, `domain`
   (domain-adapted, e.g. medical, legal, a specific language), `seq2seq`, or `unknown-role` when the lineage is
   known but the tuning is not. Do not infer `instruct` from a chat template alone; use `reasoning` only for
   checkpoints released or trained for reasoning. Any instruction- or chat-tuned checkpoint (`-Instruct`, `-it`,
   SFT/RLHF assistants) is `instruct`; use `chat` only when the release itself calls the checkpoint "chat"
   (`Llama-2-7b-chat`, `XVERSE-13B-Chat`). An official release without a tuning suffix whose card describes a
   pretrained model is `base`. The Hub tag `custom_code` only means the repo ships its own modeling code; it is
   not evidence of a code model.
7. **Artifact type**: `official` (released by the original lab), `fine-tune`, `merge`, `mirror-reupload`
   (same weights re-uploaded), `independent-pretraining` (new pretraining that reuses an architecture), `other`.
   A lab's own instruct, chat, or reasoning release (`meta-llama/Meta-Llama-3-70B-Instruct`, `google/gemma-2-27b-it`)
   is `official`, not `fine-tune`; `fine-tune` is for checkpoints trained by someone other than the original lab.
8. **Confidence** (`high`, `medium`, `low`), the strongest **evidence URL**, and **notes** for anything unusual.

## Special cases

- **Merges**: if one parent clearly dominates, use its lineage and list all parents in Notes; otherwise mark
  *insufficient information*.
- **Distillations**: the student's architecture and weights define family and size; mention the teacher in Notes.
- **Quantizations and adapters** should not appear (they were filtered out). If one does, label the underlying
  model and write "quantization" or "adapter" in Notes.
- **Independent pretraining that reuses an architecture** (e.g. a new model built on `LlamaForCausalLM`): the
  family is the new project, not `llama`. Choose `independent-pretraining` as artifact type.
  This takes priority over `official` even when the new project's own lab released it (signals: no declared
  base model, tags or card saying "from scratch" / "pretrained"). Put the borrowed architecture in Notes.
- **Name versus parameter count**: the size bucket follows the actual total parameters (`parameter_count`, the
  safetensors metadata, or `config.json`), not the marketing name. A "1B" model with 941M parameters is `<1B`,
  exact size `941m`; mention the name in Notes if they differ.

## After annotation

`annotation/agreement.py` computes agreement between annotators and writes the items on which they disagree to
`annotation/results/adjudication.csv`. A third person, or a consensus meeting after all labels are frozen, fills the
`adjudicated_*` columns using these same rules. Original annotation files are never edited.
