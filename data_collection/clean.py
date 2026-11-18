#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cleaning step: filter the raw Hugging Face Hub snapshot to the population of the paper.

Input (HF_SNAPSHOT_DIR, set by release_config.py): the newest *.flat.parquet in data/hf_snapshots_<release>/.

Outputs (OOME_CLEAN_DIR):
  1. oome_broad_clean.parquet          language-like repositories after removing obvious noise: no weight files,
                                       duplicates, quantizations, adapters, checkpoints, embeddings, and so on
  2. oome_core_text_llm.parquet        the paper's population: open-weight generative text LLMs
  3. oome_multimodal_language.parquet  multimodal models with a language component (image-text-to-text, VQA, ...)
  4. oome_noncore_language_like.parquet language-like but outside the core (BERT-like, fill-mask, classifiers, ...)
  5. oome_rejected.parquet             rejected repositories, with explicit reasons
  6. oome_cleaning_manifest.json       counts and parameters
  7. oome_cleaning_summary.txt         text report for inspection

The repository ships only (2), (6) and (7), with the columns later steps read; see data/README.md.

    python run_all_experiments.py --with-clean --only clean
"""

import os
import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from family_guess import corrected_family_guess

import pandas as pd
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

SNAPSHOT_DIR = Path(os.environ.get("HF_SNAPSHOT_DIR", "data/hf_snapshots_20261105"))
OUT_DIR = Path(os.environ.get("OOME_CLEAN_DIR", "data/oome_clean_20261105"))

# Se None, pega automaticamente o snapshot flat mais recente.
INPUT_FILE = None

# Tamanho de chunk para CSV e para Parquet via PyArrow.
CHUNK_SIZE = 100_000

# Se True, salva CSV além de Parquet. Para 2M+ linhas pode ser grande.
SAVE_CSV = False

# Critérios principais.
REQUIRE_WEIGHT_FILES = True
KEEP_GATED_MODELS = True

DROP_QUANTIZATIONS = True
DROP_ADAPTERS = True
DROP_CHECKPOINTS = True

# Merges podem ser ecologicamente relevantes no open-weight.
# Eu manteria no broad e no core se forem text LLMs.
DROP_MERGES = False

# Score mínimo para entrar no broad.
MIN_BROAD_LLM_RELEVANCE_SCORE = 3

# Score mínimo para entrar no core textual.
MIN_CORE_LLM_RELEVANCE_SCORE = 5

# Se True, modelos sem pipeline_tag ainda podem entrar no core se tiverem
# arquitetura/nome forte de LLM.
ALLOW_MISSING_PIPELINE_IN_CORE_WITH_STRONG_EVIDENCE = True

# Número de exemplos por motivo no relatório.
N_EXAMPLES_PER_REASON = 12

# Colunas para preservar.
PREFERRED_COLUMNS = [
    "repo_id",
    "author",
    "sha",

    "downloads_30d",
    "downloads_all_time",
    "likes",
    "trending_score",
    "children_model_count",

    "created_at",
    "last_modified",

    "private",
    "disabled",
    "gated",
    "inference",

    "pipeline_tag",
    "library_name",
    "tags",

    "license",
    "languages",
    "datasets",
    "architecture",
    "model_type",
    "model_family_guess",
    "base_models",
    "base_model_relation",
    "parameter_count",
    "context_length",
    "artifact_type_guess",

    "num_siblings",
    "has_config_json",
    "has_tokenizer",
    "has_safetensors",
    "has_pytorch_bin",
    "has_gguf",
    "num_spaces",
    "spaces",

    "card_data_json",
    "config_json",
    "safetensors_json",
    "gguf_json",
    "siblings",
]


# =============================================================================
# PIPELINES
# =============================================================================

CORE_TEXT_PIPELINES = {
    "text-generation",
    "text2text-generation",
    "conversational",
    "summarization",
    "translation",
    "question-answering",
    "multiple-choice",
}

# Esses são modelos de linguagem multimodais; não entram no core text,
# mas são úteis como população auxiliar.
MULTIMODAL_LANGUAGE_PIPELINES = {
    "image-text-to-text",
    "video-text-to-text",
    "audio-text-to-text",
    "visual-question-answering",
    "document-question-answering",
    "image-to-text",
}

# Pipelines que não devem entrar no core textual.
# Alguns podem aparecer no broad se houver evidência forte, mas normalmente são noncore.
NONCORE_LANGUAGE_PIPELINES = {
    "fill-mask",
    "text-classification",
    "token-classification",
    "zero-shot-classification",
    "text-ranking",
    "sentence-similarity",
    "feature-extraction",
    "table-question-answering",
}

# Pipelines claramente fora do escopo de LLM textual.
HARD_EXCLUDE_PIPELINES = {
    "text-to-image",
    "image-to-image",
    "image-text-to-image",
    "image-classification",
    "image-segmentation",
    "image-feature-extraction",
    "object-detection",
    "zero-shot-object-detection",
    "mask-generation",
    "depth-estimation",

    "automatic-speech-recognition",
    "text-to-speech",
    "text-to-audio",
    "audio-to-audio",
    "audio-classification",
    "voice-activity-detection",

    "robotics",
    "time-series-forecasting",
    "tabular-classification",
    "tabular-regression",
    "reinforcement-learning",

    "text-to-video",
    "image-to-video",
    "video-classification",
    "unconditional-image-generation",
}


# =============================================================================
# PADRÕES DE MODELOS
# =============================================================================

# Arquiteturas/model types fortes para LLM generativo textual.
CORE_GENERATIVE_MODEL_TYPES = {
    "llama",
    "qwen",
    "qwen2",
    "qwen3",
    "mistral",
    "mixtral",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma4",
    "deepseek",
    "deepseek_v2",
    "deepseek_v3",
    "phi",
    "phi3",
    "phi4",
    "olmo",
    "olmo2",
    "falcon",
    "gpt2",
    "gptj",
    "gpt-j",
    "gpt_neo",
    "gpt-neo",
    "gpt_neox",
    "gpt-neox",
    "bloom",
    "mpt",
    "starcoder",
    "starcoder2",
    "codegen",
    "codellama",
    "stablelm",
    "internlm",
    "internlm2",
    "internlm3",
    "yi",
    "baichuan",
    "glm",
    "chatglm",
    "minicpm",
    "aquila",
    "xglm",
    "rwkv",
    "mamba",
    "mamba2",
    "nemotron",
    "command-r",
    "granite",
    "openelm",
    "exaone",
    "smollm",
    "dbrx",
    "jamba",
    "solar",
}

# Seq2seq generativos: entram no core se pipeline/tags sugerirem geração textual.
SEQ2SEQ_GENERATIVE_MODEL_TYPES = {
    "t5",
    "mt5",
    "flan-t5",
    "bart",
    "mbart",
    "marian",
    "nllb",
    "pegasus",
    "longt5",
    "byt5",
}

# Arquiteturas que parecem NLP mas normalmente não são LLMs generativos.
ENCODER_ONLY_OR_NON_GENERATIVE_TYPES = {
    "bert",
    "roberta",
    "distilbert",
    "xlm-roberta",
    "xlm_roberta",
    "deberta",
    "deberta-v2",
    "deberta_v2",
    "electra",
    "albert",
    "camembert",
    "mpnet",
    "xlm",
    "flaubert",
    "ernie",
    "rembert",
    "layoutlm",
    "layoutlmv2",
    "layoutlmv3",
    "canine",
    "big_bird",
    "bigbird",
}

# Tipos claramente fora de LLM textual.
NON_LANGUAGE_OR_MODAL_TYPES = {
    "clip",
    "vit",
    "swin",
    "convnext",
    "resnet",
    "efficientnet",
    "yolos",
    "detr",
    "segformer",
    "sam",
    "stable-diffusion",
    "stable_diffusion",
    "unet",
    "vae",
    "diffusion",
    "whisper",
    "wav2vec2",
    "hubert",
    "seamless_m4t",
    "speecht5",
    "parler_tts",
    "bark",
    "musicgen",
    "encodec",
    "timesformer",
    "vision-encoder-decoder",
    "vision_encoder_decoder",
}

# Famílias/nome fortes para LLM.
CORE_LLM_NAME_PATTERNS = [
    r"\bllama\b",
    r"\bqwen\b",
    r"\bmistral\b",
    r"\bmixtral\b",
    r"\bgemma\b",
    r"\bdeepseek\b",
    r"\bphi[-_ ]?\d*\b",
    r"\bolmo\b",
    r"\bfalcon\b",
    r"\bstarcoder\b",
    r"\bcodellama\b",
    r"\bbaichuan\b",
    r"\byi[-_ ]?\d*\b",
    r"\bbloom\b",
    r"\bgpt[-_ ]?neo\b",
    r"\bgpt[-_ ]?neox\b",
    r"\bgpt[-_ ]?j\b",
    r"\bgpt2\b",
    r"\bmpt\b",
    r"\bstablelm\b",
    r"\binternlm\b",
    r"\bchatglm\b",
    r"\bglm[-_ ]?\d*\b",
    r"\bminicpm\b",
    r"\baquila\b",
    r"\bxglm\b",
    r"\brwkv\b",
    r"\bmamba\b",
    r"\bnemotron\b",
    r"\bgranite\b",
    r"\bopenelm\b",
    r"\bexaone\b",
    r"\bsmollm\b",
    r"\bdbrx\b",
    r"\bjamba\b",
    r"\bsolar\b",
]

# Termos fortes de uso generativo/chat/instruct.
GENERATION_USAGE_PATTERNS = [
    r"\binstruct\b",
    r"\bchat\b",
    r"\bassistant\b",
    r"\breasoning\b",
    r"\bbase\b",
    r"\bcausal[-_ ]?lm\b",
    r"\bcausallm\b",
    r"\btext[-_ ]?generation\b",
    r"\bgeneration\b",
    r"\bdecoder[-_ ]?only\b",
    r"\blanguage[-_ ]?model\b",
    r"\bllm\b",
    r"\bcoder\b",
    r"\bcode\b",
]

EMBEDDING_PATTERNS = [
    r"\bembedding\b",
    r"\bembeddings\b",
    r"\bembed\b",
    r"\bsentence[-_ ]?transformer\b",
    r"\bsentence[-_ ]?similarity\b",
    r"\bretriever\b",
    r"\breranker\b",
    r"\brank\b",
    r"\bcolbert\b",
    r"\bcontriever\b",
    r"\bdpr\b",
    r"\bbge\b",
    r"\be5\b",
    r"\bgte\b",
    r"\bmxbai\b",
]

CLASSIFIER_PATTERNS = [
    r"\bclassification\b",
    r"\bclassifier\b",
    r"\bsentiment\b",
    r"\bner\b",
    r"\btoken[-_ ]?classification\b",
    r"\bsequence[-_ ]?classification\b",
    r"\bforsequenceclassification\b",
    r"\bfortokenclassification\b",
    r"\bquestion[-_ ]?answering\b",
]

QUANTIZATION_PATTERNS = [
    r"\bgguf\b",
    r"\bgptq\b",
    r"\bawq\b",
    r"\bexl2\b",
    r"\bexllama\b",
    r"\bquant\b",
    r"\bquantized\b",
    r"\bint4\b",
    r"\bint8\b",
    r"\b4bit\b",
    r"\b8bit\b",
    r"\bnf4\b",
    r"\bq2_k\b",
    r"\bq3_k\b",
    r"\bq4\b",
    r"\bq4_k\b",
    r"\bq5\b",
    r"\bq5_k\b",
    r"\bq6\b",
    r"\bq8\b",
    r"\bfp8\b",
]

ADAPTER_PATTERNS = [
    r"\blora\b",
    r"\badapter\b",
    r"\badapters\b",
    r"\bqlora\b",
    r"\bpeft\b",
    r"\bdora\b",
    r"\bdelta\b",
]

CHECKPOINT_PATTERNS = [
    r"\bcheckpoint\b",
    r"\bckpt\b",
    r"\bepoch[-_ ]?\d+\b",
    r"\bstep[-_ ]?\d+\b",
    r"\bglobal[-_ ]?step\b",
    r"\biter[-_ ]?\d+\b",
    r"\bintermediate\b",
]

MERGE_PATTERNS = [
    r"\bmerge\b",
    r"\bmerged\b",
    r"\bslerp\b",
    r"\bties\b",
    r"\bdare\b",
    r"\bmodel[-_ ]?stock\b",
]


# =============================================================================
# UTILITÁRIOS DE I/O
# =============================================================================

def find_latest_input_file() -> Path:
    parquet_files = sorted(
        SNAPSHOT_DIR.glob("*.flat.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if parquet_files:
        return parquet_files[0]

    # The repository ships the snapshot split into parts (*.flat.part-NN.parquet), read in order as one table.
    parts = sorted(SNAPSHOT_DIR.glob("*.flat.part-*.parquet"))
    if parts:
        return parts[0]

    csv_files = sorted(
        SNAPSHOT_DIR.glob("*.flat.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if csv_files:
        return csv_files[0]

    raise FileNotFoundError(
        f"Nenhum arquivo .flat.parquet ou .flat.csv encontrado em {SNAPSHOT_DIR}"
    )


def iter_input_chunks(path: Path, chunk_size: int) -> Iterable[pd.DataFrame]:
    if path.suffix == ".csv":
        for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
            yield chunk
        return

    if path.suffix == ".parquet":
        part = re.fullmatch(r"(.+\.flat)\.part-\d+\.parquet", path.name)
        paths = sorted(path.parent.glob(part.group(1) + ".part-*.parquet")) if part else [path]
        for one in paths:
            pf = pq.ParquetFile(one)
            for batch in pf.iter_batches(batch_size=chunk_size):
                table = pa.Table.from_batches([batch])
                yield table.to_pandas()
        return

    raise ValueError(f"Formato não suportado: {path}")


class DatasetWriter:
    def __init__(self, parquet_path: Path, csv_path: Optional[Path] = None):
        self.parquet_path = parquet_path
        self.csv_path = csv_path
        self.writer = None
        self.schema = None
        self.rows = 0

    def write(self, df: pd.DataFrame):
        if df is None or len(df) == 0:
            return

        df = normalize_output_df(df)
        table = pa.Table.from_pandas(df, preserve_index=False)

        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(
                self.parquet_path,
                self.schema,
                compression="snappy",
            )
        else:
            table = table.cast(self.schema)

        self.writer.write_table(table)
        self.rows += len(df)

        if self.csv_path is not None:
            write_header = not self.csv_path.exists()
            df.to_csv(self.csv_path, mode="a", index=False, header=write_header)

    def close(self):
        if self.writer is not None:
            self.writer.close()


# =============================================================================
# NORMALIZAÇÃO
# =============================================================================

def is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        return pd.isna(x)
    except Exception:
        return False


def norm_str(x: Any) -> str:
    if is_missing(x):
        return ""
    return str(x).strip()


def norm_lower(x: Any) -> str:
    return norm_str(x).lower()


def split_pipe_field(x: Any) -> List[str]:
    s = norm_str(x)
    if not s:
        return []
    return [v.strip() for v in s.split("|") if v.strip()]


def safe_bool(x: Any) -> bool:
    s = norm_lower(x)
    return s in {"true", "1", "yes", "y"}


def safe_int(x: Any) -> Optional[int]:
    if is_missing(x):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    if is_missing(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def log1p_or_zero(x: Any) -> float:
    val = safe_int(x)
    if val is None or val < 0:
        return 0.0
    return math.log1p(val)


def pipe_join(values: Iterable[Any]) -> str:
    out = []
    for v in values:
        s = norm_str(v)
        if s:
            out.append(s)
    return "|".join(sorted(set(out)))


def has_pattern(text: str, patterns: List[str]) -> bool:
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False


def token_in_set(value: str, options: set) -> bool:
    v = norm_lower(value)
    return v in options


def normalize_model_type(x: Any) -> str:
    s = norm_lower(x)
    s = s.replace("_", "-")
    return s


def normalize_output_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza tipos para evitar erro de PyArrow.
    """
    df = df.copy()

    # Ordena colunas de modo estável: preferred + novas colunas.
    preferred = [c for c in PREFERRED_COLUMNS if c in df.columns]
    remaining = [c for c in df.columns if c not in preferred]
    df = df[preferred + remaining]

    for col in df.columns:
        if col in {
            "downloads_30d",
            "downloads_all_time",
            "likes",
            "children_model_count",
            "parameter_count",
            "context_length",
            "num_siblings",
            "has_config_json",
            "has_tokenizer",
            "has_safetensors",
            "has_pytorch_bin",
            "has_gguf",
            "num_spaces",
            "parameter_count_clean",
            "downloads_30d_clean",
            "downloads_all_time_clean",
            "likes_clean",
            "children_model_count_clean",
            "llm_relevance_score",
            "core_relevance_score",
        }:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif col in {
            "trending_score",
            "adoption_proxy_log",
        }:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")
        elif col in {
            "keep_broad",
            "is_core_text_llm",
            "is_multimodal_language",
            "is_noncore_language_like",
            "is_quantized",
            "is_adapter",
            "is_checkpoint",
            "is_merge",
            "is_embedding_like",
            "is_classifier_like",
            "is_encoder_only_like",
            "is_non_language_or_modal",
            "has_weight_files",
            "duplicate_repo_id",
            "duplicate_sha",
        }:
            df[col] = df[col].fillna(False).astype(bool)
        else:
            df[col] = df[col].map(norm_str).astype("string")

    return df


# =============================================================================
# TEXTO DE BUSCA E FEATURES
# =============================================================================

def make_search_text(row: pd.Series) -> str:
    fields = [
        "repo_id",
        "author",
        "pipeline_tag",
        "library_name",
        "tags",
        "license",
        "languages",
        "datasets",
        "architecture",
        "model_type",
        "model_family_guess",
        "base_models",
        "artifact_type_guess",
        "siblings",
        "card_data_json",
        "config_json",
    ]
    return " ".join(norm_lower(row.get(f, "")) for f in fields)


def extract_model_type_family(row: pd.Series) -> Tuple[str, str, str]:
    model_type = normalize_model_type(row.get("model_type", ""))
    family = normalize_model_type(row.get("model_family_guess", ""))
    architecture = normalize_model_type(row.get("architecture", ""))
    return model_type, family, architecture


def has_core_generative_type(row: pd.Series, text: str) -> bool:
    model_type, family, architecture = extract_model_type_family(row)

    type_tokens = {model_type, family}
    arch_tokens = set()
    for part in re.split(r"[,|;/\s]+", architecture):
        p = normalize_model_type(part)
        if p:
            arch_tokens.add(p)

    all_tokens = type_tokens | arch_tokens

    for t in all_tokens:
        if t in CORE_GENERATIVE_MODEL_TYPES:
            return True

    if has_pattern(text, CORE_LLM_NAME_PATTERNS):
        return True

    return False


def has_seq2seq_generative_type(row: pd.Series, text: str) -> bool:
    model_type, family, architecture = extract_model_type_family(row)

    tokens = {model_type, family}
    for part in re.split(r"[,|;/\s]+", architecture):
        p = normalize_model_type(part)
        if p:
            tokens.add(p)

    for t in tokens:
        if t in SEQ2SEQ_GENERATIVE_MODEL_TYPES:
            return True

    # fallback por nome/tags
    if re.search(r"\b(flan[-_ ]?t5|mt5|byt5|longt5|nllb|mbart|marian)\b", text, re.I):
        return True

    return False


def is_encoder_only_like(row: pd.Series, text: str) -> bool:
    model_type, family, architecture = extract_model_type_family(row)

    tokens = {model_type, family}
    for part in re.split(r"[,|;/\s]+", architecture):
        p = normalize_model_type(part)
        if p:
            tokens.add(p)

    if any(t in ENCODER_ONLY_OR_NON_GENERATIVE_TYPES for t in tokens):
        return True

    # Alguns aparecem só no repo_id.
    if re.search(
        r"\b(bert|roberta|distilbert|deberta|electra|albert|mpnet|xlm-roberta|camembert)\b",
        text,
        re.I,
    ):
        return True

    return False


def is_non_language_or_modal_type(row: pd.Series, text: str) -> bool:
    model_type, family, architecture = extract_model_type_family(row)

    tokens = {model_type, family}
    for part in re.split(r"[,|;/\s]+", architecture):
        p = normalize_model_type(part)
        if p:
            tokens.add(p)

    if any(t in NON_LANGUAGE_OR_MODAL_TYPES for t in tokens):
        return True

    if re.search(
        r"\b(stable[-_ ]?diffusion|whisper|wav2vec|hubert|parler[-_ ]?tts|bark|musicgen|clip|vit|diffusion|flux|sdxl)\b",
        text,
        re.I,
    ):
        return True

    return False


# =============================================================================
# CLASSIFICAÇÃO PRINCIPAL
# =============================================================================

def classify_row(
    row: pd.Series,
    seen_repo_ids: set,
    seen_sha: set,
) -> Dict[str, Any]:
    repo_id = norm_str(row.get("repo_id", ""))
    repo_id_norm = repo_id.lower()

    sha = norm_str(row.get("sha", ""))
    pipeline_tag = norm_lower(row.get("pipeline_tag", ""))
    tags = split_pipe_field(row.get("tags", ""))
    text = make_search_text(row)

    broad_reject_reasons = []
    warnings = []
    evidence = []

    # -------------------------------------------------------------------------
    # Duplicatas e estado do repo
    # -------------------------------------------------------------------------

    duplicate_repo_id = False
    duplicate_sha = False

    if not repo_id:
        broad_reject_reasons.append("missing_repo_id")
    else:
        if repo_id_norm in seen_repo_ids:
            duplicate_repo_id = True
            broad_reject_reasons.append("duplicate_repo_id")
        seen_repo_ids.add(repo_id_norm)

    if sha:
        if sha in seen_sha:
            duplicate_sha = True
            # Não rejeito por SHA duplicado por padrão; pode ser reupload relevante.
            warnings.append("duplicate_sha")
        seen_sha.add(sha)

    if safe_bool(row.get("private", False)):
        broad_reject_reasons.append("private_repo")

    if safe_bool(row.get("disabled", False)):
        broad_reject_reasons.append("disabled_repo")

    gated_value = norm_lower(row.get("gated", ""))
    if gated_value and gated_value not in {"false", "none", "nan", "0"}:
        warnings.append("gated_model")
        if not KEEP_GATED_MODELS:
            broad_reject_reasons.append("gated_repo")

    # -------------------------------------------------------------------------
    # Arquivos de peso
    # -------------------------------------------------------------------------

    has_safetensors = safe_bool(row.get("has_safetensors", False))
    has_pytorch_bin = safe_bool(row.get("has_pytorch_bin", False))
    has_gguf = safe_bool(row.get("has_gguf", False))
    has_weight_files = has_safetensors or has_pytorch_bin or has_gguf

    if REQUIRE_WEIGHT_FILES and not has_weight_files:
        broad_reject_reasons.append("no_weight_files_detected")

    if has_safetensors:
        evidence.append("has_safetensors")
    if has_pytorch_bin:
        evidence.append("has_pytorch_bin")
    if has_gguf:
        evidence.append("has_gguf")

    # -------------------------------------------------------------------------
    # Tipos de artefato
    # -------------------------------------------------------------------------

    artifact_guess = norm_lower(row.get("artifact_type_guess", ""))
    # Declared by the Hub in baseModels; empty in snapshots collected before 2026-11.
    base_relation = norm_lower(row.get("base_model_relation", ""))

    is_quantized = (
        base_relation == "quantized"
        or has_gguf
        or "quantization" in artifact_guess
        or has_pattern(text, QUANTIZATION_PATTERNS)
    )

    is_adapter = (
        base_relation == "adapter"
        or "adapter" in artifact_guess
        or has_pattern(text, ADAPTER_PATTERNS)
    )

    is_checkpoint = (
        "checkpoint" in artifact_guess
        or has_pattern(text, CHECKPOINT_PATTERNS)
    )

    is_merge = (
        base_relation == "merge"
        or "merge" in artifact_guess
        or has_pattern(text, MERGE_PATTERNS)
    )

    if is_quantized:
        warnings.append("quantized_or_compressed_variant")
        if DROP_QUANTIZATIONS:
            broad_reject_reasons.append("quantized_or_compressed_variant")

    if is_adapter:
        warnings.append("adapter_or_lora_variant")
        if DROP_ADAPTERS:
            broad_reject_reasons.append("adapter_or_lora_variant")

    if is_checkpoint:
        warnings.append("checkpoint_or_intermediate_artifact")
        if DROP_CHECKPOINTS:
            broad_reject_reasons.append("checkpoint_or_intermediate_artifact")

    if is_merge:
        warnings.append("merge_model")
        if DROP_MERGES:
            broad_reject_reasons.append("merge_model")

    # -------------------------------------------------------------------------
    # Relevância language-like / broad
    # -------------------------------------------------------------------------

    score = 0

    if pipeline_tag in CORE_TEXT_PIPELINES:
        score += 4
        evidence.append(f"core_text_pipeline:{pipeline_tag}")

    if pipeline_tag in MULTIMODAL_LANGUAGE_PIPELINES:
        score += 3
        evidence.append(f"multimodal_language_pipeline:{pipeline_tag}")

    if pipeline_tag in NONCORE_LANGUAGE_PIPELINES:
        score += 1
        warnings.append(f"noncore_language_pipeline:{pipeline_tag}")

    if pipeline_tag in HARD_EXCLUDE_PIPELINES:
        score -= 6
        warnings.append(f"hard_excluded_pipeline:{pipeline_tag}")

    core_type = has_core_generative_type(row, text)
    seq2seq_type = has_seq2seq_generative_type(row, text)
    encoder_only = is_encoder_only_like(row, text)
    non_language_or_modal = is_non_language_or_modal_type(row, text)

    if core_type:
        score += 5
        evidence.append("core_generative_model_type_or_name")

    if seq2seq_type:
        score += 3
        evidence.append("seq2seq_generative_model_type_or_name")

    if has_pattern(text, GENERATION_USAGE_PATTERNS):
        score += 2
        evidence.append("generation_usage_pattern")

    if safe_bool(row.get("has_config_json", False)):
        score += 1
        evidence.append("has_config_json")

    if safe_bool(row.get("has_tokenizer", False)):
        score += 1
        evidence.append("has_tokenizer")

    is_embedding_like = (
        has_pattern(text, EMBEDDING_PATTERNS)
        or pipeline_tag in {"feature-extraction", "sentence-similarity", "text-embedding"}
    )

    is_classifier_like = (
        has_pattern(text, CLASSIFIER_PATTERNS)
        or pipeline_tag in {"text-classification", "token-classification", "zero-shot-classification"}
    )

    if is_embedding_like:
        score -= 5
        warnings.append("embedding_or_retrieval_like")

    if is_classifier_like:
        score -= 3
        warnings.append("classifier_or_task_specific_like")

    if encoder_only:
        score -= 3
        warnings.append("encoder_only_or_masked_lm_like")

    if non_language_or_modal:
        score -= 6
        warnings.append("non_language_or_modal_type")

    if score < MIN_BROAD_LLM_RELEVANCE_SCORE:
        broad_reject_reasons.append("low_llm_relevance_score")

    if pipeline_tag in HARD_EXCLUDE_PIPELINES:
        broad_reject_reasons.append(f"hard_excluded_pipeline:{pipeline_tag}")

    # -------------------------------------------------------------------------
    # Missingness warnings
    # -------------------------------------------------------------------------

    if not norm_str(row.get("license", "")):
        warnings.append("missing_license")

    if not norm_str(row.get("languages", "")):
        warnings.append("missing_language_metadata")

    if not norm_str(row.get("base_models", "")):
        warnings.append("missing_base_model_metadata")

    # -------------------------------------------------------------------------
    # Classificação em subpopulações
    # -------------------------------------------------------------------------

    keep_broad = len(broad_reject_reasons) == 0

    core_reasons = []
    core_score = 0

    if pipeline_tag in CORE_TEXT_PIPELINES:
        core_score += 4
        core_reasons.append(f"core_text_pipeline:{pipeline_tag}")

    if core_type:
        core_score += 5
        core_reasons.append("core_generative_type")

    if seq2seq_type:
        core_score += 3
        core_reasons.append("seq2seq_generative_type")

    if has_pattern(text, GENERATION_USAGE_PATTERNS):
        core_score += 2
        core_reasons.append("generation_usage_pattern")

    if encoder_only:
        core_score -= 6
        core_reasons.append("exclude_encoder_only")

    if non_language_or_modal:
        core_score -= 8
        core_reasons.append("exclude_non_language_or_modal")

    if is_embedding_like:
        core_score -= 8
        core_reasons.append("exclude_embedding_like")

    if is_classifier_like and pipeline_tag not in {"question-answering", "multiple-choice"}:
        core_score -= 5
        core_reasons.append("exclude_classifier_like")

    if pipeline_tag in HARD_EXCLUDE_PIPELINES:
        core_score -= 10
        core_reasons.append(f"exclude_hard_pipeline:{pipeline_tag}")

    if pipeline_tag in MULTIMODAL_LANGUAGE_PIPELINES:
        core_score -= 6
        core_reasons.append(f"exclude_multimodal_pipeline:{pipeline_tag}")

    if pipeline_tag in NONCORE_LANGUAGE_PIPELINES:
        core_score -= 3
        core_reasons.append(f"exclude_noncore_language_pipeline:{pipeline_tag}")

    has_allowed_core_pipeline = pipeline_tag in CORE_TEXT_PIPELINES
    has_missing_pipeline = pipeline_tag == ""

    strong_missing_pipeline_case = (
        ALLOW_MISSING_PIPELINE_IN_CORE_WITH_STRONG_EVIDENCE
        and has_missing_pipeline
        and (core_type or seq2seq_type)
        and has_pattern(text, GENERATION_USAGE_PATTERNS)
        and not encoder_only
        and not non_language_or_modal
        and not is_embedding_like
    )

    is_core_text_llm = (
        keep_broad
        and core_score >= MIN_CORE_LLM_RELEVANCE_SCORE
        and not is_quantized
        and not is_adapter
        and not is_checkpoint
        and not encoder_only
        and not non_language_or_modal
        and not is_embedding_like
        and pipeline_tag not in HARD_EXCLUDE_PIPELINES
        and pipeline_tag not in MULTIMODAL_LANGUAGE_PIPELINES
        and (
            has_allowed_core_pipeline
            or strong_missing_pipeline_case
            or core_type
        )
    )

    is_multimodal_language = (
        keep_broad
        and pipeline_tag in MULTIMODAL_LANGUAGE_PIPELINES
        and not is_embedding_like
        and not is_quantized
        and not is_adapter
        and not is_checkpoint
    )

    is_noncore_language_like = (
        keep_broad
        and not is_core_text_llm
        and not is_multimodal_language
    )

    # -------------------------------------------------------------------------
    # Adoção observável simples
    # -------------------------------------------------------------------------

    parameter_count = safe_int(row.get("parameter_count", None))
    downloads_30d = safe_int(row.get("downloads_30d", None))
    downloads_all_time = safe_int(row.get("downloads_all_time", None))
    likes = safe_int(row.get("likes", None))
    children_model_count = safe_int(row.get("children_model_count", None))

    adoption_proxy_log = (
        log1p_or_zero(downloads_30d)
        + 0.25 * log1p_or_zero(downloads_all_time)
        + 0.50 * log1p_or_zero(likes)
        + 0.25 * log1p_or_zero(children_model_count)
    )

    return {
        "keep_broad": keep_broad,
        "is_core_text_llm": is_core_text_llm,
        "is_multimodal_language": is_multimodal_language,
        "is_noncore_language_like": is_noncore_language_like,

        "cleaning_reasons": pipe_join(broad_reject_reasons),
        "cleaning_warnings": pipe_join(warnings),
        "llm_relevance_score": score,
        "llm_relevance_evidence": pipe_join(evidence),

        "core_relevance_score": core_score,
        "core_relevance_evidence": pipe_join(core_reasons),

        "duplicate_repo_id": duplicate_repo_id,
        "duplicate_sha": duplicate_sha,

        "is_quantized": is_quantized,
        "is_adapter": is_adapter,
        "is_checkpoint": is_checkpoint,
        "is_merge": is_merge,
        "is_embedding_like": is_embedding_like,
        "is_classifier_like": is_classifier_like,
        "is_encoder_only_like": encoder_only,
        "is_non_language_or_modal": non_language_or_modal,
        "has_weight_files": has_weight_files,

        "adoption_proxy_log": adoption_proxy_log,
        "parameter_count_clean": parameter_count,
        "downloads_30d_clean": downloads_30d,
        "downloads_all_time_clean": downloads_all_time,
        "likes_clean": likes,
        "children_model_count_clean": children_model_count,
    }


# =============================================================================
# CONTADORES / SUMMARY
# =============================================================================

def update_counter_from_pipe(counter: Counter, value: Any):
    s = norm_str(value)
    if not s:
        counter["MISSING"] += 1
        return

    for part in s.split("|"):
        p = part.strip()
        if p:
            counter[p] += 1


def update_top_counter(counter: Counter, value: Any):
    s = norm_str(value)
    if not s:
        s = "MISSING"
    counter[s] += 1


def add_examples(example_dict: Dict[str, List[str]], reasons: str, repo_id: str):
    if not reasons:
        return

    for reason in reasons.split("|"):
        reason = reason.strip()
        if not reason:
            continue
        if len(example_dict[reason]) < N_EXAMPLES_PER_REASON:
            example_dict[reason].append(repo_id)


def write_summary(
    manifest: Dict[str, Any],
    examples_by_rejection_reason: Dict[str, List[str]],
    summary_path: Path,
):
    lines = []
    lines.append("=" * 80)
    lines.append("OOME CLEANING V2 SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Input file: {manifest['input_file']}")
    lines.append(f"Input rows: {manifest['counts']['input_rows']:,}")
    lines.append(f"Broad clean rows: {manifest['counts']['broad_clean_rows']:,}")
    lines.append(f"Core text LLM rows: {manifest['counts']['core_text_llm_rows']:,}")
    lines.append(f"Multimodal-language rows: {manifest['counts']['multimodal_language_rows']:,}")
    lines.append(f"Noncore language-like rows: {manifest['counts']['noncore_language_like_rows']:,}")
    lines.append(f"Rejected rows: {manifest['counts']['rejected_rows']:,}")

    if manifest["counts"]["input_rows"]:
        frac = manifest["counts"]["broad_clean_rows"] / manifest["counts"]["input_rows"]
        lines.append(f"Broad clean fraction: {frac:.4f}")

    if manifest["counts"]["broad_clean_rows"]:
        frac_core = manifest["counts"]["core_text_llm_rows"] / manifest["counts"]["broad_clean_rows"]
        lines.append(f"Core among broad fraction: {frac_core:.4f}")

    lines.append("")
    lines.append("Top rejection reasons:")
    for k, v in manifest["rejection_counts"][:40]:
        lines.append(f"  - {k}: {v:,}")

    lines.append("")
    lines.append("Top warnings among broad clean:")
    for k, v in manifest["broad_warning_counts"][:40]:
        lines.append(f"  - {k}: {v:,}")

    lines.append("")
    lines.append("Top core pipeline tags:")
    for k, v in manifest["core_top_pipeline_tags"][:40]:
        lines.append(f"  - {k}: {v:,}")

    lines.append("")
    lines.append("Top core model families:")
    for k, v in manifest["core_top_model_families"][:40]:
        lines.append(f"  - {k}: {v:,}")

    lines.append("")
    lines.append("Top multimodal-language pipeline tags:")
    for k, v in manifest["multimodal_top_pipeline_tags"][:40]:
        lines.append(f"  - {k}: {v:,}")

    lines.append("")
    lines.append("Top noncore pipeline tags:")
    for k, v in manifest["noncore_top_pipeline_tags"][:40]:
        lines.append(f"  - {k}: {v:,}")

    lines.append("")
    lines.append("Examples by rejection reason:")
    for reason, examples in examples_by_rejection_reason.items():
        lines.append(f"  - {reason}: {examples}")

    text = "\n".join(lines)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_path = Path(INPUT_FILE) if INPUT_FILE else find_latest_input_file()

    broad_parquet = OUT_DIR / "oome_broad_clean.parquet"
    core_parquet = OUT_DIR / "oome_core_text_llm.parquet"
    multimodal_parquet = OUT_DIR / "oome_multimodal_language.parquet"
    noncore_parquet = OUT_DIR / "oome_noncore_language_like.parquet"
    rejected_parquet = OUT_DIR / "oome_rejected.parquet"

    broad_csv = OUT_DIR / "oome_broad_clean.csv" if SAVE_CSV else None
    core_csv = OUT_DIR / "oome_core_text_llm.csv" if SAVE_CSV else None
    multimodal_csv = OUT_DIR / "oome_multimodal_language.csv" if SAVE_CSV else None
    noncore_csv = OUT_DIR / "oome_noncore_language_like.csv" if SAVE_CSV else None
    rejected_csv = OUT_DIR / "oome_rejected.csv" if SAVE_CSV else None

    manifest_path = OUT_DIR / "oome_cleaning_manifest.json"
    summary_path = OUT_DIR / "oome_cleaning_summary.txt"

    # Remove outputs antigos para evitar append acidental.
    for p in [
        broad_parquet, core_parquet, multimodal_parquet, noncore_parquet, rejected_parquet,
        broad_csv, core_csv, multimodal_csv, noncore_csv, rejected_csv,
        manifest_path, summary_path,
    ]:
        if p is not None and p.exists():
            p.unlink()

    writers = {
        "broad": DatasetWriter(broad_parquet, broad_csv),
        "core": DatasetWriter(core_parquet, core_csv),
        "multimodal": DatasetWriter(multimodal_parquet, multimodal_csv),
        "noncore": DatasetWriter(noncore_parquet, noncore_csv),
        "rejected": DatasetWriter(rejected_parquet, rejected_csv),
    }

    seen_repo_ids = set()
    seen_sha = set()

    counts = Counter()
    rejection_counts = Counter()
    broad_warning_counts = Counter()

    core_pipeline_counter = Counter()
    core_family_counter = Counter()
    multimodal_pipeline_counter = Counter()
    noncore_pipeline_counter = Counter()

    examples_by_rejection_reason = defaultdict(list)

    print("=" * 80)
    print("OOME CLEANING V2")
    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print("=" * 80)

    try:
        for chunk in tqdm(iter_input_chunks(input_path, CHUNK_SIZE), desc="Processing chunks", unit="chunk"):
            counts["input_rows"] += len(chunk)

            # Garante colunas esperadas.
            for col in PREFERRED_COLUMNS:
                if col not in chunk.columns:
                    chunk[col] = ""

            # Coletas antigas guardam um palpite por substring; recalcula com limites de token.
            chunk["model_family_guess"] = [
                corrected_family_guess(r, tg, lg, mt)
                for r, tg, lg, mt in zip(
                    chunk["repo_id"], chunk["tags"], chunk["languages"], chunk["model_type"]
                )
            ]

            # Classifica linha por linha.
            decisions = []
            for _, row in chunk.iterrows():
                decisions.append(classify_row(row, seen_repo_ids, seen_sha))

            decisions_df = pd.DataFrame(decisions)
            chunk2 = pd.concat(
                [
                    chunk.reset_index(drop=True),
                    decisions_df.reset_index(drop=True),
                ],
                axis=1,
            )

            broad_df = chunk2[chunk2["keep_broad"] == True].copy()
            core_df = chunk2[chunk2["is_core_text_llm"] == True].copy()
            multimodal_df = chunk2[chunk2["is_multimodal_language"] == True].copy()
            noncore_df = chunk2[chunk2["is_noncore_language_like"] == True].copy()
            rejected_df = chunk2[chunk2["keep_broad"] == False].copy()

            counts["broad_clean_rows"] += len(broad_df)
            counts["core_text_llm_rows"] += len(core_df)
            counts["multimodal_language_rows"] += len(multimodal_df)
            counts["noncore_language_like_rows"] += len(noncore_df)
            counts["rejected_rows"] += len(rejected_df)

            # Contadores de rejeição.
            for _, r in rejected_df.iterrows():
                repo_id = norm_str(r.get("repo_id", ""))
                reasons = norm_str(r.get("cleaning_reasons", ""))
                for reason in reasons.split("|"):
                    reason = reason.strip()
                    if reason:
                        rejection_counts[reason] += 1
                add_examples(examples_by_rejection_reason, reasons, repo_id)

            # Warnings no broad.
            for warnings in broad_df["cleaning_warnings"].fillna("").tolist():
                for w in str(warnings).split("|"):
                    w = w.strip()
                    if w:
                        broad_warning_counts[w] += 1

            # Top counters.
            for _, r in core_df.iterrows():
                update_top_counter(core_pipeline_counter, r.get("pipeline_tag", ""))
                update_top_counter(core_family_counter, r.get("model_family_guess", ""))

            for _, r in multimodal_df.iterrows():
                update_top_counter(multimodal_pipeline_counter, r.get("pipeline_tag", ""))

            for _, r in noncore_df.iterrows():
                update_top_counter(noncore_pipeline_counter, r.get("pipeline_tag", ""))

            # Escrita.
            writers["broad"].write(broad_df)
            writers["core"].write(core_df)
            writers["multimodal"].write(multimodal_df)
            writers["noncore"].write(noncore_df)
            writers["rejected"].write(rejected_df)

    finally:
        for w in writers.values():
            w.close()

    manifest = {
        "input_file": os.path.relpath(input_path),
        "output_dir": os.path.relpath(OUT_DIR),
        "config": {
            "require_weight_files": REQUIRE_WEIGHT_FILES,
            "keep_gated_models": KEEP_GATED_MODELS,
            "drop_quantizations": DROP_QUANTIZATIONS,
            "drop_adapters": DROP_ADAPTERS,
            "drop_checkpoints": DROP_CHECKPOINTS,
            "drop_merges": DROP_MERGES,
            "min_broad_llm_relevance_score": MIN_BROAD_LLM_RELEVANCE_SCORE,
            "min_core_llm_relevance_score": MIN_CORE_LLM_RELEVANCE_SCORE,
            "allow_missing_pipeline_in_core_with_strong_evidence": ALLOW_MISSING_PIPELINE_IN_CORE_WITH_STRONG_EVIDENCE,
        },
        "counts": {
            "input_rows": int(counts["input_rows"]),
            "broad_clean_rows": int(counts["broad_clean_rows"]),
            "core_text_llm_rows": int(counts["core_text_llm_rows"]),
            "multimodal_language_rows": int(counts["multimodal_language_rows"]),
            "noncore_language_like_rows": int(counts["noncore_language_like_rows"]),
            "rejected_rows": int(counts["rejected_rows"]),
        },
        "outputs": {
            "broad_clean_parquet": os.path.relpath(broad_parquet),
            "core_text_llm_parquet": os.path.relpath(core_parquet),
            "multimodal_language_parquet": os.path.relpath(multimodal_parquet),
            "noncore_language_like_parquet": os.path.relpath(noncore_parquet),
            "rejected_parquet": os.path.relpath(rejected_parquet),
            "broad_clean_csv": os.path.relpath(broad_csv) if broad_csv else None,
            "core_text_llm_csv": os.path.relpath(core_csv) if core_csv else None,
            "multimodal_language_csv": os.path.relpath(multimodal_csv) if multimodal_csv else None,
            "noncore_language_like_csv": os.path.relpath(noncore_csv) if noncore_csv else None,
            "rejected_csv": os.path.relpath(rejected_csv) if rejected_csv else None,
        },
        "rejection_counts": rejection_counts.most_common(),
        "broad_warning_counts": broad_warning_counts.most_common(),
        "core_top_pipeline_tags": core_pipeline_counter.most_common(100),
        "core_top_model_families": core_family_counter.most_common(100),
        "multimodal_top_pipeline_tags": multimodal_pipeline_counter.most_common(100),
        "noncore_top_pipeline_tags": noncore_pipeline_counter.most_common(100),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    write_summary(manifest, examples_by_rejection_reason, summary_path)

    print()
    print("=" * 80)
    print("FINALIZADO")
    print("=" * 80)
    print(f"Broad clean: {broad_parquet}")
    print(f"Core text LLM: {core_parquet}")
    print(f"Multimodal-language: {multimodal_parquet}")
    print(f"Noncore language-like: {noncore_parquet}")
    print(f"Rejected: {rejected_parquet}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()