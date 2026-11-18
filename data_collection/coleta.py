#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect a snapshot of every model on the Hugging Face Hub (network; about 3.2 million models in November 2026).

Run from the repository root with HF_SNAPSHOT_DIR pointing to a new folder:
    HF_SNAPSHOT_DIR=data/hf_snapshots_YYYYMMDD python data_collection/coleta.py
HF_MAX_MODELS limits the collection, for tests. A new snapshot is a new release: add it to release_config.py.

Changes from the June 2026 collector:
  - Uses `expand` instead of full=True, which returns no baseModels, safetensors, downloadsAllTime, or gguf.
  - Sorts by created_at: the default order (trending) changes during the crawl, so the cursor can repeat or skip models.
  - Fills base_models, parameter_count, context_length, and downloads_all_time, which were empty in June.

Outputs (timestamped):
  - .raw.jsonl.gz   : raw responses, the full ModelInfo
  - .flat.parquet   : flattened table, written in chunks with normalized types
  - .manifest.json  : collection statistics and parameters
  - .errors.jsonl   : per-model errors, if any

Comments below this docstring are partly in Portuguese.
"""

import os
import csv
import gzip
import json
import math
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from huggingface_hub import HfApi
import huggingface_hub

from family_guess import corrected_family_guess

import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

OUT_DIR = Path(os.environ.get("HF_SNAPSHOT_DIR", "data/hf_snapshots_new"))

# Opções:
#   "all_hub_models" -> coleta todos os modelos públicos indexáveis
#   "llm_candidates" -> coleta candidatos a LLMs via pipeline_tag + busca textual
COLLECTION_MODE = "all_hub_models"

# Use None para coletar tudo.
# Para teste rápido, use 1000, 10000 etc.
MAX_MODELS_TOTAL = int(os.environ["HF_MAX_MODELS"]) if os.environ.get("HF_MAX_MODELS") else None

# Token opcional:
#   export HF_TOKEN="hf_xxx"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# A API não aceita expand junto com full, cardData ou fetch_config.
EXPAND = [
    "author", "baseModels", "cardData", "config", "createdAt", "disabled", "downloads",
    "downloadsAllTime", "gated", "gguf", "inference", "lastModified", "library_name", "likes",
    "model-index", "pipeline_tag", "private", "safetensors", "sha", "siblings", "spaces", "tags",
    "transformersInfo", "trendingScore",
]
SORT = "created_at"
SAVE_CSV = False

# Tamanho do chunk para escrita incremental.
FLUSH_EVERY = 5000

# Se você já tem um .flat.csv da coleta que quebrou no final,
# coloque o caminho aqui para gerar o Parquet sem recoletar.
#
# Exemplo:
# RECOVER_FROM_EXISTING_FLAT_CSV = "data/hf_snapshots/hf_model_snapshot_all_hub_models_XXXX.flat.csv"
#
# Se None, faz coleta normal.
RECOVER_FROM_EXISTING_FLAT_CSV = None

# Campos JSON compactados no CSV/Parquet.
MAX_TEXT_FIELD_CHARS = 2000


# =============================================================================
# CONFIGURAÇÃO PARA llm_candidates
# =============================================================================

LLM_RELEVANT_PIPELINE_TAGS = [
    "text-generation",
    "text2text-generation",
    "conversational",
    "question-answering",
    "summarization",
    "translation",
    "fill-mask",
    "zero-shot-classification",
    "text-classification",
]

LLM_SEARCH_TERMS = [
    "llm",
    "chat",
    "instruct",
    "reasoning",
    "language model",
    "causal lm",
    "qwen",
    "llama",
    "mistral",
    "gemma",
    "deepseek",
    "phi",
    "olmo",
    "falcon",
    "baichuan",
    "yi",
]


# =============================================================================
# COLUNAS E TIPOS ESTÁVEIS PARA PARQUET
# =============================================================================

FLAT_COLUMNS = [
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

INT_COLUMNS = {
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
}

FLOAT_COLUMNS = {
    "trending_score",
}

STRING_COLUMNS = [c for c in FLAT_COLUMNS if c not in INT_COLUMNS and c not in FLOAT_COLUMNS]


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_datetime_to_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if hasattr(x, "isoformat"):
        return x.isoformat()
    return str(x)


def get_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def get_attr_any(obj: Any, names: List[str], default: Any = None) -> Any:
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return val
    return default


def to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if hasattr(obj, "isoformat"):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    if hasattr(obj, "__dict__"):
        return {
            str(k): to_jsonable(v)
            for k, v in obj.__dict__.items()
            if not k.startswith("_")
        }

    return str(obj)


def compact_json(obj: Any, max_chars: int = MAX_TEXT_FIELD_CHARS) -> str:
    if obj is None:
        return ""
    try:
        s = json.dumps(to_jsonable(obj), ensure_ascii=False, sort_keys=True)
    except Exception:
        s = str(obj)

    if len(s) > max_chars:
        return s[:max_chars] + "...[TRUNCATED]"
    return s


def listify(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def safe_int_or_none(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return int(float(x))
    except Exception:
        return None


def safe_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return None


def safe_str_or_empty(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def pipe_join(values: List[Any]) -> str:
    out = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    return "|".join(sorted(set(out)))


def extract_card_data_dict(model_info: Any) -> Dict[str, Any]:
    card_data = get_attr_any(model_info, ["card_data", "cardData"], None)
    if card_data is None:
        return {}

    card_json = to_jsonable(card_data)
    if isinstance(card_json, dict):
        return card_json

    return {"_raw_card_data": card_json}


def extract_config_dict(model_info: Any) -> Dict[str, Any]:
    config = get_attr(model_info, "config", None)
    if isinstance(config, dict):
        return to_jsonable(config)
    return {}


def extract_safetensors_dict(model_info: Any) -> Dict[str, Any]:
    safetensors = get_attr(model_info, "safetensors", None)
    st = to_jsonable(safetensors)
    if isinstance(st, dict):
        return st
    return {}


# =============================================================================
# INFERÊNCIAS BÁSICAS
# =============================================================================

def infer_license(tags: List[str], card_data: Dict[str, Any]) -> Optional[str]:
    for key in ["license", "licenses"]:
        val = card_data.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, list) and val:
            return ",".join(map(str, val))

    for tag in tags:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.replace("license:", "", 1)

    return None


def infer_languages(tags: List[str], card_data: Dict[str, Any]) -> List[str]:
    langs = []

    for key in ["language", "languages"]:
        val = card_data.get(key)
        if isinstance(val, str):
            langs.append(val)
        elif isinstance(val, list):
            langs.extend([str(v) for v in val])

    for tag in tags:
        if isinstance(tag, str) and tag.startswith("language:"):
            langs.append(tag.replace("language:", "", 1))

    return sorted(set([x for x in langs if x]))


def infer_datasets(card_data: Dict[str, Any], tags: List[str]) -> List[str]:
    datasets = []

    for key in ["datasets", "dataset", "train_datasets", "training_data"]:
        val = card_data.get(key)
        if isinstance(val, str):
            datasets.append(val)
        elif isinstance(val, list):
            datasets.extend([str(v) for v in val])

    for tag in tags:
        if isinstance(tag, str) and tag.startswith("dataset:"):
            datasets.append(tag.replace("dataset:", "", 1))

    return sorted(set([x for x in datasets if x]))


def infer_architecture(config: Dict[str, Any], tags: List[str]) -> Optional[str]:
    archs = config.get("architectures")
    if isinstance(archs, list) and len(archs) > 0:
        return ",".join(map(str, archs))

    model_type = config.get("model_type")
    if isinstance(model_type, str):
        return model_type

    for tag in tags:
        if isinstance(tag, str) and tag.startswith("architecture:"):
            return tag.replace("architecture:", "", 1)

    return None


def infer_model_family(repo_id: str, tags: List[str], config: Dict[str, Any]) -> Optional[str]:
    text = " ".join([repo_id.lower()] + [str(t).lower() for t in tags])

    families = [
        "llama", "qwen", "mistral", "mixtral", "gemma", "deepseek", "phi",
        "olmo", "falcon", "starcoder", "codellama", "baichuan", "yi",
        "bloom", "gpt2", "gpt-neo", "gpt-j", "mpt", "stablelm",
        "command-r", "openchat", "vicuna", "alpaca", "zephyr", "solar",
        "internlm", "minicpm", "glm", "aquila", "xglm", "t5", "flan-t5",
        "rwkv", "mamba",
    ]

    for fam in families:
        if fam in text:
            return fam

    model_type = config.get("model_type")
    if isinstance(model_type, str):
        return model_type.lower()

    return None


def extract_base_models(model_info: Any) -> tuple:
    """Return (repo ids, relation) from the baseModels field, which is a dict."""
    field = get_attr(model_info, "base_models", None)
    if not isinstance(field, dict):
        return [], None
    models = [m.get("id") for m in field.get("models") or [] if isinstance(m, dict)]
    return models, field.get("relation")


def extract_gguf_dict(model_info: Any) -> Dict[str, Any]:
    """Keep the numeric GGUF fields; chat templates stay in the raw record only."""
    gguf = get_attr(model_info, "gguf", None)
    if not isinstance(gguf, dict):
        return {}
    return {k: gguf[k] for k in ["total", "architecture", "context_length"] if k in gguf}


def infer_parameter_count(
    config: Dict[str, Any], safetensors: Dict[str, Any], gguf: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    total = safe_int_or_none(safetensors.get("total"))
    if total:
        return total

    pc = safetensors.get("parameters") or safetensors.get("parameter_count")

    if isinstance(pc, dict):
        total = 0
        found = False
        for v in pc.values():
            try:
                total += int(v)
                found = True
            except Exception:
                pass
        if found:
            return total

    for key in ["num_parameters", "n_params", "parameter_count"]:
        val = config.get(key)
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)

    return safe_int_or_none((gguf or {}).get("total")) or None


def infer_context_length(config: Dict[str, Any], gguf: Optional[Dict[str, Any]] = None) -> Optional[int]:
    keys = [
        "max_position_embeddings",
        "n_positions",
        "seq_length",
        "max_sequence_length",
        "model_max_length",
        "sliding_window",
    ]

    for key in keys:
        val = config.get(key)
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)

    return safe_int_or_none((gguf or {}).get("context_length")) or None


def detect_artifact_type(repo_id: str, tags: List[str], siblings: List[Any]) -> str:
    text = " ".join([repo_id.lower()] + [str(t).lower() for t in tags])
    filenames = " ".join([
        str(get_attr(s, "rfilename", "")).lower()
        for s in siblings
    ])

    if any(x in text for x in [
        "gguf", "gptq", "awq", "exl2", "bnb", "int4", "int8",
        "quantized", "quantization", "4bit", "8bit", "q4", "q5", "q8"
    ]):
        return "quantization_or_compressed_variant"

    if any(x in filenames for x in ["adapter_model", "adapter_config"]):
        return "adapter_or_lora"

    if any(x in text for x in ["lora", "adapter", "qlora", "peft"]):
        return "adapter_or_lora"

    if any(x in text for x in ["merge", "merged", "slerp", "ties", "dare"]):
        return "merge"

    if any(x in text for x in ["checkpoint", "ckpt", "global_step"]):
        return "checkpoint"

    return "unknown_or_canonical_candidate"


# =============================================================================
# FLATTEN
# =============================================================================

def flatten_model_info(model_info: Any) -> Dict[str, Any]:
    repo_id = get_attr_any(model_info, ["id", "modelId"], None)

    tags = listify(get_attr(model_info, "tags", []))
    siblings = listify(get_attr(model_info, "siblings", []))
    spaces = listify(get_attr(model_info, "spaces", []))
    base_models, base_model_relation = extract_base_models(model_info)

    card_data = extract_card_data_dict(model_info)
    config = extract_config_dict(model_info)
    safetensors = extract_safetensors_dict(model_info)
    gguf = extract_gguf_dict(model_info)

    license_name = infer_license(tags, card_data)
    languages = infer_languages(tags, card_data)
    datasets = infer_datasets(card_data, tags)
    architecture = infer_architecture(config, tags)
    model_family_guess = corrected_family_guess(
        repo_id or "", pipe_join(tags), pipe_join(languages), config.get("model_type")
    )
    parameter_count = infer_parameter_count(config, safetensors, gguf)
    context_length = infer_context_length(config, gguf)
    artifact_type_guess = detect_artifact_type(repo_id or "", tags, siblings)

    sibling_filenames = [
        get_attr(s, "rfilename", None)
        for s in siblings
        if get_attr(s, "rfilename", None)
    ]

    row = {
        "repo_id": repo_id,
        "author": get_attr(model_info, "author", None),
        "sha": get_attr(model_info, "sha", None),

        "downloads_30d": get_attr(model_info, "downloads", None),
        "downloads_all_time": get_attr(model_info, "downloads_all_time", None),
        "likes": get_attr(model_info, "likes", None),
        "trending_score": get_attr(model_info, "trending_score", None),
        "children_model_count": get_attr(model_info, "children_model_count", None),

        "created_at": safe_datetime_to_str(
            get_attr_any(model_info, ["created_at", "createdAt"], None)
        ),
        "last_modified": safe_datetime_to_str(
            get_attr_any(model_info, ["last_modified", "lastModified"], None)
        ),

        # Importante: esses campos ficam como string no Parquet,
        # porque `gated` pode ser False, True, "manual" etc.
        "private": get_attr(model_info, "private", None),
        "disabled": get_attr(model_info, "disabled", None),
        "gated": get_attr(model_info, "gated", None),
        "inference": get_attr(model_info, "inference", None),

        "pipeline_tag": get_attr(model_info, "pipeline_tag", None),
        "library_name": get_attr(model_info, "library_name", None),
        "tags": pipe_join(tags),

        "license": license_name,
        "languages": pipe_join(languages),
        "datasets": pipe_join(datasets),
        "architecture": architecture,
        "model_type": config.get("model_type"),
        "model_family_guess": model_family_guess,
        "base_models": pipe_join(base_models),
        "base_model_relation": base_model_relation,
        "parameter_count": parameter_count,
        "context_length": context_length,
        "artifact_type_guess": artifact_type_guess,

        "num_siblings": len(sibling_filenames),
        "has_config_json": int("config.json" in sibling_filenames),
        "has_tokenizer": int(any("tokenizer" in f.lower() for f in sibling_filenames)),
        "has_safetensors": int(any(f.endswith(".safetensors") for f in sibling_filenames)),
        "has_pytorch_bin": int(any(f.endswith(".bin") for f in sibling_filenames)),
        "has_gguf": int(any(f.endswith(".gguf") for f in sibling_filenames)),
        "num_spaces": len(spaces),
        "spaces": pipe_join(spaces),

        "card_data_json": compact_json(card_data),
        "config_json": compact_json(config),
        "safetensors_json": compact_json(safetensors),
        "gguf_json": compact_json(gguf) if gguf else "",
        "siblings": pipe_join(sibling_filenames[:300]),
    }

    # Garante colunas fixas.
    return {col: row.get(col, None) for col in FLAT_COLUMNS}


def raw_model_info_dict(model_info: Any) -> Dict[str, Any]:
    return {
        "snapshot_collected_at": utc_now_iso(),
        "model_info": to_jsonable(model_info),
    }


# =============================================================================
# NORMALIZAÇÃO PARA PARQUET/CSV
# =============================================================================

def normalize_flat_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza tipos para impedir erros do PyArrow.

    Principal correção:
      - `gated`, `private`, `disabled`, `inference` e campos mistos viram string.
      - contagens viram Int64 nullable.
      - score vira Float64.
    """
    df = df.copy()

    for col in FLAT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[FLAT_COLUMNS]

    for col in STRING_COLUMNS:
        df[col] = df[col].map(safe_str_or_empty).astype("string")

    for col in INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")

    return df


# =============================================================================
# PATHS / WRITERS
# =============================================================================

def build_snapshot_paths() -> Dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"hf_model_snapshot_{COLLECTION_MODE}_{stamp}"

    return {
        "jsonl_gz": OUT_DIR / f"{prefix}.raw.jsonl.gz",
        "csv": OUT_DIR / f"{prefix}.flat.csv",
        "parquet": OUT_DIR / f"{prefix}.flat.parquet",
        "manifest": OUT_DIR / f"{prefix}.manifest.json",
        "errors": OUT_DIR / f"{prefix}.errors.jsonl",
    }


class IncrementalParquetWriter:
    def __init__(self, path: Path):
        self.path = path
        self.writer = None
        self.schema = None
        self.n_rows = 0

    def write_df(self, df: pd.DataFrame):
        if df.empty:
            return

        df = normalize_flat_df(df)
        table = pa.Table.from_pandas(df, preserve_index=False)

        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="snappy")
        else:
            table = table.cast(self.schema)

        self.writer.write_table(table)
        self.n_rows += len(df)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def append_csv(path: Path, df: pd.DataFrame):
    df = normalize_flat_df(df)
    write_header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=write_header)


def write_error(errors_path: Path, repo_id: Optional[str], error: Exception):
    record = {
        "time": utc_now_iso(),
        "repo_id": repo_id,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    with open(errors_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =============================================================================
# ITERAÇÃO DE MODELOS
# =============================================================================

def iter_models(api: HfApi):
    common_kwargs = dict(
        limit=None,
        expand=EXPAND,
        sort=SORT,
        token=HF_TOKEN,
    )

    if COLLECTION_MODE == "all_hub_models":
        for model in api.list_models(**common_kwargs):
            yield model
        return

    if COLLECTION_MODE == "llm_candidates":
        seen = set()

        for pipeline_tag in LLM_RELEVANT_PIPELINE_TAGS:
            for model in api.list_models(pipeline_tag=pipeline_tag, **common_kwargs):
                repo_id = get_attr_any(model, ["id", "modelId"], None)
                if repo_id and repo_id not in seen:
                    seen.add(repo_id)
                    yield model

        for term in LLM_SEARCH_TERMS:
            for model in api.list_models(search=term, **common_kwargs):
                repo_id = get_attr_any(model, ["id", "modelId"], None)
                if repo_id and repo_id not in seen:
                    seen.add(repo_id)
                    yield model

        return

    raise ValueError(f"COLLECTION_MODE inválido: {COLLECTION_MODE}")


# =============================================================================
# RECUPERAR CSV EXISTENTE
# =============================================================================

def recover_parquet_from_existing_csv(csv_path: Path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_parquet = csv_path.with_suffix("").with_suffix(".recovered.parquet")
    manifest_path = csv_path.with_suffix("").with_suffix(".recovered.manifest.json")

    print("=" * 80)
    print("RECOVERING PARQUET FROM EXISTING CSV")
    print("=" * 80)
    print(f"Input CSV: {csv_path}")
    print(f"Output Parquet: {out_parquet}")
    print("=" * 80)

    writer = IncrementalParquetWriter(out_parquet)

    n_rows = 0
    chunksize = 50000

    for chunk in tqdm(
        pd.read_csv(csv_path, chunksize=chunksize, low_memory=False),
        desc="Converting CSV chunks",
        unit="chunk",
    ):
        writer.write_df(chunk)
        n_rows += len(chunk)

    writer.close()

    manifest = {
        "mode": "recover_from_existing_csv",
        "input_csv": str(csv_path),
        "output_parquet": str(out_parquet),
        "rows": n_rows,
        "created_at": utc_now_iso(),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("RECOVERY FINALIZADO")
    print("=" * 80)
    print(f"Rows: {n_rows:,}")
    print(f"Parquet: {out_parquet}")
    print(f"Manifest: {manifest_path}")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    if RECOVER_FROM_EXISTING_FLAT_CSV is not None:
        recover_parquet_from_existing_csv(Path(RECOVER_FROM_EXISTING_FLAT_CSV))
        return

    # Cards with malformed model-index would print one warning each.
    huggingface_hub.utils.logging.set_verbosity_error()

    paths = build_snapshot_paths()

    started_at = utc_now_iso()
    api = HfApi(token=HF_TOKEN)

    print("=" * 80)
    print("HF MODEL SNAPSHOT COLLECTION")
    print("=" * 80)
    print(f"collection_mode: {COLLECTION_MODE}")
    print(f"max_models_total: {MAX_MODELS_TOTAL}")
    print(f"expand: {EXPAND}")
    print(f"sort: {SORT}")
    print(f"out_jsonl_gz: {paths['jsonl_gz']}")
    print(f"out_csv: {paths['csv'] if SAVE_CSV else 'disabled'}")
    print(f"out_parquet: {paths['parquet']}")
    print("=" * 80)

    parquet_writer = IncrementalParquetWriter(paths["parquet"])

    buffer_rows = []

    count = 0
    error_count = 0
    first_repo_id = None
    last_repo_id = None

    sum_downloads_30d = 0
    sum_downloads_all_time = 0
    sum_likes = 0

    top_pipeline_tags = {}
    top_families_guess = {}
    top_artifact_types_guess = {}

    def update_counter_dict(d: Dict[str, int], key: Any):
        key = safe_str_or_empty(key)
        if not key:
            key = "MISSING"
        d[key] = d.get(key, 0) + 1

    def flush_buffer():
        nonlocal buffer_rows

        if not buffer_rows:
            return

        df_chunk = pd.DataFrame(buffer_rows, columns=FLAT_COLUMNS)

        if SAVE_CSV:
            append_csv(paths["csv"], df_chunk)
        parquet_writer.write_df(df_chunk)

        buffer_rows = []

    try:
        with gzip.open(paths["jsonl_gz"], "wt", encoding="utf-8") as raw_f:
            pbar = tqdm(desc="Collecting HF models", unit="model", mininterval=30)

            for model_info in iter_models(api):
                if MAX_MODELS_TOTAL is not None and count >= MAX_MODELS_TOTAL:
                    break

                repo_id = get_attr_any(model_info, ["id", "modelId"], None)

                try:
                    if first_repo_id is None:
                        first_repo_id = repo_id
                    last_repo_id = repo_id

                    raw_record = raw_model_info_dict(model_info)
                    raw_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")

                    flat_row = flatten_model_info(model_info)
                    buffer_rows.append(flat_row)

                    count += 1

                    d30 = safe_int_or_none(flat_row.get("downloads_30d"))
                    dall = safe_int_or_none(flat_row.get("downloads_all_time"))
                    likes = safe_int_or_none(flat_row.get("likes"))

                    sum_downloads_30d += d30 or 0
                    sum_downloads_all_time += dall or 0
                    sum_likes += likes or 0

                    update_counter_dict(top_pipeline_tags, flat_row.get("pipeline_tag"))
                    update_counter_dict(top_families_guess, flat_row.get("model_family_guess"))
                    update_counter_dict(top_artifact_types_guess, flat_row.get("artifact_type_guess"))

                    pbar.update(1)

                    if count % FLUSH_EVERY == 0:
                        raw_f.flush()
                        flush_buffer()

                except Exception as e:
                    error_count += 1
                    write_error(paths["errors"], repo_id, e)

            pbar.close()

        flush_buffer()

    finally:
        parquet_writer.close()

    finished_at = utc_now_iso()

    def top_dict(d: Dict[str, int], n: int = 50) -> Dict[str, int]:
        return dict(sorted(d.items(), key=lambda x: x[1], reverse=True)[:n])

    manifest = {
        "started_at": started_at,
        "finished_at": finished_at,
        "collection_mode": COLLECTION_MODE,
        "max_models_total": MAX_MODELS_TOTAL,
        "expand": EXPAND,
        "sort": SORT,
        "hf_token_used": bool(HF_TOKEN),
        "huggingface_hub_version": huggingface_hub.__version__,
        "outputs": {k: str(v) for k, v in paths.items() if SAVE_CSV or k != "csv"},
        "counts": {
            "models_collected": count,
            "errors": error_count,
            "first_repo_id": first_repo_id,
            "last_repo_id": last_repo_id,
            "parquet_rows_written": parquet_writer.n_rows,
        },
        "summary": {
            "sum_downloads_30d": sum_downloads_30d,
            "sum_downloads_all_time": sum_downloads_all_time,
            "sum_likes": sum_likes,
            "top_pipeline_tags": top_dict(top_pipeline_tags),
            "top_families_guess": top_dict(top_families_guess),
            "top_artifact_types_guess": top_dict(top_artifact_types_guess),
        },
        "llm_relevant_pipeline_tags": LLM_RELEVANT_PIPELINE_TAGS,
        "llm_search_terms": LLM_SEARCH_TERMS,
    }

    with open(paths["manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 80)
    print("FINALIZADO")
    print("=" * 80)
    print(f"Modelos coletados: {count:,}")
    print(f"Erros: {error_count:,}")
    print(f"JSONL bruto: {paths['jsonl_gz']}")
    print(f"CSV achatado: {paths['csv'] if SAVE_CSV else 'disabled'}")
    print(f"Parquet achatado: {paths['parquet']}")
    print(f"Manifest: {paths['manifest']}")
    print("=" * 80)


if __name__ == "__main__":
    main()