#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lineage step: assign every repository of the cleaned textual population to the identity hierarchy

    repo_id
      -> model_entity_id       # observed model entity: official variant, derivative, or mirror
      -> variant_id            # role-specific variant: base, instruct, chat, reasoning, code, etc.
      -> base_model_id         # family + series + size
      -> series_id             # qwen2.5, llama3.1, gemma3, etc.
      -> family_id             # qwen, llama, gemma, etc.

Family, series, size, and role come from the repository name, declared base models, Hub tags, model type, and
parameter count (never from free-form model cards). See the technical supplement, Section A.

Input (OOME_CLEAN_DIR, set by release_config.py):
    data/oome_clean_<release>/oome_core_text_llm.parquet

Outputs (OOME_LINEAGE_DIR):
    oome_models_with_lineage.parquet   one row per repository, input columns plus the lineage labels
    oome_model_entities.parquet, oome_variants.parquet, oome_base_models.parquet, oome_series.parquet,
    oome_families.parquet, oome_lineage_edges.parquet, oome_lineage_manifest.json, oome_lineage_summary.txt

    python run_all_experiments.py --only lineage
"""

import os
import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from family_guess import corrected_family_guess, family_relevant_tags

import pandas as pd
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

CLEAN_DIR = Path(os.environ.get("OOME_CLEAN_DIR", "data/oome_clean_20261105"))
INPUT_FILE = CLEAN_DIR / "oome_core_text_llm.parquet"
FALLBACK_INPUT_FILE = Path("data/oome_clean/oome_models_clean.parquet")

OUT_DIR = Path(os.environ.get("OOME_LINEAGE_DIR", "data/oome_lineages_20261105"))

# The Hub tag `custom_code` only says a repository ships its own modeling code, but the code-role pattern in
# infer_role() matched it. The human validation of the taxonomy found this on 2026-11-17 (see annotation/README.md).
# "1" (the default) applies the fix. The release reported in the paper (20261105) was built before the fix, so
# release_config.py sets "0" for it to reproduce the reported results, and defines release 20261105_rolefix with "1".
IGNORE_CUSTOM_CODE_TAG = os.environ.get("OOME_ROLE_IGNORE_CUSTOM_CODE", "1") == "1"

SAVE_CSV = False

# If True, each community derivative is kept as its own model entity.
# This prevents over-collapsing GPT-2/Llama/Qwen fine-tunes into the base model.
PRESERVE_COMMUNITY_DERIVATIVES = True

# If True, non-official mirrors/reuploads are kept as separate model entities.
# This is conservative and avoids merging unrelated repos.
PRESERVE_REUPLOADS_AS_ENTITIES = True

# If True, create inferred parent links from instruct/chat/reasoning/code to base.
CREATE_INFERRED_PARENT_EDGES = True

# If True, remove obvious multimodal families if they survived core cleaning.
FILTER_MULTIMODAL_LEAKS_IN_LINEAGE = True

# If True, write CSV copies. For large outputs, set False.
SAVE_LARGE_CSV = True

N_SUMMARY_EXAMPLES = 25


# =============================================================================
# STRICT FIELD CHOICES
# =============================================================================

# Family inference should avoid noisy free-form model card/config text.
# These fields are used for family/series inference.
FAMILY_FIELDS = [
    "repo_id",
    "author",
    "model_family_guess",
    "model_type",
    "architecture",
    "base_models",
    "tags",
]

# Role inference should mostly use names/tags/base refs, not arbitrary config JSON.
ROLE_FIELDS = [
    "repo_id",
    "base_models",
    "tags",
    "pipeline_tag",
]

# Size inference should use names, base refs, parameter_count, and config-derived parameter count.
SIZE_FIELDS = [
    "repo_id",
    "base_models",
    "tags",
    "model_type",
    "architecture",
]


# =============================================================================
# FAMILY / SERIES PATTERNS
# =============================================================================

# Patterns are intentionally stricter than v1.
# They mostly require separators/boundaries so short names like "yi" do not match random text.
FAMILY_SERIES_PATTERNS = [
    # Qwen
    (r"(?<![a-z0-9])qwq(?![a-z0-9])", "qwen", "qwq"),
    (r"(?<![a-z0-9])qwen[-_ ]?3(?:\.\d+)?(?![a-z0-9])", "qwen", "qwen3"),
    (r"(?<![a-z0-9])qwen[-_ ]?2\.5(?![a-z0-9])", "qwen", "qwen2.5"),
    (r"(?<![a-z0-9])qwen[-_ ]?2(?![a-z0-9])", "qwen", "qwen2"),
    (r"(?<![a-z0-9])qwen[-_ ]?1\.5(?![a-z0-9])", "qwen", "qwen1.5"),
    (r"(?<![a-z0-9])qwen(?![a-z0-9])", "qwen", "qwen"),

    # Llama
    (r"(?<![a-z0-9])code[-_ ]?llama(?![a-z0-9])", "llama", "codellama"),
    (r"(?<![a-z0-9])llama[-_ ]?4(?![a-z0-9])", "llama", "llama4"),
    (r"(?<![a-z0-9])llama[-_ ]?3\.3(?![a-z0-9])", "llama", "llama3.3"),
    (r"(?<![a-z0-9])llama[-_ ]?3\.2(?![a-z0-9])", "llama", "llama3.2"),
    (r"(?<![a-z0-9])llama[-_ ]?3\.1(?![a-z0-9])", "llama", "llama3.1"),
    (r"(?<![a-z0-9])llama[-_ ]?3(?![a-z0-9])", "llama", "llama3"),
    (r"(?<![a-z0-9])llama[-_ ]?2(?![a-z0-9])", "llama", "llama2"),
    (r"(?<![a-z0-9])llama(?![a-z0-9])", "llama", "llama"),

    # Gemma
    (r"(?<![a-z0-9])gemma[-_ ]?4(?![a-z0-9])", "gemma", "gemma4"),
    (r"(?<![a-z0-9])gemma[-_ ]?3(?![a-z0-9])", "gemma", "gemma3"),
    (r"(?<![a-z0-9])gemma[-_ ]?2(?![a-z0-9])", "gemma", "gemma2"),
    (r"(?<![a-z0-9])gemma(?![a-z0-9])", "gemma", "gemma"),

    # DeepSeek
    (r"(?<![a-z0-9])deepseek[-_ ]?r1(?![a-z0-9])", "deepseek", "deepseek-r1"),
    (r"(?<![a-z0-9])deepseek[-_ ]?v4(?![a-z0-9])", "deepseek", "deepseek-v4"),
    (r"(?<![a-z0-9])deepseek[-_ ]?v3(?![a-z0-9])", "deepseek", "deepseek-v3"),
    (r"(?<![a-z0-9])deepseek[-_ ]?v2(?![a-z0-9])", "deepseek", "deepseek-v2"),
    (r"(?<![a-z0-9])deepseek[-_ ]?coder(?![a-z0-9])", "deepseek", "deepseek-coder"),
    (r"(?<![a-z0-9])deepseek[-_ ]?math(?![a-z0-9])", "deepseek", "deepseek-math"),
    (r"(?<![a-z0-9])deepseek(?![a-z0-9])", "deepseek", "deepseek"),

    # Mistral
    (r"(?<![a-z0-9])mixtral(?![a-z0-9])", "mistral", "mixtral"),
    (r"(?<![a-z0-9])mistral[-_ ]?small(?![a-z0-9])", "mistral", "mistral-small"),
    (r"(?<![a-z0-9])mistral[-_ ]?medium(?![a-z0-9])", "mistral", "mistral-medium"),
    (r"(?<![a-z0-9])mistral[-_ ]?large(?![a-z0-9])", "mistral", "mistral-large"),
    (r"(?<![a-z0-9])mistral(?![a-z0-9])", "mistral", "mistral"),

    # Phi
    (r"(?<![a-z0-9])phi[-_ ]?4(?![a-z0-9])", "phi", "phi4"),
    (r"(?<![a-z0-9])phi[-_ ]?3\.5(?![a-z0-9])", "phi", "phi3.5"),
    (r"(?<![a-z0-9])phi[-_ ]?3(?![a-z0-9])", "phi", "phi3"),
    (r"(?<![a-z0-9])phi[-_ ]?2(?![a-z0-9])", "phi", "phi2"),
    (r"(?<![a-z0-9])phi(?![a-z0-9])", "phi", "phi"),

    # OLMo
    (r"(?<![a-z0-9])olmo[-_ ]?e(?![a-z0-9])", "olmo", "olmoe"),
    (r"(?<![a-z0-9])olmo[-_ ]?2(?![a-z0-9])", "olmo", "olmo2"),
    (r"(?<![a-z0-9])olmo(?![a-z0-9])", "olmo", "olmo"),

    # Other decoder LLMs
    (r"(?<![a-z0-9])falcon[-_ ]?3(?![a-z0-9])", "falcon", "falcon3"),
    (r"(?<![a-z0-9])falcon[-_ ]?2(?![a-z0-9])", "falcon", "falcon2"),
    (r"(?<![a-z0-9])falcon(?![a-z0-9])", "falcon", "falcon"),

    (r"(?<![a-z0-9])starcoder[-_ ]?2(?![a-z0-9])", "starcoder", "starcoder2"),
    (r"(?<![a-z0-9])starcoder(?![a-z0-9])", "starcoder", "starcoder"),

    (r"(?<![a-z0-9])baichuan[-_ ]?2(?![a-z0-9])", "baichuan", "baichuan2"),
    (r"(?<![a-z0-9])baichuan(?![a-z0-9])", "baichuan", "baichuan"),

    # Yi is dangerous because it is short. Keep strict separator boundary.
    (r"(?<![a-z0-9])yi[-_ ]?1\.5(?![a-z0-9])", "yi", "yi1.5"),
    (r"(?<![a-z0-9])yi[-_ ]?34b(?![a-z0-9])", "yi", "yi"),
    (r"(?<![a-z0-9])yi[-_ ]?9b(?![a-z0-9])", "yi", "yi"),
    (r"(?<![a-z0-9])yi[-_ ]?6b(?![a-z0-9])", "yi", "yi"),
    (r"(?<![a-z0-9])yi(?![a-z0-9])", "yi", "yi"),

    (r"(?<![a-z0-9])bloomz(?![a-z0-9])", "bloom", "bloomz"),
    (r"(?<![a-z0-9])bloom(?![a-z0-9])", "bloom", "bloom"),

    (r"(?<![a-z0-9])gpt[-_ ]?oss(?![a-z0-9])", "gpt-oss", "gpt-oss"),
    (r"(?<![a-z0-9])gpt[-_ ]?neox(?![a-z0-9])", "gpt-neox", "gpt-neox"),
    (r"(?<![a-z0-9])gpt[-_ ]?neo(?![a-z0-9])", "gpt-neo", "gpt-neo"),
    (r"(?<![a-z0-9])gpt[-_ ]?j(?![a-z0-9])", "gpt-j", "gpt-j"),
    (r"(?<![a-z0-9])gpt[-_ ]?2(?![a-z0-9])", "gpt2", "gpt2"),
    (r"(?<![a-z0-9])gpt2(?![a-z0-9])", "gpt2", "gpt2"),

    (r"(?<![a-z0-9])opt[-_ ]?\d", "opt", "opt"),
    (r"(?<![a-z0-9])opt(?![a-z0-9])", "opt", "opt"),

    (r"(?<![a-z0-9])mpt(?![a-z0-9])", "mpt", "mpt"),
    (r"(?<![a-z0-9])stable[-_ ]?lm(?![a-z0-9])", "stablelm", "stablelm"),

    (r"(?<![a-z0-9])internlm[-_ ]?3(?![a-z0-9])", "internlm", "internlm3"),
    (r"(?<![a-z0-9])internlm[-_ ]?2(?![a-z0-9])", "internlm", "internlm2"),
    (r"(?<![a-z0-9])internlm(?![a-z0-9])", "internlm", "internlm"),

    (r"(?<![a-z0-9])chatglm[-_ ]?3(?![a-z0-9])", "glm", "chatglm3"),
    (r"(?<![a-z0-9])chatglm[-_ ]?2(?![a-z0-9])", "glm", "chatglm2"),
    (r"(?<![a-z0-9])chatglm(?![a-z0-9])", "glm", "chatglm"),
    (r"(?<![a-z0-9])glm[-_ ]?4(?![a-z0-9])", "glm", "glm4"),
    (r"(?<![a-z0-9])glm(?![a-z0-9])", "glm", "glm"),

    (r"(?<![a-z0-9])minicpm(?![a-z0-9])", "minicpm", "minicpm"),
    (r"(?<![a-z0-9])aquila(?![a-z0-9])", "aquila", "aquila"),
    (r"(?<![a-z0-9])xglm(?![a-z0-9])", "xglm", "xglm"),
    (r"(?<![a-z0-9])rwkv(?![a-z0-9])", "rwkv", "rwkv"),

    (r"(?<![a-z0-9])mamba[-_ ]?2(?![a-z0-9])", "mamba", "mamba2"),
    (r"(?<![a-z0-9])mamba(?![a-z0-9])", "mamba", "mamba"),

    (r"(?<![a-z0-9])nemotron(?![a-z0-9])", "nemotron", "nemotron"),
    (r"(?<![a-z0-9])granite(?![a-z0-9])", "granite", "granite"),
    (r"(?<![a-z0-9])openelm(?![a-z0-9])", "openelm", "openelm"),
    (r"(?<![a-z0-9])exaone[-_ ]?4(?![a-z0-9])", "exaone", "exaone4"),
    (r"(?<![a-z0-9])exaone(?![a-z0-9])", "exaone", "exaone"),
    (r"(?<![a-z0-9])smollm[-_ ]?3(?![a-z0-9])", "smollm", "smollm3"),
    (r"(?<![a-z0-9])smollm[-_ ]?2(?![a-z0-9])", "smollm", "smollm2"),
    (r"(?<![a-z0-9])smollm(?![a-z0-9])", "smollm", "smollm"),
    (r"(?<![a-z0-9])dbrx(?![a-z0-9])", "dbrx", "dbrx"),
    (r"(?<![a-z0-9])jamba(?![a-z0-9])", "jamba", "jamba"),
    (r"(?<![a-z0-9])solar(?![a-z0-9])", "solar", "solar"),
    (r"(?<![a-z0-9])lfm2(?:\.5)?(?![a-z0-9])", "lfm2", "lfm2"),
    (r"(?<![a-z0-9])cohere2(?![a-z0-9])", "cohere", "cohere2"),
    (r"(?<![a-z0-9])command[-_ ]?r(?![a-z0-9])", "cohere", "command-r"),

    # Seq2seq
    (r"(?<![a-z0-9])flan[-_ ]?t5(?![a-z0-9])", "t5", "flan-t5"),
    (r"(?<![a-z0-9])long[-_ ]?t5(?![a-z0-9])", "t5", "long-t5"),
    (r"(?<![a-z0-9])byt5(?![a-z0-9])", "t5", "byt5"),
    (r"(?<![a-z0-9])mt5(?![a-z0-9])", "t5", "mt5"),
    (r"(?<![a-z0-9])t5(?![a-z0-9])", "t5", "t5"),
    (r"(?<![a-z0-9])mbart(?![a-z0-9])", "bart", "mbart"),
    (r"(?<![a-z0-9])bart(?![a-z0-9])", "bart", "bart"),
    (r"(?<![a-z0-9])marian(?![a-z0-9])", "marian", "marian"),
    (r"(?<![a-z0-9])nllb(?![a-z0-9])", "nllb", "nllb"),
    (r"(?<![a-z0-9])m2m[-_ ]?100(?![a-z0-9])", "m2m100", "m2m100"),
    (r"(?<![a-z0-9])pegasus(?![a-z0-9])", "pegasus", "pegasus"),
]


OFFICIAL_ORGS_BY_FAMILY = {
    "qwen": {"qwen", "alibaba", "alibaba-nlp"},
    "llama": {"meta-llama", "facebook", "meta"},
    "gemma": {"google", "google-gemma", "google-deepmind"},
    "deepseek": {"deepseek-ai"},
    "mistral": {"mistralai", "mistral"},
    "phi": {"microsoft"},
    "olmo": {"allenai"},
    "falcon": {"tiiuae"},
    "starcoder": {"bigcode"},
    "baichuan": {"baichuan-inc"},
    "yi": {"01-ai"},
    "bloom": {"bigscience"},
    "gpt2": {"openai", "openai-community"},
    "gpt-oss": {"openai"},
    "gpt-neox": {"eleutherai"},
    "gpt-neo": {"eleutherai"},
    "gpt-j": {"eleutherai"},
    "opt": {"facebook", "meta"},
    "mpt": {"mosaicml"},
    "stablelm": {"stabilityai"},
    "internlm": {"internlm"},
    "glm": {"thudm", "zai-org"},
    "minicpm": {"openbmb"},
    "t5": {"google", "google-t5"},
    "bart": {"facebook", "meta"},
    "marian": {"helsinki-nlp"},
    "nllb": {"facebook", "meta"},
    "m2m100": {"facebook", "meta"},
    "pegasus": {"google"},
    "granite": {"ibm-granite", "ibm", "ibm-research"},
    "openelm": {"apple"},
    "jamba": {"ai21labs"},
    "lfm2": {"liquidai"},
    "cohere": {"cohereforai", "cohere"},
    "exaone": {"lgai-exaone"},
    "nemotron": {"nvidia"},
}


# =============================================================================
# ROLE / TYPE PATTERNS
# =============================================================================

MULTIMODAL_LEAK_PATTERNS = [
    r"(?<![a-z0-9])llava(?![a-z0-9])",
    r"(?<![a-z0-9])internvl(?![a-z0-9])",
    r"(?<![a-z0-9])moondream(?![a-z0-9])",
    r"(?<![a-z0-9])mplug(?![a-z0-9])",
    r"(?<![a-z0-9])qwen[-_ ]?vl(?![a-z0-9])",
    r"(?<![a-z0-9])qwen2[-_. ]?5[-_ ]?vl(?![a-z0-9])",
    r"(?<![a-z0-9])qwen2[-_ ]?vl(?![a-z0-9])",
    r"(?<![a-z0-9])gemma[-_ ]?vision(?![a-z0-9])",
    r"(?<![a-z0-9])vision[-_ ]?language(?![a-z0-9])",
    r"(?<![a-z0-9])multi[-_ ]?modality(?![a-z0-9])",
    r"(?<![a-z0-9])vlm(?![a-z0-9])",
]

DERIVATIVE_PATTERNS = [
    r"(?<![a-z0-9])finetune(?![a-z0-9])",
    r"(?<![a-z0-9])fine[-_ ]?tuned(?![a-z0-9])",
    r"(?<![a-z0-9])sft(?![a-z0-9])",
    r"(?<![a-z0-9])dpo(?![a-z0-9])",
    r"(?<![a-z0-9])rlhf(?![a-z0-9])",
    r"(?<![a-z0-9])ppo(?![a-z0-9])",
    r"(?<![a-z0-9])merged?(?![a-z0-9])",
    r"(?<![a-z0-9])slerp(?![a-z0-9])",
    r"(?<![a-z0-9])ties(?![a-z0-9])",
    r"(?<![a-z0-9])dare(?![a-z0-9])",
    r"(?<![a-z0-9])hermes(?![a-z0-9])",
    r"(?<![a-z0-9])orca(?![a-z0-9])",
    r"(?<![a-z0-9])dolphin(?![a-z0-9])",
    r"(?<![a-z0-9])airoboros(?![a-z0-9])",
    r"(?<![a-z0-9])vicuna(?![a-z0-9])",
    r"(?<![a-z0-9])alpaca(?![a-z0-9])",
    r"(?<![a-z0-9])wizard(?![a-z0-9])",
    r"(?<![a-z0-9])openchat(?![a-z0-9])",
    r"(?<![a-z0-9])zephyr(?![a-z0-9])",
    r"(?<![a-z0-9])uncensored(?![a-z0-9])",
    r"(?<![a-z0-9])roleplay(?![a-z0-9])",
]

REASONING_PATTERNS = [
    r"(?<![a-z0-9])reasoning(?![a-z0-9])",
    r"(?<![a-z0-9])reasoner(?![a-z0-9])",
    r"(?<![a-z0-9])thinking(?![a-z0-9])",
    r"(?<![a-z0-9])think(?![a-z0-9])",
    r"(?<![a-z0-9])qwq(?![a-z0-9])",
    r"(?<![a-z0-9])r1(?![a-z0-9])",
]

INSTRUCT_PATTERNS = [
    r"(?<![a-z0-9])instruct(?![a-z0-9])",
    r"(?<![a-z0-9])instruction(?![a-z0-9])",
    r"(?<![a-z0-9])it(?![a-z0-9])",
]

CHAT_PATTERNS = [
    r"(?<![a-z0-9])chat(?![a-z0-9])",
    r"(?<![a-z0-9])assistant(?![a-z0-9])",
    r"(?<![a-z0-9])dialog(?![a-z0-9])",
    r"(?<![a-z0-9])conversation(?![a-z0-9])",
]

CODE_PATTERNS = [
    r"(?<![a-z0-9])code(?![a-z0-9])",
    r"(?<![a-z0-9])coder(?![a-z0-9])",
    r"(?<![a-z0-9])coding(?![a-z0-9])",
    r"(?<![a-z0-9])starcoder(?![a-z0-9])",
    r"(?<![a-z0-9])codellama(?![a-z0-9])",
]

MATH_PATTERNS = [
    r"(?<![a-z0-9])math(?![a-z0-9])",
    r"(?<![a-z0-9])gsm(?![a-z0-9])",
    r"(?<![a-z0-9])olympiad(?![a-z0-9])",
]

BASE_PATTERNS = [
    r"(?<![a-z0-9])base(?![a-z0-9])",
    r"(?<![a-z0-9])pretrain(?:ed)?(?![a-z0-9])",
    r"(?<![a-z0-9])foundation(?![a-z0-9])",
]

DOMAIN_PATTERNS = [
    r"(?<![a-z0-9])medical(?![a-z0-9])",
    r"(?<![a-z0-9])clinical(?![a-z0-9])",
    r"(?<![a-z0-9])legal(?![a-z0-9])",
    r"(?<![a-z0-9])finance(?![a-z0-9])",
    r"(?<![a-z0-9])bio(?![a-z0-9])",
]


# =============================================================================
# BASIC HELPERS
# =============================================================================

def norm_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def norm_lower(x: Any) -> str:
    return norm_str(x).lower()


def slugify(x: Any) -> str:
    s = norm_lower(x)
    s = s.replace("/", "__")
    s = s.replace("_", "-")
    s = re.sub(r"[^a-z0-9.+-]+", "-", s)
    s = re.sub(r"-+", "-", s)
    s = s.strip("-_.")
    return s or "unknown"


def split_pipe_field(x: Any) -> List[str]:
    s = norm_str(x)
    if not s:
        return []
    return [v.strip() for v in s.split("|") if v.strip()]


def safe_int(x: Any) -> Optional[int]:
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


def safe_float(x: Any) -> Optional[float]:
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


def log1p_or_zero(x: Any) -> float:
    v = safe_int(x)
    if v is None or v < 0:
        return 0.0
    return math.log1p(v)


def has_pattern(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def first_family_match(text: str) -> Optional[Tuple[str, str, str]]:
    for pattern, family, series in FAMILY_SERIES_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return family, series, f"pattern:{pattern}"
    return None


def get_repo_parts(repo_id: str) -> Tuple[str, str]:
    repo_id = norm_str(repo_id)
    if "/" in repo_id:
        a, b = repo_id.split("/", 1)
        return a, b
    return "", repo_id


def text_from_fields(row: pd.Series, fields: List[str]) -> str:
    return " ".join(norm_lower(row.get(f, "")) for f in fields)


def parse_base_models(x: Any) -> List[str]:
    vals = split_pipe_field(x)
    out = []
    for v in vals:
        v = v.strip("[]'\" ")
        if v:
            out.append(v)
    return sorted(set(out))


def normalize_name_for_similarity(x: str) -> str:
    s = norm_lower(x)
    s = s.replace("_", "-")
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9.+-]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def compute_adoption_proxy(row: pd.Series) -> float:
    existing = safe_float(row.get("adoption_proxy_log", None))
    if existing is not None and existing > 0:
        return existing

    return (
        log1p_or_zero(row.get("downloads_30d", None))
        + 0.25 * log1p_or_zero(row.get("downloads_all_time", None))
        + 0.50 * log1p_or_zero(row.get("likes", None))
        + 0.25 * log1p_or_zero(row.get("children_model_count", None))
    )


# =============================================================================
# FAMILY / SERIES / SIZE / ROLE
# =============================================================================

def infer_family_series(row: pd.Series) -> Tuple[str, str, str]:
    """
    More conservative than v1:
      1. declared base_models
      2. repo_id
      3. model_family_guess / model_type / architecture
      4. tags
      5. fallback
    """

    base_models = parse_base_models(row.get("base_models", ""))
    for base in base_models:
        match = first_family_match(norm_lower(base))
        if match:
            fam, series, ev = match
            return fam, series, f"declared_base_model:{base}:{ev}"

    repo_id = norm_lower(row.get("repo_id", ""))
    match = first_family_match(repo_id)
    if match:
        fam, series, ev = match
        return fam, series, f"repo_id:{ev}"

    # The stored guess may come from an older collector that matched substrings of tags,
    # so it is recomputed here. Language tags must not name a family (see family_guess.py).
    guess_value = corrected_family_guess(
        row.get("repo_id", ""), row.get("tags", ""), row.get("languages", ""), row.get("model_type", "")
    )
    structured_text = " ".join([
        norm_lower(guess_value),
        norm_lower(row.get("model_type", "")),
        norm_lower(row.get("architecture", "")),
    ])
    match = first_family_match(structured_text)
    if match:
        fam, series, ev = match
        return fam, series, f"structured_metadata:{ev}"

    tags_text = " ".join(family_relevant_tags(row.get("tags", ""), row.get("languages", "")))
    match = first_family_match(tags_text)
    if match:
        fam, series, ev = match
        return fam, series, f"tags:{ev}"

    # Fallback only if model_family_guess is not missing and not a weird broad token.
    guess = slugify(guess_value)
    if guess and guess not in {"missing", "none", "unknown"}:
        return guess, guess, "model_family_guess_fallback"

    model_type = slugify(row.get("model_type", ""))
    if model_type and model_type not in {"missing", "none", "unknown"}:
        return model_type, model_type, "model_type_fallback"

    return "unknown", "unknown", "missing_family_signal"


def extract_size_label(row: pd.Series) -> Tuple[str, str]:
    """
    Improved size extraction.

    Avoids random version numbers like v0.3.
    Looks for:
      - 8x7b / 16x3.8b
      - 0.5b / 1.5b / 7b / 70b / 125m
      - parameter_count fallback
    """
    text = text_from_fields(row, SIZE_FIELDS)
    repo_id = norm_lower(row.get("repo_id", ""))
    base_models_text = norm_lower(row.get("base_models", ""))

    candidates = [repo_id, base_models_text, text]

    moe_pattern = r"(?<![a-z0-9])(\d{1,3})x(\d+(?:\.\d+)?)(b|m)(?![a-z0-9])"
    size_pattern = r"(?<![a-z0-9.])(\d+(?:\.\d+)?)[-_ ]?(b|m)(?![a-z0-9])"

    for source_name, source in [("repo_id", repo_id), ("base_models", base_models_text), ("metadata", text)]:
        m = re.search(moe_pattern, source, flags=re.IGNORECASE)
        if m:
            return f"{m.group(1)}x{m.group(2).lower()}{m.group(3).lower()}", f"{source_name}:moe_size"

        m = re.search(size_pattern, source, flags=re.IGNORECASE)
        if m:
            val = float(m.group(1))
            unit = m.group(2).lower()

            # Avoid absurd accidental matches.
            if unit == "b" and 0.01 <= val <= 1000:
                label = f"{m.group(1).lower()}{unit}"
                return label, f"{source_name}:size_pattern"

            if unit == "m" and 1 <= val <= 999999:
                label = f"{m.group(1).lower()}{unit}"
                return label, f"{source_name}:size_pattern"

    pc = safe_int(row.get("parameter_count_clean", None))
    if pc is None:
        pc = safe_int(row.get("parameter_count", None))

    label = parameter_count_label(pc)
    if label:
        return label, "parameter_count"

    return "unknown-size", "missing_size_signal"


def parameter_count_label(pc: Optional[int]) -> str:
    """Coarse size label from an exact count, so fine-tunes differing by a few weights share a label."""
    if pc is None or pc <= 0:
        return ""
    if pc >= 1_000_000_000:
        val = pc / 1_000_000_000
        if val >= 10:
            return f"{round(val):g}b"
        return f"{float(f'{val:.1f}'):g}b"
    if pc >= 1_000_000:
        val = float(f"{pc / 1_000_000:.2g}")
        return "1b" if val >= 1000 else f"{val:g}m"
    return ""


def size_label_billions(size_label: str) -> Optional[float]:
    """Parameters in billions named by a size label, or None for unknown."""
    s = norm_lower(size_label)
    m = re.fullmatch(r"(\d+)x(\d+(?:\.\d+)?)(b|m)", s)
    if m:
        total = float(m.group(1)) * float(m.group(2))
        return total if m.group(3) == "b" else total / 1000.0
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(b|m)", s)
    if m:
        return float(m.group(1)) if m.group(2) == "b" else float(m.group(1)) / 1000.0
    return None


# A count snaps to a nominal size of its series when within this ratio: Llama 2 7B has 6.7B
# parameters, Qwen2.5 7B has 7.6B, and Llama 3.2 1B has 1.1B.
SNAP_MAX_RATIO = 1.25
# A nominal size needs this many named repositories, so one odd name does not attract counts.
SNAP_MIN_SUPPORT = 2
# Among close sizes, ignore labels far rarer than the commonest one, so that a few repositories
# named "7.6b" do not pull in every count near Qwen2.5 7B and split it from "7b" again.
SNAP_MIN_RELATIVE_SUPPORT = 0.10


def collect_nominal_sizes(df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, Tuple[float, int]]]:
    """Sizes stated by names in each family and series, with each label's value and support."""
    counts: Dict[Tuple[str, str], Dict[str, int]] = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Collecting nominal sizes", unit="model"):
        label, evidence = extract_size_label(row)
        if evidence in {"parameter_count", "missing_size_signal"}:
            continue
        family_id, series_id, _ = infer_family_series(row)
        bucket = counts.setdefault((family_id, series_id), {})
        bucket[label] = bucket.get(label, 0) + 1
    nominal: Dict[Tuple[str, str], Dict[str, Tuple[float, int]]] = {}
    for key, labels in counts.items():
        kept = {}
        for label, count in labels.items():
            value = size_label_billions(label)
            if count >= SNAP_MIN_SUPPORT and value:
                kept[label] = (value, count)
        if kept:
            nominal[key] = kept
    return nominal


def snap_parameter_size(
    row: pd.Series, family_id: str, series_id: str, label: str, evidence: str,
    nominal: Optional[Dict[Tuple[str, str], Dict[str, Tuple[float, int]]]],
) -> Tuple[str, str]:
    """Give a count-derived size the nominal label of its series when one is close enough."""
    if evidence != "parameter_count" or not nominal:
        return label, evidence
    options = nominal.get((family_id, series_id))
    pc = safe_int(row.get("parameter_count_clean", None)) or safe_int(row.get("parameter_count", None))
    if not options or not pc:
        return label, evidence
    value = pc / 1_000_000_000
    close = {name: opt for name, opt in options.items() if abs(math.log(value / opt[0])) <= math.log(SNAP_MAX_RATIO)}
    if not close:
        return label, evidence
    strongest = max(opt[1] for opt in close.values())
    common = {name: opt for name, opt in close.items() if opt[1] >= SNAP_MIN_RELATIVE_SUPPORT * strongest}
    best = min(common, key=lambda name: abs(math.log(value / common[name][0])))
    return best, f"parameter_count_snapped:{best}"


def size_to_bucket(size_label: str) -> str:
    s = norm_lower(size_label)

    m = re.search(r"(\d+)x(\d+(?:\.\d+)?)(b|m)", s)
    if m:
        n = float(m.group(1))
        each = float(m.group(2))
        unit = m.group(3)
        total_b = n * each if unit == "b" else n * each / 1000.0
        return numeric_size_to_bucket(total_b)

    m = re.search(r"(\d+(?:\.\d+)?)(b|m)", s)
    if not m:
        return "unknown"

    val = float(m.group(1))
    unit = m.group(2)
    total_b = val if unit == "b" else val / 1000.0
    return numeric_size_to_bucket(total_b)


def numeric_size_to_bucket(total_b: float) -> str:
    if total_b < 1:
        return "<1B"
    if total_b < 3:
        return "1B-3B"
    if total_b < 7:
        return "3B-7B"
    if total_b < 13:
        return "7B-13B"
    if total_b < 34:
        return "13B-34B"
    if total_b < 70:
        return "34B-70B"
    if total_b < 180:
        return "70B-180B"
    return "180B+"


def infer_role(row: pd.Series, family_id: str, series_id: str) -> Tuple[str, str]:
    """
    Role inference uses only repo/base/tags/pipeline, not config/card JSON.

    Priority:
      reasoning > instruct > chat > code > math > domain > base > seq2seq > unknown

    This fixes cases where config metadata caused Llama-Instruct to become code.
    """
    text = text_from_fields(row, ROLE_FIELDS)
    # The Hub tag `custom_code` (also `custom-code`) only says the repo ships its own modeling code; without this
    # line the code pattern below matched it and labeled 64% of all "code" repositories (bug found in the human
    # annotation of the taxonomy, 2026-11-17). See IGNORE_CUSTOM_CODE_TAG.
    if IGNORE_CUSTOM_CODE_TAG:
        text = re.sub(r"custom[_-]code", " ", text)
    repo_text = norm_lower(row.get("repo_id", ""))

    # Series-level override.
    if series_id in {"deepseek-r1", "qwq"}:
        return "reasoning", f"series:{series_id}"

    if has_pattern(text, REASONING_PATTERNS):
        return "reasoning", "role_pattern:reasoning"

    if has_pattern(text, INSTRUCT_PATTERNS):
        return "instruct", "role_pattern:instruct"

    if has_pattern(text, CHAT_PATTERNS):
        return "chat", "role_pattern:chat"

    # Code only from names/tags, and after instruct/chat.
    if has_pattern(text, CODE_PATTERNS):
        return "code", "role_pattern:code"

    if has_pattern(text, MATH_PATTERNS):
        return "math", "role_pattern:math"

    if has_pattern(text, DOMAIN_PATTERNS):
        return "domain", "role_pattern:domain"

    if has_pattern(text, BASE_PATTERNS):
        return "base", "role_pattern:base"

    pipeline = norm_lower(row.get("pipeline_tag", ""))
    if pipeline in {"translation", "summarization", "text2text-generation"}:
        return "seq2seq", f"pipeline:{pipeline}"

    # Official base models often omit "base" in the repo name.
    # Example: Qwen/Qwen3-4B, facebook/opt-125m.
    org, name = get_repo_parts(row.get("repo_id", ""))
    official, _ = is_likely_official_model(row, family_id)
    if official and not has_pattern(text, INSTRUCT_PATTERNS + CHAT_PATTERNS + REASONING_PATTERNS):
        return "base", "official_no_tuning_marker"

    return "unknown-role", "missing_role_signal"


def infer_training_stage(role: str, row: pd.Series) -> Tuple[str, str]:
    text = text_from_fields(row, ROLE_FIELDS)

    if role == "base":
        return "base", "role_base"

    if role in {"instruct", "chat"}:
        return "instruction_or_chat_tuned", f"role:{role}"

    if role == "reasoning":
        return "reasoning_tuned", "role_reasoning"

    if role in {"code", "math", "domain", "seq2seq"}:
        return "task_or_domain_specialized", f"role:{role}"

    if has_pattern(text, DERIVATIVE_PATTERNS):
        return "derived_or_finetuned", "derivative_pattern"

    return "unknown_training_stage", "missing_training_stage_signal"


def is_likely_official_model(row: pd.Series, family_id: str) -> Tuple[bool, str]:
    repo_id = norm_str(row.get("repo_id", ""))
    org, _ = get_repo_parts(repo_id)
    org_slug = slugify(org)

    official_orgs = OFFICIAL_ORGS_BY_FAMILY.get(family_id, set())

    if org_slug in official_orgs:
        return True, f"official_org:{org_slug}"

    if family_id != "unknown" and org_slug == family_id:
        return True, "org_matches_family"

    return False, "not_official_org"


def is_derivative_model(row: pd.Series, family_id: str) -> Tuple[bool, str]:
    text = text_from_fields(row, ROLE_FIELDS + ["author"])
    official, off_ev = is_likely_official_model(row, family_id)

    if has_pattern(text, DERIVATIVE_PATTERNS):
        return True, "derivative_pattern"

    base_models = parse_base_models(row.get("base_models", ""))
    if base_models and not official:
        return True, "non_official_with_declared_base_model"

    return False, off_ev


def is_multimodal_leak(row: pd.Series) -> Tuple[bool, str]:
    if not FILTER_MULTIMODAL_LEAKS_IN_LINEAGE:
        return False, ""

    text = text_from_fields(row, ["repo_id", "tags", "model_type", "architecture", "model_family_guess"])
    pipeline = norm_lower(row.get("pipeline_tag", ""))

    multimodal_pipelines = {
        "image-text-to-text",
        "image-to-text",
        "video-text-to-text",
        "audio-text-to-text",
        "visual-question-answering",
        "document-question-answering",
        "any-to-any",
        "visual-document-retrieval",
    }

    if pipeline in multimodal_pipelines:
        return True, f"pipeline:{pipeline}"

    if has_pattern(text, MULTIMODAL_LEAK_PATTERNS):
        return True, "multimodal_name_or_type_pattern"

    return False, ""


# =============================================================================
# ID CONSTRUCTION
# =============================================================================

def make_base_model_id(family_id: str, series_id: str, size_label: str) -> str:
    return f"base::{slugify(family_id)}::{slugify(series_id)}::{slugify(size_label)}"


def make_variant_id(base_model_id: str, role: str) -> str:
    return f"variant::{base_model_id.replace('base::', '')}::{slugify(role)}"


def make_model_entity_id(
    row: pd.Series,
    base_model_id: str,
    variant_id: str,
    family_id: str,
    role: str,
    official: bool,
    derivative: bool,
) -> Tuple[str, str, str]:
    repo_id = norm_str(row.get("repo_id", ""))
    repo_slug = slugify(repo_id)

    if official:
        return variant_id, "official_variant", "official_model_uses_variant_id"

    if derivative and PRESERVE_COMMUNITY_DERIVATIVES:
        return f"derived::{repo_slug}", "community_derivative", "preserved_derivative_repo"

    if PRESERVE_REUPLOADS_AS_ENTITIES:
        return f"mirror_or_reupload::{repo_slug}", "mirror_or_reupload", "preserved_non_official_repo"

    # Optional aggressive collapse, disabled by default.
    return variant_id, "collapsed_non_official_variant", "collapsed_to_variant_id"


def infer_parent_ref(
    row: pd.Series,
    base_model_id: str,
    variant_id: str,
    role: str,
) -> Tuple[str, str, float]:
    base_models = parse_base_models(row.get("base_models", ""))
    if base_models:
        return base_models[0], "declared_base_model", 1.0

    if not CREATE_INFERRED_PARENT_EDGES:
        return "", "", 0.0

    if role in {"instruct", "chat", "reasoning", "code", "math", "domain"}:
        return base_model_id, "inferred_same_base_model", 0.55

    return "", "", 0.0


# =============================================================================
# LINEAGE INFERENCE
# =============================================================================

def infer_lineage_for_row(row: pd.Series, nominal=None) -> Dict[str, Any]:
    family_id, series_id, family_ev = infer_family_series(row)
    size_label, size_ev = extract_size_label(row)
    size_label, size_ev = snap_parameter_size(row, family_id, series_id, size_label, size_ev, nominal)
    size_bucket = size_to_bucket(size_label)

    role, role_ev = infer_role(row, family_id, series_id)
    training_stage, training_ev = infer_training_stage(role, row)

    official, official_ev = is_likely_official_model(row, family_id)
    derivative, derivative_ev = is_derivative_model(row, family_id)

    multimodal_leak, multimodal_ev = is_multimodal_leak(row)

    base_model_id = make_base_model_id(family_id, series_id, size_label)
    variant_id = make_variant_id(base_model_id, role)

    model_entity_id, entity_type, entity_ev = make_model_entity_id(
        row=row,
        base_model_id=base_model_id,
        variant_id=variant_id,
        family_id=family_id,
        role=role,
        official=official,
        derivative=derivative,
    )

    parent_ref, parent_ev, parent_conf = infer_parent_ref(
        row=row,
        base_model_id=base_model_id,
        variant_id=variant_id,
        role=role,
    )

    adoption_proxy = compute_adoption_proxy(row)

    return {
        "family_id": family_id,
        "series_id": series_id,
        "base_model_id": base_model_id,
        "variant_id": variant_id,
        "model_entity_id": model_entity_id,
        "model_entity_type": entity_type,

        "size_label": size_label,
        "size_bucket": size_bucket,
        "model_role": role,
        "training_stage": training_stage,

        "is_likely_official_model": official,
        "is_derivative_model": derivative,
        "is_multimodal_leak": multimodal_leak,
        "multimodal_leak_evidence": multimodal_ev,

        "lineage_parent_ref": parent_ref,
        "lineage_parent_evidence": parent_ev,
        "lineage_parent_confidence": parent_conf,

        "lineage_family_evidence": family_ev,
        "lineage_size_evidence": size_ev,
        "lineage_role_evidence": role_ev,
        "lineage_training_stage_evidence": training_ev,
        "lineage_official_evidence": official_ev,
        "lineage_derivative_evidence": derivative_ev,
        "lineage_entity_evidence": entity_ev,

        "adoption_proxy_log_lineage": adoption_proxy,
    }


def build_lineage_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    edges = []
    nominal = collect_nominal_sizes(df)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inferring lineages", unit="model"):
        repo_id = norm_str(row.get("repo_id", ""))
        info = infer_lineage_for_row(row, nominal)

        rec = row.to_dict()
        rec.update(info)
        records.append(rec)

        # Skip edge construction if repo_id missing.
        if repo_id:
            edges.append({
                "source_id": repo_id,
                "source_type": "repo",
                "target_id": info["model_entity_id"],
                "target_type": "model_entity",
                "edge_type": "repo_belongs_to_model_entity",
                "confidence": 1.0,
                "evidence": info["lineage_entity_evidence"],
            })

        edges.append({
            "source_id": info["model_entity_id"],
            "source_type": "model_entity",
            "target_id": info["variant_id"],
            "target_type": "variant",
            "edge_type": "model_entity_instantiates_variant",
            "confidence": 1.0,
            "evidence": info["model_entity_type"],
        })

        edges.append({
            "source_id": info["variant_id"],
            "source_type": "variant",
            "target_id": info["base_model_id"],
            "target_type": "base_model",
            "edge_type": "variant_belongs_to_base_model",
            "confidence": 1.0,
            "evidence": info["lineage_role_evidence"],
        })

        edges.append({
            "source_id": info["base_model_id"],
            "source_type": "base_model",
            "target_id": info["series_id"],
            "target_type": "series",
            "edge_type": "base_model_belongs_to_series",
            "confidence": 1.0 if info["series_id"] != "unknown" else 0.2,
            "evidence": info["lineage_family_evidence"],
        })

        edges.append({
            "source_id": info["series_id"],
            "source_type": "series",
            "target_id": info["family_id"],
            "target_type": "family",
            "edge_type": "series_belongs_to_family",
            "confidence": 1.0 if info["family_id"] != "unknown" else 0.2,
            "evidence": info["lineage_family_evidence"],
        })

        base_models = parse_base_models(row.get("base_models", ""))
        for base in base_models:
            edges.append({
                "source_id": repo_id,
                "source_type": "repo",
                "target_id": base,
                "target_type": "declared_base_model_ref",
                "edge_type": "declares_base_model",
                "confidence": 1.0,
                "evidence": "base_models_metadata",
            })

        if info["lineage_parent_ref"]:
            edges.append({
                "source_id": info["model_entity_id"],
                "source_type": "model_entity",
                "target_id": info["lineage_parent_ref"],
                "target_type": "parent_model_ref",
                "edge_type": "has_parent_model",
                "confidence": info["lineage_parent_confidence"],
                "evidence": info["lineage_parent_evidence"],
            })

    models_df = pd.DataFrame(records)
    edges_df = pd.DataFrame(edges).drop_duplicates()

    return models_df, edges_df


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_table(
    df: pd.DataFrame,
    group_cols: List[str],
    entity_name: str,
) -> pd.DataFrame:
    rows = []

    grouped = df.groupby(group_cols, dropna=False)

    for keys, g in tqdm(grouped, desc=f"Aggregating {entity_name}", unit=entity_name):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))

        g2 = g.copy()
        g2["_downloads_30d"] = pd.to_numeric(g2.get("downloads_30d", 0), errors="coerce").fillna(0)
        g2["_downloads_all_time"] = pd.to_numeric(g2.get("downloads_all_time", 0), errors="coerce").fillna(0)
        g2["_likes"] = pd.to_numeric(g2.get("likes", 0), errors="coerce").fillna(0)
        g2["_adoption"] = pd.to_numeric(g2.get("adoption_proxy_log_lineage", 0), errors="coerce").fillna(0)

        top_idx = g2["_adoption"].idxmax()
        top = g2.loc[top_idx]

        row.update({
            "repo_count": int(len(g2)),
            "top_repo_id": norm_str(top.get("repo_id", "")),
            "top_author": norm_str(top.get("author", "")),
            "sum_downloads_30d": int(g2["_downloads_30d"].sum()),
            "sum_downloads_all_time": int(g2["_downloads_all_time"].sum()),
            "sum_likes": int(g2["_likes"].sum()),
            "max_adoption_proxy_log": float(g2["_adoption"].max()),
            "mean_adoption_proxy_log": float(g2["_adoption"].mean()),
            "official_repo_count": int(g2.get("is_likely_official_model", False).fillna(False).sum()),
            "derivative_repo_count": int(g2.get("is_derivative_model", False).fillna(False).sum()),
            "multimodal_leak_count": int(g2.get("is_multimodal_leak", False).fillna(False).sum()),
            "role_counts_json": json.dumps(
                g2.get("model_role", pd.Series(dtype=str)).fillna("unknown-role").value_counts().to_dict(),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "size_bucket_counts_json": json.dumps(
                g2.get("size_bucket", pd.Series(dtype=str)).fillna("unknown").value_counts().to_dict(),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "entity_type_counts_json": json.dumps(
                g2.get("model_entity_type", pd.Series(dtype=str)).fillna("unknown").value_counts().to_dict(),
                ensure_ascii=False,
                sort_keys=True,
            ),
        })

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(
            ["sum_downloads_30d", "sum_likes", "repo_count"],
            ascending=[False, False, False],
        )

    return out


def build_aggregates(models_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    model_entities = aggregate_table(
        models_df,
        [
            "model_entity_id",
            "model_entity_type",
            "variant_id",
            "base_model_id",
            "series_id",
            "family_id",
            "size_label",
            "size_bucket",
            "model_role",
            "training_stage",
        ],
        "model_entities",
    )

    variants = aggregate_table(
        models_df,
        [
            "variant_id",
            "base_model_id",
            "series_id",
            "family_id",
            "size_label",
            "size_bucket",
            "model_role",
            "training_stage",
        ],
        "variants",
    )

    base_models = aggregate_table(
        models_df,
        [
            "base_model_id",
            "series_id",
            "family_id",
            "size_label",
            "size_bucket",
        ],
        "base_models",
    )

    series = aggregate_table(
        models_df,
        [
            "series_id",
            "family_id",
        ],
        "series",
    )

    families = aggregate_table(
        models_df,
        [
            "family_id",
        ],
        "families",
    )

    return {
        "model_entities": model_entities,
        "variants": variants,
        "base_models": base_models,
        "series": series,
        "families": families,
    }


# =============================================================================
# SAVE / TYPES
# =============================================================================

def normalize_output_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bool_cols = {
        "is_likely_official_model",
        "is_derivative_model",
        "is_multimodal_leak",
    }

    int_cols = {
        "repo_count",
        "official_repo_count",
        "derivative_repo_count",
        "multimodal_leak_count",
        "sum_downloads_30d",
        "sum_downloads_all_time",
        "sum_likes",
        "downloads_30d",
        "downloads_all_time",
        "likes",
        "children_model_count",
        "parameter_count",
        "context_length",
        "parameter_count_clean",
        "downloads_30d_clean",
        "downloads_all_time_clean",
        "likes_clean",
        "children_model_count_clean",
        "llm_relevance_score",
        "core_relevance_score",
        "num_siblings",
        "has_config_json",
        "has_tokenizer",
        "has_safetensors",
        "has_pytorch_bin",
        "has_gguf",
        "num_spaces",
    }

    float_cols = {
        "adoption_proxy_log",
        "adoption_proxy_log_lineage",
        "max_adoption_proxy_log",
        "mean_adoption_proxy_log",
        "trending_score",
        "confidence",
        "lineage_parent_confidence",
    }

    for col in df.columns:
        if col in bool_cols:
            df[col] = df[col].fillna(False).astype(bool)
        elif col in int_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif col in float_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")
        else:
            df[col] = df[col].map(norm_str).astype("string")

    return df


def save_table(df: pd.DataFrame, parquet_path: Path, csv_path: Optional[Path] = None):
    df2 = normalize_output_types(df)
    df2.to_parquet(parquet_path, index=False)
    if csv_path is not None:
        df2.to_csv(csv_path, index=False)


# =============================================================================
# SUMMARY
# =============================================================================

def top_counts(df: pd.DataFrame, col: str, n: int = 40) -> List[Tuple[str, int]]:
    if col not in df.columns:
        return []
    vc = df[col].fillna("MISSING").replace("", "MISSING").value_counts().head(n)
    return [(str(k), int(v)) for k, v in vc.items()]


def write_summary(
    models_df: pd.DataFrame,
    aggregates: Dict[str, pd.DataFrame],
    edges_df: pd.DataFrame,
    summary_path: Path,
):
    lines = []

    model_entities = aggregates["model_entities"]
    variants = aggregates["variants"]
    base_models = aggregates["base_models"]
    series = aggregates["series"]
    families = aggregates["families"]

    lines.append("=" * 80)
    lines.append("OOME LINEAGE V2 SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Models with lineage: {len(models_df):,}")
    lines.append(f"Model entities: {len(model_entities):,}")
    lines.append(f"Variants: {len(variants):,}")
    lines.append(f"Base models: {len(base_models):,}")
    lines.append(f"Series: {len(series):,}")
    lines.append(f"Families: {len(families):,}")
    lines.append(f"Lineage edges: {len(edges_df):,}")

    if "is_multimodal_leak" in models_df.columns:
        leak_count = int(models_df["is_multimodal_leak"].fillna(False).sum())
        lines.append(f"Multimodal leaks detected in input: {leak_count:,}")

    lines.append("")
    lines.append("Top families by 30d downloads:")
    for _, r in families.head(30).iterrows():
        lines.append(
            f"  - {r['family_id']}: "
            f"repos={int(r['repo_count']):,}, "
            f"downloads_30d={int(r['sum_downloads_30d']):,}, "
            f"top={r['top_repo_id']}"
        )

    lines.append("")
    lines.append("Top variants by 30d downloads:")
    for _, r in variants.head(30).iterrows():
        lines.append(
            f"  - {r['variant_id']}: "
            f"repos={int(r['repo_count']):,}, "
            f"downloads_30d={int(r['sum_downloads_30d']):,}, "
            f"top={r['top_repo_id']}"
        )

    lines.append("")
    lines.append("Top model entities by 30d downloads:")
    for _, r in model_entities.head(30).iterrows():
        lines.append(
            f"  - {r['model_entity_id']}: "
            f"type={r['model_entity_type']}, "
            f"repos={int(r['repo_count']):,}, "
            f"downloads_30d={int(r['sum_downloads_30d']):,}, "
            f"top={r['top_repo_id']}"
        )

    lines.append("")
    lines.append("Family counts:")
    for k, v in top_counts(models_df, "family_id", 60):
        lines.append(f"  - {k}: {v:,}")

    lines.append("")
    lines.append("Series counts:")
    for k, v in top_counts(models_df, "series_id", 60):
        lines.append(f"  - {k}: {v:,}")

    lines.append("")
    lines.append("Role counts:")
    for k, v in top_counts(models_df, "model_role", 60):
        lines.append(f"  - {k}: {v:,}")

    lines.append("")
    lines.append("Training stage counts:")
    for k, v in top_counts(models_df, "training_stage", 60):
        lines.append(f"  - {k}: {v:,}")

    lines.append("")
    lines.append("Size bucket counts:")
    for k, v in top_counts(models_df, "size_bucket", 60):
        lines.append(f"  - {k}: {v:,}")

    lines.append("")
    lines.append("Model entity type counts:")
    for k, v in top_counts(models_df, "model_entity_type", 60):
        lines.append(f"  - {k}: {v:,}")

    lines.append("")
    lines.append("Family evidence counts:")
    for k, v in top_counts(models_df, "lineage_family_evidence", 40):
        lines.append(f"  - {k}: {v:,}")

    lines.append("")
    lines.append("Role evidence counts:")
    for k, v in top_counts(models_df, "lineage_role_evidence", 40):
        lines.append(f"  - {k}: {v:,}")

    lines.append("")
    lines.append("Size evidence counts:")
    for k, v in top_counts(models_df, "lineage_size_evidence", 40):
        lines.append(f"  - {k}: {v:,}")

    lines.append("")
    lines.append("Edge type counts:")
    for k, v in top_counts(edges_df, "edge_type", 40):
        lines.append(f"  - {k}: {v:,}")

    lines.append("")
    lines.append("Examples of unknown family:")
    unknown_fam = models_df[models_df["family_id"] == "unknown"].head(N_SUMMARY_EXAMPLES)
    for _, r in unknown_fam.iterrows():
        lines.append(f"  - {r.get('repo_id', '')}")

    lines.append("")
    lines.append("Examples of unknown role:")
    unknown_role = models_df[models_df["model_role"] == "unknown-role"].head(N_SUMMARY_EXAMPLES)
    for _, r in unknown_role.iterrows():
        lines.append(f"  - {r.get('repo_id', '')}")

    lines.append("")
    lines.append("Examples of unknown size:")
    unknown_size = models_df[models_df["size_bucket"] == "unknown"].head(N_SUMMARY_EXAMPLES)
    for _, r in unknown_size.iterrows():
        lines.append(f"  - {r.get('repo_id', '')}")

    text = "\n".join(lines)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)


# =============================================================================
# INPUT
# =============================================================================

def choose_input_file() -> Path:
    if INPUT_FILE.exists():
        return INPUT_FILE
    if FALLBACK_INPUT_FILE.exists():
        return FALLBACK_INPUT_FILE
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE} or fallback {FALLBACK_INPUT_FILE}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_file = choose_input_file()

    print("=" * 80)
    print("OOME LINEAGE CONSTRUCTION V2")
    print("=" * 80)
    print(f"Input: {input_file}")
    print(f"Output dir: {OUT_DIR}")
    print("=" * 80)

    print("Reading input...")
    if input_file.suffix == ".parquet":
        df = pd.read_parquet(input_file)
    elif input_file.suffix == ".csv":
        df = pd.read_csv(input_file, low_memory=False)
    else:
        raise ValueError(f"Unsupported input format: {input_file}")

    print(f"Rows loaded: {len(df):,}")

    needed_cols = sorted(set(
        FAMILY_FIELDS
        + ROLE_FIELDS
        + SIZE_FIELDS
        + [
            "repo_id",
            "author",
            "downloads_30d",
            "downloads_all_time",
            "likes",
            "children_model_count",
            "parameter_count",
            "parameter_count_clean",
            "adoption_proxy_log",
            "pipeline_tag",
            "model_family_guess",
            "model_type",
            "architecture",
            "base_models",
            "tags",
            "gated",
            "license",
            "languages",
        ]
    ))

    for col in needed_cols:
        if col not in df.columns:
            df[col] = ""

    print("Inferring lineage hierarchy...")
    models_df, edges_df = build_lineage_tables(df)

    # Optional diagnostic only: keep rows but mark leaks.
    # If you want to fully remove them from lineage outputs, uncomment below:
    #
    # models_df = models_df[models_df["is_multimodal_leak"] == False].copy()
    # edges_df = edges_df[
    #     ~edges_df["source_id"].isin(set(models_df.loc[models_df["is_multimodal_leak"], "repo_id"]))
    # ].copy()

    print("Building aggregate tables...")
    aggregates = build_aggregates(models_df)

    print("Saving outputs...")
    outputs = {
        "models_with_lineage": OUT_DIR / "oome_models_with_lineage.parquet",
        "model_entities": OUT_DIR / "oome_model_entities.parquet",
        "variants": OUT_DIR / "oome_variants.parquet",
        "base_models": OUT_DIR / "oome_base_models.parquet",
        "series": OUT_DIR / "oome_series.parquet",
        "families": OUT_DIR / "oome_families.parquet",
        "lineage_edges": OUT_DIR / "oome_lineage_edges.parquet",
    }

    csv_outputs = {}
    if SAVE_CSV and SAVE_LARGE_CSV:
        csv_outputs = {
            k: p.with_suffix(".csv")
            for k, p in outputs.items()
        }
    elif SAVE_CSV:
        csv_outputs = {
            "model_entities": outputs["model_entities"].with_suffix(".csv"),
            "variants": outputs["variants"].with_suffix(".csv"),
            "base_models": outputs["base_models"].with_suffix(".csv"),
            "series": outputs["series"].with_suffix(".csv"),
            "families": outputs["families"].with_suffix(".csv"),
            "lineage_edges": outputs["lineage_edges"].with_suffix(".csv"),
        }

    save_table(
        models_df,
        outputs["models_with_lineage"],
        csv_outputs.get("models_with_lineage"),
    )
    save_table(
        aggregates["model_entities"],
        outputs["model_entities"],
        csv_outputs.get("model_entities"),
    )
    save_table(
        aggregates["variants"],
        outputs["variants"],
        csv_outputs.get("variants"),
    )
    save_table(
        aggregates["base_models"],
        outputs["base_models"],
        csv_outputs.get("base_models"),
    )
    save_table(
        aggregates["series"],
        outputs["series"],
        csv_outputs.get("series"),
    )
    save_table(
        aggregates["families"],
        outputs["families"],
        csv_outputs.get("families"),
    )
    save_table(
        edges_df,
        outputs["lineage_edges"],
        csv_outputs.get("lineage_edges"),
    )

    summary_path = OUT_DIR / "oome_lineage_summary.txt"
    manifest_path = OUT_DIR / "oome_lineage_manifest.json"

    write_summary(
        models_df=models_df,
        aggregates=aggregates,
        edges_df=edges_df,
        summary_path=summary_path,
    )

    manifest = {
        "input_file": os.path.relpath(input_file),
        "output_dir": os.path.relpath(OUT_DIR),
        "config": {
            "preserve_community_derivatives": PRESERVE_COMMUNITY_DERIVATIVES,
            "preserve_reuploads_as_entities": PRESERVE_REUPLOADS_AS_ENTITIES,
            "create_inferred_parent_edges": CREATE_INFERRED_PARENT_EDGES,
            "filter_multimodal_leaks_in_lineage": FILTER_MULTIMODAL_LEAKS_IN_LINEAGE,
            "save_csv": SAVE_CSV,
            "save_large_csv": SAVE_LARGE_CSV,
            "ignore_custom_code_tag": IGNORE_CUSTOM_CODE_TAG,
        },
        "counts": {
            "input_rows": int(len(df)),
            "models_with_lineage": int(len(models_df)),
            "model_entities": int(len(aggregates["model_entities"])),
            "variants": int(len(aggregates["variants"])),
            "base_models": int(len(aggregates["base_models"])),
            "series": int(len(aggregates["series"])),
            "families": int(len(aggregates["families"])),
            "lineage_edges": int(len(edges_df)),
            "multimodal_leaks": int(models_df["is_multimodal_leak"].fillna(False).sum())
            if "is_multimodal_leak" in models_df.columns else 0,
        },
        "outputs": {k: os.path.relpath(v) for k, v in outputs.items()},
        "csv_outputs": {k: os.path.relpath(v) for k, v in csv_outputs.items()},
        "top_counts": {
            "family_id": dict(top_counts(models_df, "family_id", 100)),
            "series_id": dict(top_counts(models_df, "series_id", 100)),
            "model_role": dict(top_counts(models_df, "model_role", 100)),
            "size_bucket": dict(top_counts(models_df, "size_bucket", 100)),
            "model_entity_type": dict(top_counts(models_df, "model_entity_type", 100)),
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 80)
    print("FINALIZADO")
    print("=" * 80)
    print(f"Models with lineage: {outputs['models_with_lineage']}")
    print(f"Model entities: {outputs['model_entities']}")
    print(f"Variants: {outputs['variants']}")
    print(f"Base models: {outputs['base_models']}")
    print(f"Series: {outputs['series']}")
    print(f"Families: {outputs['families']}")
    print(f"Edges: {outputs['lineage_edges']}")
    print(f"Summary: {summary_path}")
    print(f"Manifest: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()