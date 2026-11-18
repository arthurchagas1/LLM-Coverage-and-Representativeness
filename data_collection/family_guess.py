"""Token-bounded family guess shared by the collector, cleaning, and lineage steps.

The June collector matched family names as plain substrings of the repository id
and all its tags. Language tags therefore assigned family "yi" (the ISO code for
Yiddish) to multilingual models such as bigscience/mt0-xxl and google/umt5-xxl,
and any name containing "phi" or "t5" as letters matched those families too.

Here a family must appear as its own token, digits may follow it (llama3, qwen2),
and tags that describe languages or provenance are ignored.
"""

from __future__ import annotations

import re
from typing import Iterable, List

FAMILIES = [
    "llama", "qwen", "mistral", "mixtral", "gemma", "deepseek", "phi",
    "olmo", "falcon", "starcoder", "codellama", "baichuan", "yi",
    "bloom", "gpt2", "gpt-neo", "gpt-j", "mpt", "stablelm",
    "command-r", "openchat", "vicuna", "alpaca", "zephyr", "solar",
    "internlm", "minicpm", "glm", "aquila", "xglm", "t5", "flan-t5",
    "rwkv", "mamba",
]
_FAMILY_RE = [(f, re.compile(rf"(?<![a-z0-9]){re.escape(f)}(?![a-z])")) for f in FAMILIES]

_IGNORED_TAG_PREFIXES = ("language:", "dataset:", "license:", "arxiv:", "doi:", "region:", "deploy:")


def _split(value: object) -> List[str]:
    if value is None:
        return []
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "<na>"}:
        return []
    return [part.strip() for part in re.split(r"[|,]", text) if part.strip()]


def family_relevant_tags(tags: object, languages: object) -> List[str]:
    """Drop tags that name a language or a provenance record rather than a model."""
    language_codes = set(_split(languages))
    return [
        tag for tag in _split(tags)
        if tag not in language_codes and not tag.startswith(_IGNORED_TAG_PREFIXES)
    ]


def _first_family(text: str, families: Iterable = _FAMILY_RE) -> str:
    for family, pattern in families:
        if pattern.search(text):
            return family
    return ""


def corrected_family_guess(repo_id: object, tags: object, languages: object, model_type: object) -> str:
    """Family named by the repository id or a relevant tag, else the config model type."""
    text = " ".join([str(repo_id or "").lower(), *family_relevant_tags(tags, languages)])
    family = _first_family(text)
    if family:
        return family
    fallback = str(model_type or "").strip().lower()
    return "" if fallback in {"nan", "none", "<na>"} else fallback
