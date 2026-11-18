#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model_resolver.py

Auditable resolution of model names to population entities.

It takes the model names annotated in papers and resolves them to canonical identifiers of the Hugging Face
population, without relying only on hand-written aliases.
"""

from __future__ import annotations

import re
import difflib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

FUZZY_ACCEPT_SCORE = 0.86
FUZZY_ACCEPT_MARGIN = 0.06
FUZZY_REVIEW_SCORE = 0.78
MAX_TOKEN_CANDIDATES = 400
MAX_ALTERNATIVES_TO_LOG = 8

MANUAL_ALIAS_MAP_RAW: Dict[str, str] = {
    # DeepSeek
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "DeepSeek/DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "DeepSeek-R1-0528": "deepseek-ai/DeepSeek-R1-0528",
    "DeepSeek/DeepSeek-R1-0528": "deepseek-ai/DeepSeek-R1-0528",
    "DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    "DeepSeek/DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    "DeepSeek-V3-0324": "deepseek-ai/DeepSeek-V3-0324",
    "DeepSeek/DeepSeek-V3-0324": "deepseek-ai/DeepSeek-V3-0324",
    "DeepSeek-chat-v3": "deepseek-ai/DeepSeek-V3",
    "DeepSeek-chat-v3-0324": "deepseek-ai/DeepSeek-V3-0324",
    "DeepSeek-V3.1": "deepseek-ai/DeepSeek-V3.1",
    "DeepSeek/DeepSeek-V3.1": "deepseek-ai/DeepSeek-V3.1",

    # Llama 4
    "Llama-4-Maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "Meta/Llama-4-Maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "meta-llama/Llama-4-Maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "Llama-4-Maverick-Instruct": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "Llama-4-Maverick-17B-128E-Instruct": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "Llama-4-Scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "Llama-4-Scout-17B-16E-Instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct",

    # Kimi / Moonshot
    "Kimi-K2": "moonshotai/Kimi-K2",
    "Moonshot/Kimi-K2": "moonshotai/Kimi-K2",
    "moonshot/Kimi-K2": "moonshotai/Kimi-K2",

    # OpenAI open-weight GPT-OSS
    "GPT-OSS-20B": "openai/gpt-oss-20b",
    "GPT OSS 20B": "openai/gpt-oss-20b",
    "OpenAI/GPT-OSS-20B": "openai/gpt-oss-20b",
    "OpenAI/gpt-oss-20b": "openai/gpt-oss-20b",
    "GPT-OSS-120B": "openai/gpt-oss-120b",
    "GPT OSS 120B": "openai/gpt-oss-120b",
    "OpenAI/GPT-OSS-120B": "openai/gpt-oss-120b",
    "OpenAI/gpt-oss-120b": "openai/gpt-oss-120b",

    # Qwen / QwQ
    "QwQ-32B": "Qwen/QwQ-32B",
    "QwQ-32B-Preview": "Qwen/QwQ-32B-Preview",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen3-4B-Thinking-2507": "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen3-235B": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen3-235B-A22B-Instruct-2507": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen2.5-72B": "Qwen/Qwen2.5-72B",
    "Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
    "Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct",
    "Qwen2.5-14B-Instruct": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",

    # Llama / Meta normalization
    "Llama-3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
    "Llama-3.1-70B": "meta-llama/Meta-Llama-3.1-70B",
    "Llama-3.1-405B": "meta-llama/Meta-Llama-3.1-405B",
    "Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "Llama-3.1-405B-Instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "Meta/Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Meta/Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "Meta/Llama-3.1-405B-Instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "Meta/Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Meta/Meta-Llama-3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "Meta/Meta-Llama-3.1-405B-Instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "Meta/Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "Meta/Meta-Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",

    # Skywork o1 open model; evita confundir com OpenAI o1.
    "Skywork-o1-Open-Llama-3.1-8B": "Skywork/Skywork-o1-Open-Llama-3.1-8B",
    "Skywork/Skywork-o1-Open-Llama-3.1-8B": "Skywork/Skywork-o1-Open-Llama-3.1-8B",

    # Gemma
    "Gemma-3-4B-IT": "google/gemma-3-4b-it",
    "Gemma-3-12B-IT": "google/gemma-3-12b-it",
    "Gemma-3-27B-IT": "google/gemma-3-27b-it",
    "Google/MedGemma-4B-IT": "google/medgemma-4b-it",
    "MedGemma-4B-IT": "google/medgemma-4b-it",

    # Multimodal/audio names that often appear without provider.
    "LLaVA-v1.6-Vicuna-13B": "llava-hf/llava-v1.6-vicuna-13b-hf",
    "Whisper-V3": "openai/whisper-large-v3",
    "Qwen2-Audio-7B": "Qwen/Qwen2-Audio-7B",
    "Qwen2-Audio-7B-Instruct": "Qwen/Qwen2-Audio-7B-Instruct",
}

OPEN_WEIGHT_ALLOWLIST_RAW = {
    "moonshotai/Kimi-K2", "moonshot/Kimi-K2",
    "openai/gpt-oss-20b", "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-R1-0528",
    "deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3-0324", "deepseek-ai/DeepSeek-V3.1",
    "Skywork/Skywork-o1-Open-Llama-3.1-8B",
    "zhipu/GLM-4.5", "zhipuai/GLM-4.5", "THUDM/GLM-Z1-32B",
}

GENERIC_TOKENS = {
    "model", "models", "llm", "llms", "hf", "api", "latest", "preview",
    "chat", "instruct", "instruction", "base", "it", "sft", "dpo",
    "open", "source", "weight", "weights", "turbo", "pro", "mini",
    "large", "small", "medium", "high", "low", "exp", "experimental",
    "v", "the", "as", "judge",
}

FAMILY_ALIASES = {
    "llama": {"llama", "llama2", "llama3", "llama4", "meta-llama", "metallama"},
    "qwen": {"qwen", "qwq", "codeqwen"},
    "deepseek": {"deepseek", "deepseekai"},
    "gemma": {"gemma", "medgemma"},
    "mistral": {"mistral", "mixtral", "ministral", "codestral", "mathstral"},
    "phi": {"phi"}, "yi": {"yi"}, "baichuan": {"baichuan"},
    "chatglm": {"chatglm", "glm", "zhipu", "thudm"},
    "internlm": {"internlm"}, "vicuna": {"vicuna", "lmsys"},
    "tulu": {"tulu"}, "starcoder": {"starcoder", "starcoder2"},
    "codellama": {"codellama", "code-llama"}, "granite": {"granite"},
    "jamba": {"jamba"}, "kimi": {"kimi", "moonshot", "moonshotai"},
    "gpt": {"gpt", "openai", "o1", "o3", "o4", "o5"},
    "claude": {"claude", "anthropic"}, "gemini": {"gemini"}, "grok": {"grok", "xai"},
}

ROLE_TOKENS = {
    "instruct", "chat", "base", "dpo", "sft", "coder", "code", "math",
    "reward", "guard", "distill", "reasoner", "thinking", "vl", "audio",
    "omni", "vision", "it", "rl", "preview", "long", "prover",
}

SIZE_RE = re.compile(r"(?<![a-z0-9])(?:\d+(?:\.\d+)?b|(?:\d+)x(?:\d+(?:\.\d+)?)b|a\d+b|e\d+|moe)(?![a-z0-9])", re.I)
VERSION_RE = re.compile(r"(?<![a-z0-9])(?:v?\d+(?:\.\d+){0,2}|20\d{2}[-_]?\d{2}[-_]?\d{2}|20\d{6}|[0-9]{4})(?![a-z0-9])", re.I)


def normalize_text(s: Any) -> str:
    x = "" if s is None else str(s)
    x = x.strip()
    x = re.sub(r"\[[^\]]+\]", "", x)
    x = x.replace("–", "-").replace("—", "-").replace("−", "-")
    x = x.replace("_", "-")
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def normalize_key(s: Any) -> str:
    x = normalize_text(s)
    x = x.replace(" ", "-")
    x = re.sub(r"-+", "-", x)
    return x.lower().strip("-")


def basename(s: str) -> str:
    k = normalize_key(s)
    return k.split("/", 1)[1] if "/" in k else k


def provider_of(s: str) -> str:
    k = normalize_key(s)
    return k.split("/", 1)[0] if "/" in k else ""


def tokenize(s: Any) -> List[str]:
    k = normalize_key(s)
    return [p for p in re.split(r"[/\-_.+:\[\](), ]+", k) if p]


def informative_tokens(s: Any) -> Set[str]:
    out = set()
    for t in tokenize(s):
        if len(t) < 2 or t in GENERIC_TOKENS:
            continue
        out.add(t)
    return out


def find_family(s: Any) -> str:
    toks = set(tokenize(s))
    compact = normalize_key(s).replace("-", "").replace("/", "")
    for fam, aliases in FAMILY_ALIASES.items():
        if toks & aliases:
            return fam
        if any(alias.replace("-", "") in compact for alias in aliases if len(alias) >= 4):
            return fam
    return ""


def extract_sizes(s: Any) -> Set[str]:
    return {m.group(0).lower() for m in SIZE_RE.finditer(normalize_key(s))}


def extract_versions(s: Any) -> Set[str]:
    vals = {m.group(0).lower().replace("_", "-") for m in VERSION_RE.finditer(normalize_key(s))}
    return {v for v in vals if not v.endswith("b")}


def extract_roles(s: Any) -> Set[str]:
    return set(tokenize(s)) & ROLE_TOKENS


def strip_provider(s: str) -> str:
    return basename(s)


def strip_common_suffixes(k: str) -> str:
    suffixes = ["-hf", "-it", "-chat", "-instruct", "-base", "-sft", "-dpo", "-preview", "-latest", "-turbo", "-beta", "-alpha"]
    out = k
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if out.endswith(suf):
                out = out[: -len(suf)]
                changed = True
    return out.strip("-")


def prefix_before_size(k: str) -> str:
    parts = tokenize(k)
    keep = []
    for p in parts:
        if SIZE_RE.fullmatch(p):
            break
        keep.append(p)
    if len(keep) >= 2:
        return "-".join(keep)
    return ""


def remove_date_suffix(k: str) -> str:
    return re.sub(r"[-_/](20\d{6}|20\d{2}[-_]?\d{2}[-_]?\d{2}|\d{4})$", "", k).strip("-_/")


def generate_aliases_for_repo(repo: str, extra_values: Iterable[str] = ()) -> Set[str]:
    aliases: Set[str] = set()
    values = [repo, strip_provider(repo), *[str(v) for v in extra_values if str(v).strip()]]
    for val in values:
        k = normalize_key(val)
        if not k:
            continue
        aliases.add(k)
        aliases.add(k.replace("-", " "))
        aliases.add(strip_provider(k))
        aliases.add(strip_common_suffixes(k))
        aliases.add(strip_common_suffixes(strip_provider(k)))
        no_date = remove_date_suffix(k)
        aliases.add(no_date)
        aliases.add(strip_provider(no_date))
        pfx = prefix_before_size(k)
        if pfx:
            aliases.add(pfx)
            if "/" in k:
                aliases.add(provider_of(k) + "/" + pfx)
        aliases.add(k.replace("meta/", "meta-llama/"))
        aliases.add(k.replace("deepseek/", "deepseek-ai/"))
        aliases.add(k.replace("mistral/", "mistralai/"))
    return {a for a in aliases if a and len(a) >= 3}


@dataclass
class ResolutionResult:
    input_model: str
    resolved_model: str
    status: str
    match_type: str
    confidence: float
    reason: str
    candidates: str = ""

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelResolver:
    def __init__(self, population_parquet: Optional[Path] = None, manual_alias_map: Optional[Dict[str, str]] = None, use_population: bool = True, use_fuzzy: bool = True):
        self.population_parquet = population_parquet
        self.use_population = use_population
        self.use_fuzzy = use_fuzzy
        self.manual_alias_map = {normalize_key(k): v for k, v in (manual_alias_map or MANUAL_ALIAS_MAP_RAW).items()}
        self.repo_by_exact_key: Dict[str, str] = {}
        self.alias_to_repos: Dict[str, Set[str]] = {}
        self.token_to_repos: Dict[str, Set[str]] = {}
        self.repo_features: Dict[str, Dict[str, Any]] = {}
        self.loaded_population = False
        if self.use_population and population_parquet is not None and Path(population_parquet).exists():
            self.load_population(Path(population_parquet))

    def load_population(self, path: Path) -> None:
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"WARNING: could not load population parquet {path}: {e}")
            return
        if "repo_id" not in df.columns:
            print(f"WARNING: population parquet {path} has no repo_id column.")
            return
        possible_cols = ["repo_id", "model_id", "model_name", "model_entity_id", "variant_id", "base_model_id", "series_id", "family_id", "pipeline_tag", "tags", "architectures", "library_name"]
        cols = [c for c in possible_cols if c in df.columns]
        small = df[cols].drop_duplicates()
        for _, row in small.iterrows():
            repo = normalize_text(row.get("repo_id", ""))
            if not repo:
                continue
            extra_values = []
            for c in cols:
                if c == "repo_id":
                    continue
                val = row.get(c, "")
                if isinstance(val, list):
                    extra_values.extend([str(v) for v in val])
                else:
                    val_s = normalize_text(val)
                    if val_s and val_s.lower() != "nan":
                        extra_values.append(val_s)
            self._add_repo(repo, extra_values)
        self.loaded_population = True
        print(f"Loaded ModelResolver population from {path}")
        print(f"  repos: {len(self.repo_features):,}")
        print(f"  exact keys: {len(self.repo_by_exact_key):,}")
        print(f"  alias keys: {len(self.alias_to_repos):,}")
        print(f"  token keys: {len(self.token_to_repos):,}")

    def _add_repo(self, repo: str, extra_values: Iterable[str]) -> None:
        repo_key = normalize_key(repo)
        self.repo_by_exact_key.setdefault(repo_key, repo)
        self.repo_by_exact_key.setdefault(basename(repo), repo)
        for a in generate_aliases_for_repo(repo, extra_values):
            self.alias_to_repos.setdefault(normalize_key(a), set()).add(repo)
        toks = informative_tokens(repo)
        for val in extra_values:
            toks |= informative_tokens(val)
        for t in toks:
            self.token_to_repos.setdefault(t, set()).add(repo)
        self.repo_features[repo] = self._features(repo)

    def _features(self, s: str) -> Dict[str, Any]:
        return {
            "key": normalize_key(s), "basename": basename(s), "provider": provider_of(s),
            "tokens": informative_tokens(s), "family": find_family(s), "sizes": extract_sizes(s),
            "versions": extract_versions(s), "roles": extract_roles(s),
        }

    def resolve(self, model: Any) -> ResolutionResult:
        original = normalize_text(model)
        if not original:
            return ResolutionResult("", "", "empty", "empty", 0.0, "empty input")
        key = normalize_key(original)
        if key in self.manual_alias_map:
            target = self.manual_alias_map[key]
            return ResolutionResult(original, target, "matched_manual_alias", "manual_alias", 1.0, "manual alias")
        if self.loaded_population:
            if key in self.repo_by_exact_key:
                target = self.repo_by_exact_key[key]
                return ResolutionResult(original, target, "matched_population_exact", "population_exact", 1.0, "exact key in population")
            repos = self.alias_to_repos.get(key, set())
            if len(repos) == 1:
                target = next(iter(repos))
                return ResolutionResult(original, target, "matched_population_alias", "population_generated_alias", 0.96, "unique generated alias in population")
            if len(repos) > 1:
                return ResolutionResult(original, original, "ambiguous_population_alias", "population_generated_alias_ambiguous", 0.0, "generated alias maps to multiple repos", "; ".join(sorted(repos)[:MAX_ALTERNATIVES_TO_LOG]))
            if self.use_fuzzy:
                return self._resolve_fuzzy(original)
        return ResolutionResult(original, original, "unresolved", "unchanged", 0.0, "no matching rule fired")

    def _candidate_repos_for_input(self, original: str) -> Set[str]:
        counts: Dict[str, int] = {}
        for t in informative_tokens(original):
            for repo in self.token_to_repos.get(t, set()):
                counts[repo] = counts.get(repo, 0) + 1
        if not counts:
            return set()
        return {r for r, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:MAX_TOKEN_CANDIDATES]}

    def _resolve_fuzzy(self, original: str) -> ResolutionResult:
        candidates = self._candidate_repos_for_input(original)
        if not candidates:
            return ResolutionResult(original, original, "unresolved", "fuzzy_no_candidates", 0.0, "no token candidates in population")
        scored: List[Tuple[float, str, str]] = []
        for repo in candidates:
            score, reason = self._score_candidate(original, repo)
            scored.append((score, repo, reason))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_score, top_repo, top_reason = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -1.0
        margin = top_score - second_score
        alts = "; ".join([f"{repo} ({score:.3f})" for score, repo, _ in scored[:MAX_ALTERNATIVES_TO_LOG]])
        if top_score >= FUZZY_ACCEPT_SCORE and margin >= FUZZY_ACCEPT_MARGIN:
            return ResolutionResult(original, top_repo, "matched_fuzzy_high_confidence", "fuzzy_structural", float(top_score), f"{top_reason}; margin={margin:.3f}", alts)
        if top_score >= FUZZY_REVIEW_SCORE:
            return ResolutionResult(original, original, "ambiguous_or_low_margin_fuzzy", "fuzzy_structural_review", float(top_score), f"top candidate {top_repo}; {top_reason}; margin={margin:.3f}", alts)
        return ResolutionResult(original, original, "unresolved", "fuzzy_below_threshold", float(top_score), f"top candidate {top_repo}; {top_reason}; margin={margin:.3f}", alts)

    def _score_candidate(self, original: str, repo: str) -> Tuple[float, str]:
        f_in = self._features(original)
        f_repo = self.repo_features.get(repo) or self._features(repo)
        toks_in = f_in["tokens"]
        toks_repo = f_repo["tokens"]
        jaccard = len(toks_in & toks_repo) / max(1, len(toks_in | toks_repo))
        overlap_frac = len(toks_in & toks_repo) / max(1, len(toks_in))
        seq = difflib.SequenceMatcher(None, f_in["basename"], f_repo["basename"]).ratio()
        score = 0.45 * overlap_frac + 0.20 * jaccard + 0.15 * seq
        reasons = [f"overlap={overlap_frac:.3f}", f"jaccard={jaccard:.3f}", f"seq={seq:.3f}"]
        if f_in["family"] and f_repo["family"]:
            if f_in["family"] == f_repo["family"]:
                score += 0.12
                reasons.append("family_match")
            else:
                score -= 0.25
                reasons.append(f"family_conflict:{f_in['family']}!={f_repo['family']}")
        if f_in["provider"]:
            if f_in["provider"] == f_repo["provider"]:
                score += 0.06
                reasons.append("provider_match")
            else:
                score -= 0.04
                reasons.append(f"provider_diff:{f_in['provider']}!={f_repo['provider']}")
        if f_in["sizes"]:
            if f_in["sizes"] & f_repo["sizes"]:
                score += 0.12
                reasons.append("size_match")
            elif f_repo["sizes"]:
                score -= 0.18
                reasons.append(f"size_conflict:{sorted(f_in['sizes'])}!={sorted(f_repo['sizes'])}")
        if f_in["roles"]:
            if f_in["roles"] & f_repo["roles"]:
                score += 0.06
                reasons.append("role_overlap")
            elif any(r in f_in["roles"] for r in {"instruct", "chat", "base"}):
                score -= 0.04
                reasons.append("role_missing_or_diff")
        if f_in["versions"] and f_repo["versions"] and (f_in["versions"] & f_repo["versions"]):
            score += 0.04
            reasons.append("version_overlap")
        return max(0.0, min(1.0, score)), ",".join(reasons)


def build_default_resolver(population_parquet: Optional[Path], use_population: bool = True, use_fuzzy: bool = True) -> ModelResolver:
    return ModelResolver(population_parquet=population_parquet, manual_alias_map=MANUAL_ALIAS_MAP_RAW, use_population=use_population, use_fuzzy=use_fuzzy)


def is_open_weight_allowlisted(model: Any) -> bool:
    k = normalize_key(model)
    return k in {normalize_key(x) for x in OPEN_WEIGHT_ALLOWLIST_RAW}
