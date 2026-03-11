from __future__ import annotations

import hashlib
import re
from typing import Optional

import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NON_WORD_PATTERN = re.compile(r"[^a-z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")

TOKEN_REPLACEMENTS = {
    "c++": "cplusplus",
    "c#": "csharp",
    ".net": "dotnet",
    "node.js": "nodejs",
    "react.js": "reactjs",
}


def normalize_text(text: Optional[str]) -> str:
    value = str(text or "").lower()
    for src, dest in TOKEN_REPLACEMENTS.items():
        value = value.replace(src, dest)
    value = URL_PATTERN.sub(" ", value)
    value = value.replace("&", " and ")
    value = NON_WORD_PATTERN.sub(" ", value)
    value = WHITESPACE_PATTERN.sub(" ", value).strip()
    return value


def build_model_text(
    title: Optional[str],
    description: Optional[str],
    skills_desc: Optional[str] = None,
) -> str:
    combined = " ".join(str(part or "") for part in [title, description, skills_desc] if part)
    normalized = normalize_text(combined)
    tokens = TOKEN_PATTERN.findall(normalized)
    return " ".join(tokens)


def normalized_description(text: Optional[str], strip_numbers: bool = False) -> str:
    value = normalize_text(text)
    if strip_numbers:
        value = re.sub(r"\b\d+\b", " ", value)
        value = WHITESPACE_PATTERN.sub(" ", value).strip()
    return value


def digest_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def add_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["text_raw"] = prepared.apply(
        lambda row: " ".join(
            part
            for part in [
                str(row.get("title") or ""),
                str(row.get("description") or ""),
                str(row.get("skills_desc") or ""),
            ]
            if part
        ).strip(),
        axis=1,
    )
    prepared["model_text"] = prepared.apply(
        lambda row: build_model_text(
            row.get("title"),
            row.get("description"),
            row.get("skills_desc"),
        ),
        axis=1,
    )
    prepared["token_count"] = prepared["model_text"].str.split().str.len()
    prepared["description_exact_norm"] = prepared["description"].apply(normalized_description)
    prepared["description_template_norm"] = prepared["description"].apply(
        lambda value: normalized_description(value, strip_numbers=True)
    )
    prepared["title_exact_norm"] = prepared["title"].apply(normalize_text)
    prepared["title_description_fingerprint"] = prepared.apply(
        lambda row: digest_text(
            f"{row['title_exact_norm']}||{row['description_exact_norm']}"
        ),
        axis=1,
    )
    prepared["description_template_fingerprint"] = prepared["description_template_norm"].apply(digest_text)
    return prepared


def add_duplicate_flags(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["is_exact_duplicate"] = prepared["title_description_fingerprint"].duplicated(keep="first")
    prepared["is_template_duplicate"] = prepared["description_template_fingerprint"].duplicated(keep="first")
    prepared["duplicate_group_id"] = prepared["description_template_fingerprint"].where(
        prepared["description_template_fingerprint"].map(
            prepared["description_template_fingerprint"].value_counts()
        ) > 1,
        "",
    )
    prepared["keep_for_model"] = ~(
        prepared["is_exact_duplicate"] | prepared["is_template_duplicate"]
    )
    return prepared


def prepare_text_corpus(df: pd.DataFrame) -> pd.DataFrame:
    return add_duplicate_flags(add_text_columns(df))
