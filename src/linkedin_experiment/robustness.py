from __future__ import annotations

import pandas as pd

from .lexicons import FEMININE_CODED_WORDS, MASCULINE_CODED_WORDS


def subset_labeled_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["seniority_source"].eq("linkedin_primary")].copy()


def subset_labeled_plus_backfilled(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def subset_non_deduped(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def subset_deduped(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["keep_for_model"]].copy()


def tokenize(text: str) -> list[str]:
    return [token for token in str(text or "").split() if token]


def lexicon_bias_summary(
    df: pd.DataFrame,
    text_col: str = "model_text",
    group_col: str = "seniority_final",
) -> pd.DataFrame:
    scored = df.copy()
    scored["_tokens"] = scored[text_col].fillna("").apply(tokenize)
    scored["masculine_count"] = scored["_tokens"].apply(
        lambda tokens: sum(token in MASCULINE_CODED_WORDS for token in tokens)
    )
    scored["feminine_count"] = scored["_tokens"].apply(
        lambda tokens: sum(token in FEMININE_CODED_WORDS for token in tokens)
    )
    scored["token_total"] = scored["_tokens"].apply(len)
    summary = (
        scored.groupby(group_col, as_index=False)[["masculine_count", "feminine_count", "token_total"]]
        .sum()
        .rename(columns={group_col: "group"})
    )
    summary["masculine_per_1000"] = summary["masculine_count"] / summary["token_total"] * 1000
    summary["feminine_per_1000"] = summary["feminine_count"] / summary["token_total"] * 1000
    summary["bias_score"] = summary["masculine_per_1000"] - summary["feminine_per_1000"]
    return summary
