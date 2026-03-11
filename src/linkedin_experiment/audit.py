from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .constants import AUDIT_RESULTS_DIR
from .io import (
    ensure_dir,
    load_companies,
    load_industries,
    load_job_industries,
    load_postings,
    write_csv,
)

NOISY_ENTRY_TITLE_PATTERN = re.compile(r"(?ix)\b(intern|junior|jr\.?|entry|graduate|new\s*grad)\b")
NOISY_LEADERSHIP_TITLE_PATTERN = re.compile(r"(?ix)\b(chief|vice\s+president|\bvp\b|director|head\s+of)\b")
NOISY_SENIOR_TITLE_PATTERN = re.compile(r"(?ix)\b(senior|sr\.?|principal|staff|lead)\b")


def _sample_by_level(df: pd.DataFrame, n_per_level: int = 5, seed: int = 42) -> pd.DataFrame:
    samples = []
    level_series = df["formatted_experience_level"].fillna("MISSING")
    for level in level_series.unique():
        subset = df.loc[level_series.eq(level)].copy()
        if subset.empty:
            continue
        sample = subset.sample(min(n_per_level, len(subset)), random_state=seed)
        sample["experience_bucket"] = level
        samples.append(sample)
    return pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()


def _noisy_examples(df: pd.DataFrame, n_per_bucket: int = 10) -> pd.DataFrame:
    noisy_masks = {
        "entry_title_not_entry": df["title"].fillna("").str.contains(NOISY_ENTRY_TITLE_PATTERN)
        & ~df["formatted_experience_level"].isin(["Entry level", "Internship"]),
        "leadership_title_not_leadership": df["title"].fillna("").str.contains(NOISY_LEADERSHIP_TITLE_PATTERN)
        & ~df["formatted_experience_level"].isin(["Director", "Executive"]),
        "senior_title_not_seniorish": df["title"].fillna("").str.contains(NOISY_SENIOR_TITLE_PATTERN)
        & ~df["formatted_experience_level"].isin(["Mid-Senior level", "Director", "Executive"]),
    }
    frames = []
    for bucket, mask in noisy_masks.items():
        subset = df.loc[mask].head(n_per_bucket).copy()
        subset["noise_bucket"] = bucket
        frames.append(subset)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_dataset_audit(output_dir: Path = AUDIT_RESULTS_DIR) -> dict[str, Path]:
    ensure_dir(output_dir)

    postings = load_postings()
    companies = load_companies(usecols=["company_id"])
    job_industries = load_job_industries()
    industries = load_industries()

    summary_rows = [
        {"metric": "rows", "value": len(postings)},
        {"metric": "columns", "value": len(postings.columns)},
        {"metric": "duplicate_job_id_count", "value": int(postings["job_id"].duplicated().sum())},
        {
            "metric": "description_empty_count",
            "value": int(postings["description"].fillna("").str.strip().eq("").sum()),
        },
    ]

    date_coverage = []
    for column in ["original_listed_time", "listed_time", "expiry", "closed_time"]:
        parsed = pd.to_datetime(postings[column], unit="ms", errors="coerce")
        date_coverage.append(
            {
                "column": column,
                "non_null": int(parsed.notna().sum()),
                "min": parsed.min(),
                "max": parsed.max(),
            }
        )

    text_lengths = postings["description"].fillna("").str.len()
    text_length_summary = pd.DataFrame(
        {
            "quantile": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
            "description_length": text_lengths.quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).values,
        }
    )

    null_rates = (
        postings.isna()
        .mean()
        .mul(100)
        .rename("null_pct")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column"})
    )

    industry_map = job_industries.merge(industries, on="industry_id", how="left")
    join_coverage = pd.DataFrame(
        [
            {
                "join": "companies",
                "matched_rows": int(postings["company_id"].isin(companies["company_id"]).sum()),
                "total_rows": len(postings),
            },
            {
                "join": "job_industries",
                "matched_rows": int(postings["job_id"].isin(industry_map["job_id"]).sum()),
                "total_rows": len(postings),
            },
        ]
    )

    examples = _sample_by_level(
        postings[
            [
                "job_id",
                "title",
                "company_name",
                "formatted_experience_level",
                "formatted_work_type",
                "location",
                "description",
            ]
        ]
    )
    noisy = _noisy_examples(
        postings[
            [
                "job_id",
                "title",
                "company_name",
                "formatted_experience_level",
                "description",
            ]
        ]
    )

    summary_path = write_csv(pd.DataFrame(summary_rows), output_dir / "audit_summary.csv")
    nulls_path = write_csv(null_rates, output_dir / "null_rates.csv")
    dates_path = write_csv(pd.DataFrame(date_coverage), output_dir / "date_coverage.csv")
    text_path = write_csv(text_length_summary, output_dir / "text_length_summary.csv")
    joins_path = write_csv(join_coverage, output_dir / "join_coverage.csv")
    examples_path = write_csv(examples, output_dir / "experience_examples.csv")
    noisy_path = write_csv(noisy, output_dir / "noisy_label_examples.csv")

    top_nulls = null_rates.head(12)
    plt.figure(figsize=(10, 4))
    plt.bar(top_nulls["column"], top_nulls["null_pct"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Null %")
    plt.title("Top Missing Columns In postings.csv")
    plt.tight_layout()
    null_plot_path = output_dir / "null_rates_top12.png"
    plt.savefig(null_plot_path, dpi=160)
    plt.close()

    exp_counts = (
        postings["formatted_experience_level"]
        .fillna("MISSING")
        .value_counts()
        .reset_index()
    )
    exp_counts.columns = ["level", "count"]
    plt.figure(figsize=(8, 4))
    plt.bar(exp_counts["level"], exp_counts["count"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Postings")
    plt.title("LinkedIn Experience Levels")
    plt.tight_layout()
    exp_plot_path = output_dir / "experience_level_distribution.png"
    plt.savefig(exp_plot_path, dpi=160)
    plt.close()

    return {
        "summary": summary_path,
        "null_rates": nulls_path,
        "date_coverage": dates_path,
        "text_length_summary": text_path,
        "join_coverage": joins_path,
        "examples": examples_path,
        "noisy_examples": noisy_path,
        "null_rate_plot": null_plot_path,
        "experience_plot": exp_plot_path,
    }
