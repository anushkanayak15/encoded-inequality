from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import (
    GROUP_ORDER,
    LABELING_RESULTS_DIR,
    PROCESSED_DATA_DIR,
    ROBUSTNESS_RESULTS_DIR,
    SCOPE_RESULTS_DIR,
)
from .io import ensure_dir, load_industries, load_job_industries, load_postings, write_csv
from .labeling import apply_seniority_labels, build_override_review
from .robustness import lexicon_bias_summary, subset_labeled_only
from .scope import (
    apply_scope_filter,
    build_industry_lookup,
    build_scope_audit_samples,
    build_scope_summary,
)
from .text import prepare_text_corpus

SCOPE_COLUMNS = [
    "job_id",
    "company_id",
    "company_name",
    "title",
    "description",
    "skills_desc",
    "formatted_experience_level",
    "formatted_work_type",
    "location",
    "remote_allowed",
    "pay_period",
    "min_salary",
    "med_salary",
    "max_salary",
    "posting_domain",
    "listed_time",
    "original_listed_time",
]


def build_scope_outputs(
    processed_dir: Path = PROCESSED_DATA_DIR,
    results_dir: Path = SCOPE_RESULTS_DIR,
    nrows: int | None = None,
) -> dict[str, Path]:
    ensure_dir(processed_dir)
    ensure_dir(results_dir)

    postings = load_postings(usecols=SCOPE_COLUMNS, nrows=nrows)
    industry_lookup = build_industry_lookup(load_job_industries(), load_industries())
    scope_df = apply_scope_filter(postings, industry_lookup)
    core_df = scope_df.loc[scope_df["scope_include"]].copy()
    broad_df = scope_df.loc[scope_df["is_broad_tech_company"]].copy()
    summary = build_scope_summary(scope_df)
    included_audit, excluded_audit = build_scope_audit_samples(scope_df)

    all_scope_path = write_csv(scope_df, processed_dir / "linkedin_scope_all.csv")
    core_path = write_csv(core_df, processed_dir / "linkedin_digital_tech_core.csv")
    broad_path = write_csv(broad_df, processed_dir / "linkedin_broad_tech_company.csv")
    summary_path = write_csv(summary, results_dir / "scope_summary.csv")
    included_path = write_csv(included_audit, results_dir / "included_title_audit_sample.csv")
    excluded_path = write_csv(excluded_audit, results_dir / "excluded_title_audit_sample.csv")

    return {
        "scope_all": all_scope_path,
        "core": core_path,
        "broad": broad_path,
        "summary": summary_path,
        "included_sample": included_path,
        "excluded_sample": excluded_path,
    }


def export_group_corpora(
    labeled_df: pd.DataFrame,
    processed_dir: Path,
    prefix: str,
    suffix: str,
    keep_for_model_only: bool,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    working = labeled_df.loc[labeled_df["keep_for_model"]].copy() if keep_for_model_only else labeled_df.copy()
    for group in GROUP_ORDER:
        subset = working.loc[working["seniority_final"].eq(group)].copy()
        filename = f"{prefix}_{group}_{suffix}.csv"
        paths[group] = write_csv(subset, processed_dir / filename)
    return paths


def build_labeling_outputs(
    processed_dir: Path = PROCESSED_DATA_DIR,
    results_dir: Path = LABELING_RESULTS_DIR,
    input_path: Path | None = None,
    prefix: str = "linkedin_digital_tech",
) -> dict[str, Path]:
    ensure_dir(processed_dir)
    ensure_dir(results_dir)

    source_path = input_path or processed_dir / f"{prefix}_core.csv"
    core_df = pd.read_csv(source_path)
    labeled_df = prepare_text_corpus(apply_seniority_labels(core_df))
    override_review = build_override_review(labeled_df)
    labeling_summary = (
        labeled_df.groupby(["seniority_final", "seniority_source"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )
    duplicate_summary = pd.DataFrame(
        [
            {"metric": "rows_total", "value": len(labeled_df)},
            {"metric": "rows_kept_for_model", "value": int(labeled_df["keep_for_model"].sum())},
            {"metric": "exact_duplicates", "value": int(labeled_df["is_exact_duplicate"].sum())},
            {"metric": "template_duplicates", "value": int(labeled_df["is_template_duplicate"].sum())},
        ]
    )

    labeled_path = write_csv(labeled_df, processed_dir / f"{prefix}_labeled.csv")
    labeled_only_path = write_csv(
        subset_labeled_only(labeled_df),
        processed_dir / f"{prefix}_labeled_only.csv",
    )
    override_path = write_csv(override_review, results_dir / f"{prefix}_seniority_override_review.csv")
    summary_path = write_csv(labeling_summary, results_dir / f"{prefix}_seniority_summary.csv")
    duplicate_path = write_csv(duplicate_summary, results_dir / f"{prefix}_duplicate_summary.csv")

    clean_paths = export_group_corpora(
        labeled_df,
        processed_dir,
        prefix=prefix,
        suffix="clean",
        keep_for_model_only=True,
    )
    nondedup_paths = export_group_corpora(
        labeled_df,
        processed_dir,
        prefix=prefix,
        suffix="nondedup",
        keep_for_model_only=False,
    )

    return {
        "labeled": labeled_path,
        "labeled_only": labeled_only_path,
        "override_review": override_path,
        "seniority_summary": summary_path,
        "duplicate_summary": duplicate_path,
        **{f"clean_{group}": path for group, path in clean_paths.items()},
        **{f"nondedup_{group}": path for group, path in nondedup_paths.items()},
    }


def build_robustness_outputs(
    processed_dir: Path = PROCESSED_DATA_DIR,
    results_dir: Path = ROBUSTNESS_RESULTS_DIR,
    prefix: str = "linkedin_digital_tech",
) -> dict[str, Path]:
    ensure_dir(results_dir)
    labeled_df = pd.read_csv(processed_dir / f"{prefix}_labeled.csv")
    labeled_only_df = subset_labeled_only(labeled_df)
    deduped_df = labeled_df.loc[labeled_df["keep_for_model"]].copy()
    broad_df = pd.read_csv(processed_dir / "linkedin_broad_tech_company.csv")

    labeled_plus_backfilled_path = write_csv(
        lexicon_bias_summary(deduped_df),
        results_dir / f"{prefix}_lexicon_bias_labeled_plus_backfilled.csv",
    )
    labeled_only_path = write_csv(
        lexicon_bias_summary(labeled_only_df.loc[labeled_only_df["keep_for_model"]].copy()),
        results_dir / f"{prefix}_lexicon_bias_labeled_only.csv",
    )
    nondedup_path = write_csv(
        lexicon_bias_summary(labeled_df),
        results_dir / f"{prefix}_lexicon_bias_nondedup.csv",
    )
    broad_summary = pd.DataFrame(
        [
            {"metric": "broad_tech_company_rows", "value": len(broad_df)},
            {"metric": "digital_tech_rows", "value": len(labeled_df)},
        ]
    )
    broad_path = write_csv(broad_summary, results_dir / f"{prefix}_broader_tech_company_summary.csv")

    return {
        "lexicon_bias_labeled_plus_backfilled": labeled_plus_backfilled_path,
        "lexicon_bias_labeled_only": labeled_only_path,
        "lexicon_bias_nondedup": nondedup_path,
        "broad_tech_company_summary": broad_path,
    }
