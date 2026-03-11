from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd


TECH_INDUSTRY_PATTERN = re.compile(
    r"(?i)\b("
    r"software|"
    r"information technology|"
    r"technology,\s*information\s*and\s*(internet|media)|"
    r"it services|"
    r"computer|"
    r"data infrastructure|"
    r"cybersecurity|"
    r"telecommunications|"
    r"semiconductor|"
    r"internet"
    r")\b"
)

CORE_DIGITAL_TITLE_PATTERN = re.compile(
    r"(?ix)\b("
    r"software\s+engineer|software\s+developer|application\s+developer|developer|"
    r"frontend\s+developer|front\s*end\s+developer|"
    r"backend\s+developer|back\s*end\s+developer|"
    r"full\s*stack\s+(engineer|developer)|"
    r"web\s+developer|mobile\s+application\s+developer|"
    r"ios\s+developer|android\s+developer|"
    r"java\s+developer|python\s+developer|dotnet\s+developer|salesforce\s+developer|"
    r"data\s+(scientist|engineer|analyst|architect|modeler)|"
    r"machine\s+learning\s+engineer|ml\s+engineer|ai\s+engineer|computer\s+scientist|"
    r"cloud\s+(engineer|architect)|"
    r"devops\s+engineer|site\s+reliability\s+engineer|\bsre\b|platform\s+engineer|"
    r"network\s+engineer|systems?\s+administrator|database\s+administrator|"
    r"security\s+(engineer|analyst)|cybersecurity|information\s+security|"
    r"embedded\s+software\s+engineer|firmware\s+engineer|"
    r"qa\s+engineer|quality\s+assurance\s+(engineer|specialist)|"
    r"software\s+engineer\s+in\s+test|test\s+engineer|automation\s+engineer|"
    r"solutions\s+architect|technical\s+product\s+manager"
    r")\b"
)

TECH_LEADERSHIP_TITLE_PATTERN = re.compile(
    r"(?ix)\b("
    r"software\s+engineering\s+manager|engineering\s+manager|"
    r"data\s+engineering\s+manager|platform\s+engineering\s+manager|"
    r"information\s+security\s+manager|security\s+manager|"
    r"it\s+manager|technology\s+manager|"
    r"(director|head)\s+(of\s+)?(engineering|technology|data|security|it|software|platform|cloud|devops|product)|"
    r"(engineering|technology|data|security|it|software|platform|cloud|devops|product)\s+(director|head)|"
    r"(vice\s+president|vp|chief)\s+(of\s+)?(engineering|technology|data|security|it|software|platform|cloud|devops|product)"
    r")\b"
)

AMBIGUOUS_TECH_TITLE_PATTERN = re.compile(
    r"(?ix)\b("
    r"product\s+manager|project\s+manager|program\s+manager|"
    r"architect|technical\s+lead|lead\s+developer|lead\s+engineer"
    r")\b"
)

TECH_DESCRIPTION_PATTERN = re.compile(
    r"(?ix)\b("
    r"software|developer|programming|codebase|api|"
    r"cloud|data platform|data pipeline|machine learning|artificial intelligence|"
    r"network security|cybersecurity|information security|"
    r"devops|site reliability|database|etl|analytics engineering|"
    r"java|python|javascript|c\+\+|c\#|sql|aws|azure|gcp|react|kubernetes"
    r")\b"
)

NON_DIGITAL_ENGINEERING_PATTERN = re.compile(
    r"(?ix)\b("
    r"project\s+architect|building\s+engineer|"
    r"mechanical|civil|structural|electrical|industrial|manufacturing|"
    r"quality\s+engineer|process\s+engineer|construction|"
    r"field\s+service\s+technician|service\s+technician"
    r")\b"
)

OFF_TARGET_ROLE_PATTERN = re.compile(
    r"(?ix)\b("
    r"data\s+entry|entry\s+clerk|"
    r"sales\s+(associate|manager|supervisor|development)|"
    r"account\s+(manager|executive)|"
    r"customer\s+service|success\s+manager|"
    r"executive\s+assistant|administrative\s+assistant|marketing\s+assistant|"
    r"business\s+development|technical\s+writer|retail"
    r")\b"
)


@dataclass(frozen=True)
class ScopeDecision:
    include: bool
    reason: str
    exclusion_reason: str


def normalize(text: Optional[str]) -> str:
    return str(text or "").strip().lower()


def is_tech_industry(industry_name: Optional[str]) -> bool:
    return bool(TECH_INDUSTRY_PATTERN.search(normalize(industry_name)))


def classify_scope(
    title: Optional[str],
    description: Optional[str],
    is_tech_industry_flag: bool = False,
) -> ScopeDecision:
    title_s = normalize(title)
    description_s = normalize(description)

    if not title_s:
        return ScopeDecision(False, "missing_title", "missing_title")

    if NON_DIGITAL_ENGINEERING_PATTERN.search(title_s):
        return ScopeDecision(False, "excluded", "non_digital_engineering")

    if OFF_TARGET_ROLE_PATTERN.search(title_s):
        return ScopeDecision(False, "excluded", "off_target_business_role")

    if CORE_DIGITAL_TITLE_PATTERN.search(title_s):
        return ScopeDecision(True, "title_core", "")

    if TECH_LEADERSHIP_TITLE_PATTERN.search(title_s):
        return ScopeDecision(True, "title_leadership", "")

    if AMBIGUOUS_TECH_TITLE_PATTERN.search(title_s):
        if is_tech_industry_flag or TECH_DESCRIPTION_PATTERN.search(description_s):
            return ScopeDecision(True, "ambiguous_title_supported", "")
        return ScopeDecision(False, "excluded", "ambiguous_title_without_tech_context")

    return ScopeDecision(False, "excluded", "no_tech_scope_signal")


def build_industry_lookup(
    job_industries: pd.DataFrame,
    industries: pd.DataFrame,
) -> pd.DataFrame:
    merged = job_industries.merge(industries, on="industry_id", how="left")
    merged["is_tech_industry"] = merged["industry_name"].apply(is_tech_industry)
    summary = (
        merged.groupby("job_id")
        .agg(
            is_tech_industry=("is_tech_industry", "max"),
            industry_name=("industry_name", lambda s: " | ".join(sorted({str(v) for v in s.dropna()})[:3])),
        )
        .reset_index()
    )
    return summary


def apply_scope_filter(
    postings: pd.DataFrame,
    industry_lookup: pd.DataFrame,
) -> pd.DataFrame:
    merged = postings.merge(industry_lookup, on="job_id", how="left")
    merged["is_tech_industry"] = merged["is_tech_industry"].fillna(False)
    decisions = [
        classify_scope(title, description, bool(is_tech_industry))
        for title, description, is_tech_industry in zip(
            merged["title"],
            merged["description"],
            merged["is_tech_industry"],
        )
    ]
    merged["scope_include"] = [decision.include for decision in decisions]
    merged["scope_reason"] = [decision.reason for decision in decisions]
    merged["scope_exclusion_reason"] = [decision.exclusion_reason for decision in decisions]
    merged["is_broad_tech_company"] = merged["is_tech_industry"].astype(bool)
    return merged


def build_scope_summary(scope_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "rows_total",
                "rows_in_scope",
                "rows_out_of_scope",
                "broad_tech_company_rows",
            ],
            "value": [
                len(scope_df),
                int(scope_df["scope_include"].sum()),
                int((~scope_df["scope_include"]).sum()),
                int(scope_df["is_broad_tech_company"].sum()),
            ],
        }
    )


def build_scope_audit_samples(
    scope_df: pd.DataFrame,
    n_per_bucket: int = 25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    include_cols = [
        "job_id",
        "title",
        "company_name",
        "formatted_experience_level",
        "industry_name",
        "scope_reason",
    ]
    exclude_cols = include_cols + ["scope_exclusion_reason"]
    included = (
        scope_df[scope_df["scope_include"]]
        .sample(min(n_per_bucket, int(scope_df["scope_include"].sum())), random_state=seed)
        .loc[:, [c for c in include_cols if c in scope_df.columns]]
        .copy()
    )
    excluded = (
        scope_df[~scope_df["scope_include"]]
        .sample(min(n_per_bucket, int((~scope_df["scope_include"]).sum())), random_state=seed)
        .loc[:, [c for c in exclude_cols if c in scope_df.columns]]
        .copy()
    )
    included["manual_precision_note"] = ""
    excluded["manual_precision_note"] = ""
    return included, excluded
