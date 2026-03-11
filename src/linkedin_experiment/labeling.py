from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd


LINKEDIN_TO_SENIORITY = {
    "Internship": "entry",
    "Entry level": "entry",
    "Associate": "mid",
    "Mid-Senior level": "senior",
    "Director": "leadership",
    "Executive": "leadership",
}

ENTRY_TITLE_PATTERN = re.compile(
    r"(?ix)\b("
    r"intern|internship|junior|jr\.?|entry|new\s*grad|graduate|apprentice|trainee"
    r")\b"
)

LEADERSHIP_TITLE_PATTERN = re.compile(
    r"(?ix)\b("
    r"chief|vice\s+president|\bvp\b|director|head\s+of|"
    r"engineering\s+manager|software\s+engineering\s+manager|"
    r"data\s+engineering\s+manager|product\s+manager|project\s+manager|program\s+manager|"
    r"it\s+manager|technology\s+manager|security\s+manager|information\s+security\s+manager"
    r")\b"
)

SENIOR_TITLE_PATTERN = re.compile(
    r"(?ix)\b("
    r"senior|sr\.?|principal|staff|lead"
    r")\b"
)


@dataclass(frozen=True)
class SeniorityDecision:
    seniority_final: str
    seniority_source: str
    seniority_rule: str


def normalize(text: Optional[str]) -> str:
    return str(text or "").strip().lower()


def collapse_linkedin_level(raw_level: Optional[str]) -> Optional[str]:
    return LINKEDIN_TO_SENIORITY.get(str(raw_level or "").strip())


def assign_seniority(title: Optional[str], raw_level: Optional[str]) -> SeniorityDecision:
    title_s = normalize(title)
    collapsed = collapse_linkedin_level(raw_level)

    if ENTRY_TITLE_PATTERN.search(title_s):
        source = "title_override" if collapsed and collapsed != "entry" else "title_backfill"
        return SeniorityDecision("entry", source, "entry_title_signal")

    if LEADERSHIP_TITLE_PATTERN.search(title_s):
        source = "title_override" if collapsed and collapsed != "leadership" else "title_backfill"
        return SeniorityDecision("leadership", source, "leadership_title_signal")

    if SENIOR_TITLE_PATTERN.search(title_s):
        source = "title_override" if collapsed and collapsed != "senior" else "title_backfill"
        return SeniorityDecision("senior", source, "senior_title_signal")

    if collapsed:
        return SeniorityDecision(collapsed, "linkedin_primary", "linkedin_experience_level")

    return SeniorityDecision("mid", "title_backfill", "default_mid_backfill")


def apply_seniority_labels(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.copy()
    decisions = [
        assign_seniority(title, raw_level)
        for title, raw_level in zip(
            labeled["title"],
            labeled["formatted_experience_level"],
        )
    ]
    labeled["raw_experience_level"] = labeled["formatted_experience_level"]
    labeled["seniority_final"] = [decision.seniority_final for decision in decisions]
    labeled["seniority_source"] = [decision.seniority_source for decision in decisions]
    labeled["seniority_rule"] = [decision.seniority_rule for decision in decisions]
    return labeled


def build_override_review(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["seniority_source"].eq("title_override")
    columns = [
        "job_id",
        "title",
        "raw_experience_level",
        "seniority_final",
        "seniority_rule",
        "company_name",
    ]
    return df.loc[mask, [c for c in columns if c in df.columns]].copy()
