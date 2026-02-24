"""
src/seniority.py

Rule-based seniority classifier for job postings.

We assign each posting one of:
  - entry
  - mid
  - senior
  - leadership

Approach:
- Search for seniority keywords inside a  clean_text field.
- Use priority ordering so that higher-authority matches override lower ones.
  Example: "senior director" should be leadership, not senior.

"""

from __future__ import annotations

from typing import Iterable
import re

# -------------------------
# Keyword patterns (regex)
# -------------------------
# Use regex with word boundaries to reduce accidental matches.
# Example: "assistant" should not match "assistantship" unless intended.

LEADERSHIP_PATTERN = re.compile(
    r"\b("
    r"chief|ceo|cto|cfo|coo|"
    r"vp|vice president|"
    r"director|executive director|managing director|"
    r"head of|president|general manager"
    r")\b"
)

# NOTE: We intentionally do NOT include "manager" here because
# many "X manager" roles (e.g., account manager, project manager)
# are often non-executive. We'll treat "manager" as senior by default.
SENIOR_PATTERN = re.compile(
    r"\b("
    r"senior|sr\.?|"
    r"lead|team lead|tech lead|"
    r"principal|staff|architect|"
    r"manager|supervisor"
    r")\b"
)

ENTRY_PATTERN = re.compile(
    r"\b("
    r"intern|internship|"
    r"junior|jr\.?|"
    r"entry[-\s]?level|"
    r"associate|trainee|apprentice|"
    r"graduate|fresher|assistant"
    r")\b"
)


def classify_seniority(clean_text: str) -> str:
    """
    Classify a posting into one of: leadership, senior, entry, mid.

    Parameters
    ----------
    clean_text : str
        Pre-cleaned text (ideally lowercased, punctuation removed).
        If not lowercased, this function still works (regex is not case-sensitive here),
        but you should keep preprocessing consistent.

    Returns
    -------
    str
        One of: "leadership", "senior", "entry", "mid"

    Logic
    -----
    Priority order matters (highest to lowest authority):
      1) leadership
      2) senior
      3) entry
      4) mid (default)
    """
    text = str(clean_text).lower()

    if LEADERSHIP_PATTERN.search(text):
        return "leadership"
    if SENIOR_PATTERN.search(text):
        return "senior"
    if ENTRY_PATTERN.search(text):
        return "entry"
    return "mid"