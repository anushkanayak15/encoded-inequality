from __future__ import annotations
import re
from typing import Optional, Tuple


# -------------------------
# Regex patterns
# -------------------------

# Strong executive indicators.
# These represent true organizational leadership roles.
LEADERSHIP_STRONG = re.compile(
    r"\b("
    r"chief\s+.*officer|"
    r"cto|cfo|coo|ciso|"
    r"vice\s+president|vp\b|vp\s+of|"
    r"executive\s+director|"
    r"head\s+of|"
    r"president|general\s+manager|"
    r"founder|cofounder|owner|partner"
    r")\b"
)

# Director alone is ambiguous; we confirm leadership with experience.
DIRECTOR_PATTERN = re.compile(r"\bdirector\b")

# Clear early-career markers.
ENTRY_STRONG = re.compile(
    r"\b("
    r"intern|internship|"
    r"junior|jr\.?|"
    r"entry[-\s]*level|"
    r"new\s*grad|graduate|"
    r"trainee|apprentice"
    r")\b"
)

# Explicit senior markers.
SENIOR_STRONG = re.compile(
    r"\b("
    r"senior|sr\.?|"
    r"principal|staff|"
    r"architect|"
    r"lead\s+engineer|lead\s+developer|team\s+lead|tech\s+lead|"
    r"consultant|expert"
    r")\b"
)

# Manager-level roles are typically mid-to-senior.
MANAGER_PATTERN = re.compile(r"\b(manager|supervisor)\b")

# Terms that reduce authority when appearing with leadership words.
# Example: "Assistant Director" should NOT be leadership.
LOWERING_TERMS = re.compile(
    r"\b(assistant|associate|junior|jr\.?|trainee)\b"
)

# Leadership responsibility signals (scope-based confirmation).
# Leadership requires organizational authority, not just title.
LEADERSHIP_RESP = re.compile(
    r"\b("
    r"manage\s+teams?|"
    r"lead\s+teams?|"
    r"oversee\s+operations?|"
    r"strategic\s+planning|"
    r"budget\s+management|"
    r"drive\s+strategy|"
    r"organizational\s+leadership"
    r")\b"
)

# Senior-level responsibility signals.
SENIOR_RESP = re.compile(
    r"\b("
    r"mentor|"
    r"own\s+projects?|"
    r"architecture\s+design|"
    r"technical\s+leadership|"
    r"complex\s+systems?|"
    r"cross[-\s]?functional"
    r")\b"
)

# Broad title filter for Data Science / Research / Analytics subset.
DSRA_TITLE_PATTERN = re.compile(
    r"\b("
    r"data|"
    r"analytics?|"
    r"analyst|analysis|"
    r"research|scientist|"
    r"machine\s+learning|ml|ai|"
    r"business\s+intelligence|bi|"
    r"statistic|quant"
    r")\b"
)

# Curated include/exclude filters for tech roles.
TECH_TITLE_INCLUDE_PATTERN = re.compile(
    r"\b("
    r"software|"
    r"dev|developer|programmer|"
    r"data|"
    r"machine\s+learning|ml|ai|artificial\s+intelligence|"
    r"cloud|devops|sre|site\s+reliability|"
    r"security|cyber|"
    r"network|database|"
    r"backend|back[-\s]*end|"
    r"frontend|front[-\s]*end|"
    r"full[-\s]*stack|"
    r"qa|quality\s+assurance|test\w*|"
    r"it|information\s+technology|systems?\s+administrator|"
    r"ui|ux|web|mobile|ios|android|"
    r"product\s+manager|technical\s+product\s+manager|"
    r"business\s+intelligence|bi|"
    r"platform|automation|architect"
    r")\b"
)

TECH_TITLE_EXCLUDE_PATTERN = re.compile(
    r"\b("
    r"marketing|sales|"
    r"finance|financial|investment|"
    r"account|accountant|"
    r"legal|attorney|"
    r"nurse|physician|therapist|dental|veterinarian|"
    r"teacher|"
    r"procurement|supply\s+chain|"
    r"customer\s+service|"
    r"hr|human\s+resources|"
    r"event|wedding|"
    r"landscape\s+architect|"
    r"civil\s+engineer|mechanical\s+engineer"
    r")\b"
)

ARCHITECT_TITLE_PATTERN = re.compile(r"\barchitect\b")

EXP_RANGE = re.compile(r"(\d+)\s*(?:to|-)\s*(\d+)\s*years?", re.IGNORECASE)
EXP_SINGLE = re.compile(r"(\d+)\s*years?", re.IGNORECASE)


DEGREE_MAP = {
    "phd": "doctorate",
    "doctorate": "doctorate",
    "mba": "masters",
    "m.tech": "masters",
    "mtech": "masters",
    "mca": "masters",
    "m.com": "masters",
    "mcom": "masters",
    "msc": "masters",
    "ms": "masters",
}


def normalize(s: Optional[str]) -> str:
    return str(s or "").lower().strip()


def parse_experience(exp: Optional[str]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    exp_s = normalize(exp)
    m = EXP_RANGE.search(exp_s)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2))
        return lo, hi, hi - lo
    m = EXP_SINGLE.search(exp_s)
    if m:
        yrs = int(m.group(1))
        return yrs, yrs, 0
    return None, None, None


def degree_level(q: Optional[str]) -> Optional[str]:
    q_s = normalize(q).replace(" ", "")
    return DEGREE_MAP.get(q_s)


def is_dsra_title(title: Optional[str]) -> bool:
    return bool(DSRA_TITLE_PATTERN.search(normalize(title)))


def is_tech_title(title: Optional[str]) -> bool:
    t = normalize(title)
    return bool(TECH_TITLE_INCLUDE_PATTERN.search(t)) and not bool(
        TECH_TITLE_EXCLUDE_PATTERN.search(t)
    )


def classify_seniority_hybrid(
    title: Optional[str],
    clean_text: Optional[str],
    experience: Optional[str] = None,
    qualifications: Optional[str] = None,
) -> str:
    # qualifications is accepted for API compatibility with classify_seniority
    _ = qualifications

    t = normalize(title)
    ct = normalize(clean_text)
    lo, _, _ = parse_experience(experience)

    # 1) Entry precedence
    if ENTRY_STRONG.search(t) or (lo is not None and lo <= 1):
        return "entry"

    # 2) Leadership
    leadership_signal = (
        LEADERSHIP_STRONG.search(t)
        or DIRECTOR_PATTERN.search(t)
        or LEADERSHIP_RESP.search(ct)
        or MANAGER_PATTERN.search(ct)
        or ARCHITECT_TITLE_PATTERN.search(t)
    )
    if leadership_signal and lo is not None and lo >= 2:
        return "leadership"

    # 3) Senior
    senior_signal = (
        SENIOR_STRONG.search(t)
        or SENIOR_RESP.search(ct)
        or SENIOR_STRONG.search(ct)
        or (lo is not None and lo >= 5)
    )
    if senior_signal:
        return "senior"

    # 4) Mid fallback
    return "mid"


def classify_seniority(
    title: Optional[str],
    clean_text: Optional[str],
    experience: Optional[str] = None,
    qualifications: Optional[str] = None,
) -> str:

    t = normalize(title)
    ct = normalize(clean_text)

    lo, hi, _ = parse_experience(experience)

    # -------------------------------------------------------
    # HARD OVERRIDES (High precision rules)
    # -------------------------------------------------------

    # True executive roles.
    # We require:
    # 1) Strong leadership title
    # 2) No lowering term (assistant/associate)
    if LEADERSHIP_STRONG.search(t) and not LOWERING_TERMS.search(t):
        return "leadership"

    # Director promoted to leadership only if 8+ years
    if DIRECTOR_PATTERN.search(t) and lo is not None and lo >= 8:
        return "leadership"

    # Clear entry titles.
    if ENTRY_STRONG.search(t):
        return "entry"

    # Experience-based entry.
    if hi is not None and hi <= 1:
        return "entry"

    # Explicit demotion case (e.g., Assistant Manager).
    if LOWERING_TERMS.search(t) and MANAGER_PATTERN.search(t):
        return "mid"

    score = 0.0

    # -------------------------------------------------------
    # TITLE SIGNALS (Primary authority source)
    # -------------------------------------------------------

    if SENIOR_STRONG.search(t):
        score += 4

    if MANAGER_PATTERN.search(t):
        score += 2

    # -------------------------------------------------------
    # RESPONSIBILITY SIGNALS (Scope & autonomy)
    # -------------------------------------------------------

    if SENIOR_RESP.search(ct):
        score += 1

    if SENIOR_STRONG.search(ct):
        score += 1.5

    if MANAGER_PATTERN.search(ct):
        score += 0.75

    if ENTRY_STRONG.search(ct):
        score -= 2

    # -------------------------------------------------------
    # EXPERIENCE SIGNAL (Minimum years matters most)
    # -------------------------------------------------------

    if lo is not None:
        if lo >= 8:
            score += 3
        elif lo >= 5:
            score += 2.5
        elif lo >= 3:
            score += 1

    # -------------------------------------------------------
    # DEGREE SIGNAL (Weak modifier)
    # -------------------------------------------------------

    deg = degree_level(qualifications)

    if deg == "doctorate":
        score += 1
    elif deg == "masters":
        score += 0.5

    # -------------------------------------------------------
    # FINAL DECISION
    # -------------------------------------------------------

    if score >= 2.5:
        return "senior"

    if score <= -2:
        return "entry"

    return "mid"
