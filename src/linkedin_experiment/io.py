from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .constants import (
    COMPANIES_PATH,
    INDUSTRIES_PATH,
    JOB_INDUSTRIES_PATH,
    POSTINGS_PATH,
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        ensure_dir(path)


def load_postings(
    usecols: Sequence[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    return pd.read_csv(POSTINGS_PATH, usecols=usecols, nrows=nrows)


def load_job_industries(nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(JOB_INDUSTRIES_PATH, nrows=nrows)


def load_industries(nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(INDUSTRIES_PATH, nrows=nrows)


def load_companies(
    usecols: Sequence[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    return pd.read_csv(COMPANIES_PATH, usecols=usecols, nrows=nrows)


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
    return path
