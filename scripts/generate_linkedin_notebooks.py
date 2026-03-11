from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def markdown_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def code_cell(code: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().splitlines()],
    }


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, cells: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")


COMMON_IMPORTS = """
import os
import sys
from pathlib import Path

import pandas as pd

SRC_PATH = os.path.abspath("../src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
"""


def main() -> None:
    notebooks = {
        "Step 1 - Dataset Audit and Scope.ipynb": [
            markdown_cell(
                """
                # Step 1 - Dataset Audit and Scope

                This notebook profiles the LinkedIn postings corpus, saves reusable audit artifacts,
                and surfaces label noise before any modeling choices are made.
                """
            ),
            code_cell(
                COMMON_IMPORTS
                + """
from linkedin_experiment.audit import run_dataset_audit

outputs = run_dataset_audit()
outputs
"""
            ),
            code_cell(
                """
summary = pd.read_csv("../results/audit/audit_summary.csv")
null_rates = pd.read_csv("../results/audit/null_rates.csv")
dates = pd.read_csv("../results/audit/date_coverage.csv")
summary, null_rates.head(12), dates
"""
            ),
            code_cell(
                """
examples = pd.read_csv("../results/audit/experience_examples.csv")
noisy = pd.read_csv("../results/audit/noisy_label_examples.csv")
examples.head(12), noisy.head(12)
"""
            ),
        ],
        "Step 2 - Build Digital Tech Corpus.ipynb": [
            markdown_cell(
                """
                # Step 2 - Build Digital Tech Corpus

                This notebook constructs the digital-tech core corpus, keeps a broader tech-company
                slice for robustness, and saves manual audit samples for scope review.
                """
            ),
            code_cell(
                COMMON_IMPORTS
                + """
from linkedin_experiment.pipeline import build_scope_outputs

outputs = build_scope_outputs()
outputs
"""
            ),
            code_cell(
                """
scope_summary = pd.read_csv("../results/scope/scope_summary.csv")
included_sample = pd.read_csv("../results/scope/included_title_audit_sample.csv")
excluded_sample = pd.read_csv("../results/scope/excluded_title_audit_sample.csv")
scope_summary, included_sample.head(10), excluded_sample.head(10)
"""
            ),
            code_cell(
                """
core = pd.read_csv("../data/processed/linkedin_digital_tech_core.csv")
core[["job_id", "title", "scope_reason", "industry_name"]].head(20)
"""
            ),
        ],
        "Step 3 - Seniority Labeling and Corpus Cleaning.ipynb": [
            markdown_cell(
                """
                # Step 3 - Seniority Labeling and Corpus Cleaning

                This notebook collapses LinkedIn experience levels into four bins, applies title-based
                overrides and backfills, builds normalized modeling text, and removes duplicate templates.
                """
            ),
            code_cell(
                COMMON_IMPORTS
                + """
from linkedin_experiment.pipeline import build_labeling_outputs

outputs = build_labeling_outputs()
outputs
"""
            ),
            code_cell(
                """
summary = pd.read_csv("../results/labeling/linkedin_digital_tech_seniority_summary.csv")
duplicates = pd.read_csv("../results/labeling/linkedin_digital_tech_duplicate_summary.csv")
override_review = pd.read_csv("../results/labeling/linkedin_digital_tech_seniority_override_review.csv")
summary, duplicates, override_review.head(20)
"""
            ),
            code_cell(
                """
labeled = pd.read_csv("../data/processed/linkedin_digital_tech_labeled.csv")
labeled[
    [
        "job_id",
        "title",
        "raw_experience_level",
        "seniority_final",
        "seniority_source",
        "keep_for_model",
        "duplicate_group_id",
    ]
].head(20)
"""
            ),
        ],
        "Step 4 - Train Embeddings and Stability.ipynb": [
            markdown_cell(
                """
                # Step 4 - Train Embeddings and Stability

                This notebook trains one FastText model per seniority group over multiple seeds and
                saves training metadata for later WEAT and monotonicity analysis.
                """
            ),
            code_cell(
                COMMON_IMPORTS
                + """
from linkedin_experiment.embeddings import train_group_models

metadata = train_group_models(seeds=[7, 17, 29])
metadata
"""
            ),
            code_cell(
                """
metadata = pd.read_csv("../results/embeddings/training_metadata.csv")
metadata.groupby("group")[["documents", "token_count", "vocab_size"]].mean()
"""
            ),
        ],
        "Step 5 - WEAT and Monotonicity Testing.ipynb": [
            markdown_cell(
                """
                # Step 5 - WEAT and Monotonicity Testing

                This notebook runs the primary WEAT specification across the four seniority embeddings,
                reports effect sizes and p-values, and evaluates whether the ordering is monotonic.
                """
            ),
            code_cell(
                COMMON_IMPORTS
                + """
from linkedin_experiment.weat import run_weat_suite

seed_results, summary, monotonicity = run_weat_suite(seeds=[7, 17, 29], permutations=1000)
summary, monotonicity
"""
            ),
            code_cell(
                """
seed_results = pd.read_csv("../results/weat/weat_seed_results.csv")
summary = pd.read_csv("../results/weat/weat_summary.csv")
monotonicity = pd.read_csv("../results/weat/monotonicity_summary.csv")
seed_results.head(), summary, monotonicity
"""
            ),
        ],
        "Step 6 - Robustness and Interpretation.ipynb": [
            markdown_cell(
                """
                # Step 6 - Robustness and Interpretation

                This notebook compares labeled-only vs labeled-plus-backfilled corpora, deduped vs
                non-deduped text, and the broader tech-company slice, while also summarizing a simple
                lexicon-based gender-coded language check.
                """
            ),
            code_cell(
                COMMON_IMPORTS
                + """
from linkedin_experiment.pipeline import build_robustness_outputs

outputs = build_robustness_outputs()
outputs
"""
            ),
            markdown_cell(
                """
                ## Optional WEAT robustness reruns

                The cells below create variant corpora and reuse the same training and WEAT functions
                without overwriting the primary digital-tech outputs.
                """
            ),
            code_cell(
                COMMON_IMPORTS
                + """
from pathlib import Path

from linkedin_experiment.pipeline import build_labeling_outputs

labeled_only_variant = build_labeling_outputs(
    input_path=Path("../data/processed/linkedin_digital_tech_labeled_only.csv"),
    prefix="linkedin_digital_tech_labeled_only",
)
broad_variant = build_labeling_outputs(
    input_path=Path("../data/processed/linkedin_broad_tech_company.csv"),
    prefix="linkedin_broad_tech_company",
)
labeled_only_variant, broad_variant
"""
            ),
            code_cell(
                COMMON_IMPORTS
                + """
from pathlib import Path

from linkedin_experiment.embeddings import train_group_models
from linkedin_experiment.weat import run_weat_suite

train_group_models(
    seeds=[7, 17, 29],
    file_template="linkedin_digital_tech_labeled_only_{group}_clean.csv",
    models_dir=Path("../models/linkedin_fasttext_labeled_only"),
)

run_weat_suite(
    seeds=[7, 17, 29],
    models_dir=Path("../models/linkedin_fasttext_labeled_only"),
    results_dir=Path("../results/weat_labeled_only"),
)
"""
            ),
            code_cell(
                """
lexicon_backfilled = pd.read_csv("../results/robustness/linkedin_digital_tech_lexicon_bias_labeled_plus_backfilled.csv")
lexicon_labeled_only = pd.read_csv("../results/robustness/linkedin_digital_tech_lexicon_bias_labeled_only.csv")
lexicon_nondedup = pd.read_csv("../results/robustness/linkedin_digital_tech_lexicon_bias_nondedup.csv")
broad_summary = pd.read_csv("../results/robustness/linkedin_digital_tech_broader_tech_company_summary.csv")
lexicon_backfilled, lexicon_labeled_only, lexicon_nondedup, broad_summary
"""
            ),
        ],
    }

    for name, cells in notebooks.items():
        write_notebook(NOTEBOOKS_DIR / name, cells)


if __name__ == "__main__":
    main()
