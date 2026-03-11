from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linkedin_experiment.labeling import assign_seniority
from linkedin_experiment.scope import classify_scope
from linkedin_experiment.text import build_model_text, prepare_text_corpus
from linkedin_experiment.weat import summarize_monotonicity


class ScopeTests(unittest.TestCase):
    def test_software_engineer_is_in_scope(self) -> None:
        decision = classify_scope("Software Engineer", "Build APIs", False)
        self.assertTrue(decision.include)
        self.assertEqual(decision.reason, "title_core")

    def test_data_entry_is_out_of_scope(self) -> None:
        decision = classify_scope("Data Entry Clerk", "Update spreadsheets", False)
        self.assertFalse(decision.include)
        self.assertEqual(decision.exclusion_reason, "off_target_business_role")

    def test_ambiguous_title_requires_support(self) -> None:
        unsupported = classify_scope("Project Manager", "Coordinate vendors", False)
        supported = classify_scope("Project Manager", "Manage cloud migrations on AWS", False)
        self.assertFalse(unsupported.include)
        self.assertTrue(supported.include)


class LabelingTests(unittest.TestCase):
    def test_junior_title_overrides_noisy_level(self) -> None:
        decision = assign_seniority("Junior Software Engineer", "Mid-Senior level")
        self.assertEqual(decision.seniority_final, "entry")
        self.assertEqual(decision.seniority_source, "title_override")

    def test_senior_title_overrides_associate(self) -> None:
        decision = assign_seniority("Senior Developer", "Associate")
        self.assertEqual(decision.seniority_final, "senior")
        self.assertEqual(decision.seniority_source, "title_override")

    def test_linkedin_primary_used_when_no_title_override(self) -> None:
        decision = assign_seniority("Software Engineer", "Entry level")
        self.assertEqual(decision.seniority_final, "entry")
        self.assertEqual(decision.seniority_source, "linkedin_primary")


class TextTests(unittest.TestCase):
    def test_build_model_text_preserves_common_tech_tokens(self) -> None:
        text = build_model_text("C++ / C# Engineer", "Build Node.js APIs in .NET")
        self.assertIn("cplusplus", text)
        self.assertIn("csharp", text)
        self.assertIn("nodejs", text)
        self.assertIn("dotnet", text)

    def test_prepare_text_corpus_marks_duplicates(self) -> None:
        df = pd.DataFrame(
            [
                {"title": "Software Engineer", "description": "Build APIs", "skills_desc": None},
                {"title": "Software Engineer", "description": "Build APIs", "skills_desc": None},
            ]
        )
        prepared = prepare_text_corpus(df)
        self.assertEqual(int(prepared["keep_for_model"].sum()), 1)
        self.assertTrue(bool(prepared["is_exact_duplicate"].iloc[1]))


class WeatTests(unittest.TestCase):
    def test_monotonicity_summary_reports_supported(self) -> None:
        df = pd.DataFrame(
            [
                {"test_name": "gender_career_family", "group": "entry", "seed": 1, "effect_size": 0.1},
                {"test_name": "gender_career_family", "group": "mid", "seed": 1, "effect_size": 0.2},
                {"test_name": "gender_career_family", "group": "senior", "seed": 1, "effect_size": 0.3},
                {"test_name": "gender_career_family", "group": "leadership", "seed": 1, "effect_size": 0.4},
                {"test_name": "gender_career_family", "group": "entry", "seed": 2, "effect_size": 0.05},
                {"test_name": "gender_career_family", "group": "mid", "seed": 2, "effect_size": 0.15},
                {"test_name": "gender_career_family", "group": "senior", "seed": 2, "effect_size": 0.25},
                {"test_name": "gender_career_family", "group": "leadership", "seed": 2, "effect_size": 0.35},
            ]
        )
        summary = summarize_monotonicity(df)
        self.assertEqual(summary.iloc[0]["interpretation"], "supported")


if __name__ == "__main__":
    unittest.main()
