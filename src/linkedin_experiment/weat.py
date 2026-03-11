from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from gensim.models import FastText

from .constants import FASTTEXT_MODELS_DIR, GROUP_ORDER, WEAT_RESULTS_DIR
from .io import ensure_dir, write_csv
from .lexicons import WEAT_SPECS


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def filter_vocab(model: FastText, words: Sequence[str]) -> list[str]:
    return [word for word in words if word in model.wv.key_to_index]


def association(model: FastText, word: str, attrs_a: Sequence[str], attrs_b: Sequence[str]) -> float:
    vec = model.wv[word]
    sim_a = np.mean([cosine_similarity(vec, model.wv[attr]) for attr in attrs_a])
    sim_b = np.mean([cosine_similarity(vec, model.wv[attr]) for attr in attrs_b])
    return float(sim_a - sim_b)


def weat_effect_size(
    model: FastText,
    targets_a: Sequence[str],
    targets_b: Sequence[str],
    attrs_a: Sequence[str],
    attrs_b: Sequence[str],
) -> float:
    assoc_a = np.array([association(model, word, attrs_a, attrs_b) for word in targets_a])
    assoc_b = np.array([association(model, word, attrs_a, attrs_b) for word in targets_b])
    numerator = assoc_a.mean() - assoc_b.mean()
    denominator = np.std(np.concatenate([assoc_a, assoc_b]), ddof=1)
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def weat_test_statistic(
    model: FastText,
    targets_a: Sequence[str],
    targets_b: Sequence[str],
    attrs_a: Sequence[str],
    attrs_b: Sequence[str],
) -> float:
    score_a = sum(association(model, word, attrs_a, attrs_b) for word in targets_a)
    score_b = sum(association(model, word, attrs_a, attrs_b) for word in targets_b)
    return float(score_a - score_b)


def permutation_p_value(
    model: FastText,
    targets_a: Sequence[str],
    targets_b: Sequence[str],
    attrs_a: Sequence[str],
    attrs_b: Sequence[str],
    num_samples: int = 1000,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    observed = weat_test_statistic(model, targets_a, targets_b, attrs_a, attrs_b)
    combined = list(targets_a) + list(targets_b)
    size_a = len(targets_a)
    if len(combined) <= 16:
        choices = list(combinations(combined, size_a))
    else:
        choices = [tuple(rng.choice(combined, size=size_a, replace=False)) for _ in range(num_samples)]

    exceed_count = 0
    for subset_a in choices:
        remaining = list(combined)
        chosen = list(subset_a)
        for word in chosen:
            remaining.remove(word)
        score = weat_test_statistic(model, chosen, remaining, attrs_a, attrs_b)
        if score >= observed:
            exceed_count += 1
    return float((exceed_count + 1) / (len(choices) + 1))


def run_weat_for_model(
    model: FastText,
    spec_name: str,
    seed: int,
    permutations: int = 1000,
) -> dict[str, object]:
    spec = WEAT_SPECS[spec_name]
    targets_a = filter_vocab(model, spec["targets_a"])
    targets_b = filter_vocab(model, spec["targets_b"])
    attrs_a = filter_vocab(model, spec["attributes_a"])
    attrs_b = filter_vocab(model, spec["attributes_b"])
    coverage_ok = all([targets_a, targets_b, attrs_a, attrs_b])

    result = {
        "test_name": spec_name,
        "coverage_targets_a": len(targets_a),
        "coverage_targets_b": len(targets_b),
        "coverage_attrs_a": len(attrs_a),
        "coverage_attrs_b": len(attrs_b),
    }
    if not coverage_ok:
        result.update({"effect_size": np.nan, "p_value": np.nan})
        return result

    result.update(
        {
            "effect_size": weat_effect_size(model, targets_a, targets_b, attrs_a, attrs_b),
            "p_value": permutation_p_value(
                model,
                targets_a,
                targets_b,
                attrs_a,
                attrs_b,
                num_samples=permutations,
                seed=seed,
            ),
        }
    )
    return result


def summarize_monotonicity(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for test_name, subset in results.groupby("test_name"):
        pivot = subset.pivot_table(index="seed", columns="group", values="effect_size")
        pivot = pivot.reindex(columns=GROUP_ORDER).dropna()
        if pivot.empty:
            rows.append(
                {
                    "test_name": test_name,
                    "seed_runs": 0,
                    "mean_order_holds": False,
                    "seed_order_rate": np.nan,
                    "interpretation": "insufficient_seed_coverage",
                }
            )
            continue

        mean_effects = pivot.mean()
        mean_order_holds = bool(
            mean_effects["entry"] < mean_effects["mid"] < mean_effects["senior"] < mean_effects["leadership"]
        )
        seed_order_rate = (
            (
                (pivot["entry"] < pivot["mid"])
                & (pivot["mid"] < pivot["senior"])
                & (pivot["senior"] < pivot["leadership"])
            ).mean()
        )

        if mean_order_holds and seed_order_rate == 1.0:
            interpretation = "supported"
        elif mean_order_holds:
            interpretation = "partially_supported"
        else:
            interpretation = "not_supported"

        rows.append(
            {
                "test_name": test_name,
                "seed_runs": len(pivot),
                "entry_mean": mean_effects["entry"],
                "mid_mean": mean_effects["mid"],
                "senior_mean": mean_effects["senior"],
                "leadership_mean": mean_effects["leadership"],
                "mean_order_holds": mean_order_holds,
                "seed_order_rate": float(seed_order_rate),
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def run_weat_suite(
    groups: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    models_dir: Path = FASTTEXT_MODELS_DIR,
    results_dir: Path = WEAT_RESULTS_DIR,
    spec_names: Sequence[str] | None = None,
    permutations: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(results_dir)
    group_names = list(groups or GROUP_ORDER)
    seed_values = list(seeds or [7, 17, 29])
    specs = list(spec_names or WEAT_SPECS.keys())

    rows: list[dict[str, object]] = []
    for group in group_names:
        for seed in seed_values:
            model_path = models_dir / group / f"seed_{seed}" / "fasttext.model"
            model = FastText.load(str(model_path))
            for spec_name in specs:
                row = run_weat_for_model(model, spec_name, seed=seed, permutations=permutations)
                row.update({"group": group, "seed": seed, "model_path": model_path.as_posix()})
                rows.append(row)

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["test_name", "group"], as_index=False)
        .agg(
            effect_size_mean=("effect_size", "mean"),
            effect_size_std=("effect_size", "std"),
            p_value_mean=("p_value", "mean"),
            coverage_targets_a=("coverage_targets_a", "mean"),
            coverage_targets_b=("coverage_targets_b", "mean"),
            coverage_attrs_a=("coverage_attrs_a", "mean"),
            coverage_attrs_b=("coverage_attrs_b", "mean"),
        )
    )
    monotonicity = summarize_monotonicity(results)

    write_csv(results, results_dir / "weat_seed_results.csv")
    write_csv(summary, results_dir / "weat_summary.csv")
    write_csv(monotonicity, results_dir / "monotonicity_summary.csv")
    return results, summary, monotonicity
