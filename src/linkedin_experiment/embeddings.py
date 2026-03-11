from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from gensim.models import FastText

from .constants import (
    EMBEDDING_RESULTS_DIR,
    FASTTEXT_MODELS_DIR,
    GROUP_ORDER,
    PROCESSED_DATA_DIR,
)
from .io import ensure_dir, write_csv


@dataclass(frozen=True)
class TrainingConfig:
    vector_size: int = 150
    window: int = 5
    epochs: int = 12
    min_n: int = 3
    max_n: int = 6
    negative: int = 10
    sg: int = 1
    workers: int = 1


def get_min_count(doc_count: int) -> int:
    if doc_count < 500:
        return 1
    if doc_count < 1500:
        return 2
    if doc_count < 5000:
        return 3
    return 5


def make_sentences(text_series: pd.Series) -> list[list[str]]:
    return [str(text).split() for text in text_series.fillna("") if str(text).strip()]


def train_fasttext_model(
    sentences: Sequence[Sequence[str]],
    seed: int,
    config: TrainingConfig | None = None,
) -> FastText:
    cfg = config or TrainingConfig()
    min_count = get_min_count(len(sentences))
    model = FastText(
        sentences=sentences,
        vector_size=cfg.vector_size,
        window=cfg.window,
        epochs=cfg.epochs,
        min_count=min_count,
        workers=cfg.workers,
        sg=cfg.sg,
        negative=cfg.negative,
        min_n=cfg.min_n,
        max_n=cfg.max_n,
        seed=seed,
    )
    return model


def train_group_models(
    groups: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    processed_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = FASTTEXT_MODELS_DIR,
    results_dir: Path = EMBEDDING_RESULTS_DIR,
    config: TrainingConfig | None = None,
    file_template: str = "linkedin_digital_tech_{group}_clean.csv",
) -> pd.DataFrame:
    group_names = list(groups or GROUP_ORDER)
    seed_values = list(seeds or [7, 17, 29])
    ensure_dir(models_dir)
    ensure_dir(results_dir)

    rows: list[dict[str, object]] = []
    for group in group_names:
        corpus_path = processed_dir / file_template.format(group=group)
        df = pd.read_csv(corpus_path)
        sentences = make_sentences(df["model_text"])
        token_count = int(df["token_count"].sum()) if "token_count" in df.columns else sum(len(s) for s in sentences)
        for seed in seed_values:
            model = train_fasttext_model(sentences, seed=seed, config=config)
            group_dir = ensure_dir(models_dir / group / f"seed_{seed}")
            model_path = group_dir / "fasttext.model"
            vectors_path = group_dir / "fasttext.kv"
            model.save(str(model_path))
            model.wv.save(str(vectors_path))
            rows.append(
                {
                    "group": group,
                    "seed": seed,
                    "documents": len(df),
                    "sentences": len(sentences),
                    "token_count": token_count,
                    "vocab_size": len(model.wv),
                    "min_count": get_min_count(len(sentences)),
                    "model_path": model_path.as_posix(),
                    "vectors_path": vectors_path.as_posix(),
                }
            )

    metadata = pd.DataFrame(rows)
    write_csv(metadata, results_dir / "training_metadata.csv")
    return metadata
