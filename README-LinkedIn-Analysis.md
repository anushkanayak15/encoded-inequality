# README - LinkedIn Analysis

## What this file is for

This short guide explains how to rerun only the LinkedIn experiment in this repo.

The trained FastText models are not meant to be pushed to GitHub because they are too large. That means anyone who wants the full LinkedIn results needs to rebuild the models locally.

## What you need before you start

You need the LinkedIn source files which must be downloaded from [Kaggle](https://www.kaggle.com/).

The dataset used is [LinkedIn Job Postings (2023 - 2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

All the files should be store in a folder named `linkedin_data`. The final file structure must be:

- `linkedin_data/postings.csv`
- `linkedin_data/jobs/job_industries.csv`
- `linkedin_data/mappings/industries.csv`
- `linkedin_data/companies/companies.csv`

You also need Python `3.10`.

If the project `.venv` already exists, use it. If not, create a new virtual environment and install the main packages used by this pipeline:

- `pandas`
- `numpy`
- `matplotlib`
- `gensim`
- `scipy`
- `jupyter`

Example setup from scratch:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pandas numpy matplotlib gensim scipy jupyter
```

If PowerShell activation is blocked on another machine, it is fine to skip activation and just call `.\.venv\Scripts\python.exe` directly in every command.

## Why the experiment has to be run in this order

The order is important because every later step depends on outputs from the earlier steps.

1. `Step 1`
Audit first: this shows missing data, noisy labels, and general data quality before we make decisions.

2. `Step 1`
Scope second: this defines which rows count as digital-tech jobs. If we skip this, the experiment mixes unrelated occupations.

3. `Step 3`
Label third: this creates the four seniority groups and cleans the text. The model training step needs these cleaned files.

4. `Step 4`
Train fourth: this creates the FastText model files. These are the heavy files we are not pushing to GitHub.

5. `Step 5`
WEAT fifth: WEAT uses the saved models from Step 4, so it cannot run first.

6. `Step 6`
Robustness last: this compares alternative versions of the pipeline against the main result.

## Fastest way to rerun the LinkedIn experiment

Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

Then run the LinkedIn pipeline in this order:

```powershell
.\.venv\Scripts\python.exe src\run_linkedin_pipeline.py audit
.\.venv\Scripts\python.exe src\run_linkedin_pipeline.py scope
.\.venv\Scripts\python.exe src\run_linkedin_pipeline.py label
.\.venv\Scripts\python.exe src\run_linkedin_pipeline.py train --seeds 7 17 29
.\.venv\Scripts\python.exe src\run_linkedin_pipeline.py weat --seeds 7 17 29 --permutations 1000
.\.venv\Scripts\python.exe src\run_linkedin_pipeline.py robustness
```

If you want a shortcut for the prep stages, you can combine scope and labeling like this:

```powershell
.\.venv\Scripts\python.exe src\run_linkedin_pipeline.py prep-all
```

Then continue with:

```powershell
.\.venv\Scripts\python.exe src\run_linkedin_pipeline.py train --seeds 7 17 29
.\.venv\Scripts\python.exe src\run_linkedin_pipeline.py weat --seeds 7 17 29 --permutations 1000
.\.venv\Scripts\python.exe src\run_linkedin_pipeline.py robustness
```

## What each command creates

`audit`

- checks the raw data
- writes audit tables to `results/audit/`

`scope`

- builds the digital-tech core corpus
- writes scoped datasets to `data/processed/`
- writes scope summaries to `results/scope/`

`label`

- creates the final seniority groups
- builds cleaned modeling text
- removes duplicate or template-like postings
- writes labeled and clean corpora to `data/processed/`
- writes summaries to `results/labeling/`

`train`

- trains one FastText model per seniority group
- saves heavy model files under `models/`
- writes metadata to `results/embeddings/training_metadata.csv`

`weat`

- runs the main word embedding association test
- writes the result tables to `results/weat/`

`robustness`

- writes the extra comparison outputs to `results/robustness/`

## Notebook order

To regenerate the notebooks:

```powershell
.\.venv\Scripts\python.exe scripts\generate_linkedin_notebooks.py
```

If you prefer the notebooks `notebooks-LinkedIn-Analysis` instead of the CLI, open them in this order:

1. `Step 1 - Dataset Audit and Scope.ipynb`
2. `Step 2 - Build Digital Tech Corpus.ipynb`
3. `Step 3 - Seniority Labeling and Corpus Cleaning.ipynb`
4. `Step 4 - Train Embeddings and Stability.ipynb`
5. `Step 5 - WEAT and Monotonicity Testing.ipynb`
6. `Step 6 - Robustness and Interpretation.ipynb`

Each notebook also has a final `Report Visual Aids` section at the end. If the saved CSV results are already there, you can run just those final plotting cells to refresh the report figures without rerunning the expensive steps.

## Important note about the missing models

The `models/` folder is exactly what is too heavy to push.

So if someone clones the repo and the models are missing, that is expected. They just need to rerun:

```powershell
.\.venv\Scripts\python.exe src\run_linkedin_pipeline.py train --seeds 7 17 29
.\.venv\Scripts\python.exe src\run_linkedin_pipeline.py weat --seeds 7 17 29 --permutations 1000
```

After that, the LinkedIn experiment should have the model files and WEAT outputs again.
