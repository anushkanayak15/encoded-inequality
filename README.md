# Encoded Inequality 

Project: Gender-Coded Language and Bias in Job Postings  
Author: Anushka Nayak and Jennifer Mena  
Language: Python 3.10  
Methods: FastText Embeddings, WEAT, Lexicon Analysis  


## Overview

This project analyzes gender-coded language in job postings and how it varies by seniority level.  
The central question is whether leadership roles contain stronger masculine-coded language compared to entry-level roles.

Two parallel experiments were conducted:

1. Synthetic Job Postings Dataset (controlled experiment)  
2. LinkedIn Job Postings Dataset (real-world large-scale data)

Both experiments follow the same conceptual pipeline:

Data Cleaning → Seniority Grouping → Embedding Training → Bias Measurement (WEAT + Lexicon)


## Experiment 1 — Synthetic Job Postings

Purpose:  
To validate the methodology on a controlled dataset where language patterns are intentionally constructed and easier to interpret.

Dataset:  
Simulated job postings representing four seniority levels:

- Entry  
- Mid  
- Senior  
- Leadership  


### Pipeline Notebooks (Run in Order)

1. data_cleaning.ipynb  
   - Removes noise, punctuation, and irrelevant tokens  
   - Standardizes formatting  
   - Handles missing or malformed entries  
   - Produces a clean corpus  

2. preprocessing_and_seniority.ipynb  
   - Assigns postings to seniority groups  
   - Builds separate corpora per group  
   - Ensures meaningful comparisons  

3. train_embeddings_per_seniority_group.ipynb  
   - Trains FastText embeddings for each group  
   - Captures semantic patterns within levels  
   - Produces one model per group  

4. run_weat_per_seniority.ipynb  
   - Runs the Word Embedding Association Test  
   - Measures association between gendered terms and career concepts  
   - Produces effect sizes and significance values  

5. gender_coded_language_analysis.ipynb  
   - Lexicon-based analysis using predefined masculine/feminine word lists  
   - Counts occurrences of gender-coded terms  
   - Provides interpretable evidence alongside embedding results  


### Outputs

results/
  weat/
  lexicon/
  plots/

These outputs confirm whether the pipeline detects expected bias patterns in controlled data.


## Experiment 2 - LinkedIn Job Postings

Purpose:  
To apply the same methodology to real-world job postings and evaluate whether the findings generalize beyond synthetic data.

Dataset Source:  
LinkedIn Job Postings (2023–2024) from Kaggle  
https://www.kaggle.com/datasets/arshkon/linkedin-job-postings


### Pipeline Steps

1. Audit  
   - Evaluates raw data quality  
   - Identifies missing values and anomalies  
   - Produces audit summaries  

2. Scope  
   - Filters to digital/tech occupations  
   - Removes unrelated job categories  
   - Produces a focused corpus  

3. Label  
   - Assigns seniority levels  
   - Cleans text and removes duplicates  
   - Produces final modeling corpora  

4. Train  
   - Trains FastText embeddings per seniority group  
   - Produces large model files  

5. WEAT  
   - Computes bias metrics using trained embeddings  
   - Produces statistical effect sizes  

6. Robustness  
   - Tests stability under alternative settings  
   - Produces comparison outputs  


### Outputs

data/processed/
models/
results/audit/
results/embeddings/
results/weat/
results/robustness/


## Required Data for LinkedIn Experiment

All raw files must be placed inside a folder named:

linkedin_data/

Required structure:

linkedin_data/postings.csv  
linkedin_data/jobs/job_industries.csv  
linkedin_data/mappings/industries.csv  
linkedin_data/companies/companies.csv  


## Environment Setup

Python version: 3.10 recommended

Required packages:

- pandas  
- numpy  
- matplotlib  
- gensim  
- scipy  
- jupyter  


Example setup:

python -m venv .venv  
.\.venv\Scripts\python.exe -m pip install pandas numpy matplotlib gensim scipy jupyter  



## Important Note About Models

FastText model files are very large and are NOT included in the repository.



## Conceptual Research Goal

Across both datasets, the project investigates whether:

- Leadership roles use more masculine-coded language  
- Entry-level roles use more communal or feminine-coded language  
- Bias strength increases with seniority  
- Patterns hold in both controlled and real-world data  

Combining embedding-based WEAT with lexicon counts provides both quantitative rigor and interpretability.


