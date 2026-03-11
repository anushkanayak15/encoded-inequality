# LinkedIn Analysis Report

## What this project is asking

This LinkedIn version of the project asks a simple question:

Do higher-seniority digital-tech job ads show a stronger male-career language pattern than lower-seniority job ads?

To test that, the pipeline builds a digital-tech job corpus, labels postings by seniority, cleans the text, trains FastText models, and then runs WEAT. In this report, a higher positive WEAT score means a stronger male-career association in that seniority group.

For rerun instructions, see [README-LinkedIn-Analysis.md](README-LinkedIn-Analysis.md).

## Why the steps are in this order

The order matters because each stage depends on the one before it.

1. We audit the raw data first so we know what is missing and where the labels are noisy.
2. We define the digital-tech scope before modeling so the experiment is about a clear job market, not a mix of unrelated occupations.
3. We label seniority and clean the text before training so the models are built on the right groups and not inflated by duplicate templates.
4. We train embeddings only after the text is ready.
5. We run WEAT only after the models exist.
6. We do robustness checks last so we can compare them against the main result.

This order makes the analysis easier to explain and easier to defend.

## Dataset context

The LinkedIn rebuild uses:

- `new_data/postings.csv`
- `new_data/jobs/job_industries.csv`
- `new_data/mappings/industries.csv`
- `new_data/companies/companies.csv`

The raw LinkedIn postings table has:

- `123,849` rows
- `31` columns
- `0` duplicate `job_id` values
- only `7` empty descriptions

That sounds strong at first, but the audit also showed some important problems:

- `skills_desc` is missing for about `98.0%` of rows, so it cannot be the main text field.
- salary fields are mostly missing too.
- `formatted_experience_level` is missing for about `23.7%` of rows, so we cannot rely on the platform label alone.
- `closed_time` is almost always missing, so it is not useful for the main analysis.

## Step 1. Dataset audit

The first step was about understanding the raw data before making any design choices.

Main findings:

- The corpus is large and mostly complete on `title`, `description`, dates, and location.
- The fields that look useful at first glance are not equally reliable.
- The experience labels are helpful, but not perfect.
- We found clear examples where the title and the LinkedIn experience label do not agree.

Why this mattered:

- It justified using `description` as the main text source.
- It justified doing title-based backfills and overrides later.
- It showed that scope filtering and text cleaning were not optional.

Main saved outputs:

- `results/audit/audit_summary.csv`
- `results/audit/null_rates.csv`
- `results/audit/date_coverage.csv`
- `results/audit/experience_examples.csv`
- `results/audit/noisy_label_examples.csv`

## Step 2. Build the digital-tech corpus

The second step narrowed the project to a digital-tech core corpus. This was important because the raw LinkedIn dataset covers many occupations that are not really comparable.

Scope results:

- `123,849` total postings reviewed
- `10,511` postings kept in the digital-tech core corpus
- `113,338` postings left out of the main scope
- `20,411` postings saved in a broader tech-company comparison slice

Why this mattered:

- The main experiment is supposed to compare seniority inside a more coherent labor market.
- If we kept everything, differences could come from occupation mix instead of seniority.
- The broader tech-company slice is still useful, but only as a robustness comparison.

Main saved outputs:

- `data/processed/linkedin_scope_all.csv`
- `data/processed/linkedin_digital_tech_core.csv`
- `data/processed/linkedin_broad_tech_company.csv`
- `results/scope/scope_summary.csv`

## Step 3. Seniority labeling and corpus cleaning

The third step turned the scoped dataset into the four final seniority groups:

- `entry`
- `mid`
- `senior`
- `leadership`

This step used the LinkedIn experience label first, then used title rules to fix obvious mistakes and fill missing cases.

Final seniority counts:

| Group | Rows |
| --- | ---: |
| entry | 1,257 |
| mid | 2,296 |
| senior | 4,668 |
| leadership | 2,290 |

Where the labels came from:

| Group | LinkedIn primary | Title backfill | Title override |
| --- | ---: | ---: | ---: |
| entry | 1,082 | 138 | 37 |
| mid | 370 | 1,926 | 0 |
| senior | 2,121 | 2,417 | 130 |
| leadership | 56 | 656 | 1,578 |

This is one of the most important findings in the whole pipeline. Leadership especially depended heavily on title overrides, which tells us the raw platform labels were not enough by themselves.

The cleaning step also removed repeated or templated postings.

Duplicate summary:

- `10,511` rows before deduplication
- `9,622` rows kept for modeling
- `749` exact duplicates
- `889` template duplicates

Why this mattered:

- Without deduplication, the embedding models would learn repeated posting templates instead of general language patterns.
- The kept corpus is still large enough to model, but it is much cleaner.

Main clean corpus sizes used for the intended main experiment:

| Group | Clean documents | Total tokens |
| --- | ---: | ---: |
| entry | 1,122 | 623,900 |
| mid | 2,049 | 769,121 |
| senior | 4,322 | 2,066,579 |
| leadership | 2,129 | 1,317,684 |

Main saved outputs:

- `data/processed/linkedin_digital_tech_labeled.csv`
- `data/processed/linkedin_digital_tech_labeled_only.csv`
- `data/processed/linkedin_digital_tech_<group>_clean.csv`
- `results/labeling/linkedin_digital_tech_seniority_summary.csv`
- `results/labeling/linkedin_digital_tech_duplicate_summary.csv`

## Step 4. Train embeddings

The fourth step trains one FastText model per seniority group across multiple random seeds.

The main idea here is simple: we do not want one lucky random initialization to decide the whole story. That is why the training uses multiple seeds, and Step 5 summarizes the average result.

The current code uses:

- FastText
- seeds `7`, `17`, and `29`

Important note about the files currently on disk:

- the main clean corpora listed above are the intended inputs for the primary experiment
- the current `results/embeddings/training_metadata.csv` reflects a later labeled-only robustness rerun
- that robustness rerun is much smaller, especially for leadership

So if someone wants to fully reproduce the main experiment from scratch, Step 4 should be rerun locally to recreate the primary model files under `models/`.

Why this step mattered:

- the models are too large to keep on GitHub
- Step 5 cannot run without them
- Step 6 compares alternative versions of the modeling corpus, so reproducible training is necessary

## Step 5. WEAT and monotonicity results

This is the main hypothesis test.

Expected pattern:

`entry < mid < senior < leadership`

Current WEAT summary:

| Group | Mean effect size | Mean p-value |
| --- | ---: | ---: |
| entry | 0.538 | 0.239 |
| mid | 1.092 | 0.103 |
| senior | 0.871 | 0.108 |
| leadership | 0.012 | 0.494 |

Main result:

- the strongest effect appears in `mid`
- `senior` is also clearly positive
- `entry` is positive but smaller
- `leadership` is close to zero

Monotonicity result:

- `seed_runs = 3`
- `mean_order_holds = False`
- `seed_order_rate = 0.0`
- interpretation = `not_supported`

What that means in plain language:

The current LinkedIn experiment does **not** support the original monotonic hypothesis. The language does not steadily move upward from entry to leadership in the way we expected. Instead, the biggest male-career signal shows up in the middle and senior groups, while leadership is much weaker than expected.

This is a real result, not a mistake in the report. The pipeline was built so the hypothesis could fail, and right now it fails.

At the same time, the p-values are still above the usual `0.05` threshold, so these patterns should be read as suggestive rather than final proof.

Main saved outputs:

- `results/weat/weat_seed_results.csv`
- `results/weat/weat_summary.csv`
- `results/weat/monotonicity_summary.csv`

## Step 6. Robustness and interpretation

The final step checks whether the findings depend too much on one exact preprocessing choice.

### 6a. Labeled-only WEAT rerun

In this variant, we keep only rows that came directly from LinkedIn primary labels and drop the title backfills and overrides.

Labeled-only WEAT summary:

| Group | Mean effect size | Mean p-value |
| --- | ---: | ---: |
| entry | 0.521 | 0.236 |
| mid | 0.897 | 0.167 |
| senior | 0.748 | 0.192 |
| leadership | NA | NA |

This matters because:

- entry, mid, and senior stay positive
- leadership cannot be tested here because one of the target sets has `0` coverage
- that tells us the leadership result is especially sensitive to how the corpus is labeled

### 6b. Simple lexicon robustness check

The project also compared masculine-coded and feminine-coded word rates per 1,000 tokens.

Bias score summary:

| Group | Labeled + backfilled | Labeled-only | Non-deduped |
| --- | ---: | ---: | ---: |
| entry | 0.220 | 0.302 | 0.324 |
| mid | -0.849 | -0.510 | -0.804 |
| senior | -0.084 | -0.605 | 0.030 |
| leadership | 1.010 | -1.038 | 0.991 |

How to read that table:

- positive values mean more masculine-coded words than feminine-coded words
- negative values mean the opposite

The most important takeaway is not the exact number. The important takeaway is that leadership changes a lot across variants. That again tells us the leadership result is unstable and highly dependent on corpus construction.

### 6c. Broader scope benchmark

The broader tech-company comparison slice contains:

- `20,411` broader-tech-company rows
- `10,511` digital-tech-core rows

This is useful because it reminds us that scope decisions matter a lot. The broader slice is almost twice as large, but it is not the right main population for the hypothesis.

## Overall interpretation

This experiment followed a clear research order:

1. audit the data
2. define the scope
3. label seniority
4. train the models
5. test the hypothesis
6. check robustness

Here is the clearest simple reading of the current project:

1. The LinkedIn data is large enough to run a serious experiment, but it is noisy enough that scope, backfilling, and deduplication really matter.
2. The main hypothesis is **not supported** in the current run.
3. Male-career association is strongest in the `mid` and `senior` groups, not in `leadership`.
4. Leadership is the least stable group across robustness checks.
5. Because the p-values are still above `0.05`, the project presents these findings as careful evidence, not as a final causal claim.

## Main limitations

- Leadership is especially sensitive to labeling choices.
- The labeled-only leadership subset is too small for a full WEAT rerun.
- The main WEAT results are directionally interesting, but not conventionally significant.

