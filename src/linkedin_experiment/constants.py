from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NEW_DATA_DIR = PROJECT_ROOT / "linkedin_data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

AUDIT_RESULTS_DIR = RESULTS_DIR / "audit"
SCOPE_RESULTS_DIR = RESULTS_DIR / "scope"
LABELING_RESULTS_DIR = RESULTS_DIR / "labeling"
EMBEDDING_RESULTS_DIR = RESULTS_DIR / "embeddings"
WEAT_RESULTS_DIR = RESULTS_DIR / "weat"
ROBUSTNESS_RESULTS_DIR = RESULTS_DIR / "robustness"

FASTTEXT_MODELS_DIR = MODELS_DIR / "linkedin_fasttext"

POSTINGS_PATH = NEW_DATA_DIR / "postings.csv"
JOB_INDUSTRIES_PATH = NEW_DATA_DIR / "jobs" / "job_industries.csv"
INDUSTRIES_PATH = NEW_DATA_DIR / "mappings" / "industries.csv"
COMPANIES_PATH = NEW_DATA_DIR / "companies" / "companies.csv"

GROUP_ORDER = ["entry", "mid", "senior", "leadership"]
GROUP_FILE_TEMPLATE = "linkedin_digital_tech_{group}_clean.csv"
