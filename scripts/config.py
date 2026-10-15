"""Konfigurasi terpusat: path, nama kolom, dan konstanta proyek."""

from pathlib import Path

# ---- Path ----
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
REPORTS_DIR = ROOT_DIR / "reports"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
METADATA_PATH = DATA_DIR / "metadata.csv"

MODEL_PATH = MODELS_DIR / "pipeline_model_hgb.pkl"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission.csv"

# ---- Kolom ----
ID_COL = "building_id"
TARGET = "damage_grade"

# Fitur geografis: district & vdcmun diperlakukan sebagai kategori (<=255 level),
# ward_id (945 level) diperlakukan numerik karena kodenya hierarkis.
GEO_CATEGORICAL_COLS = ["district_id", "vdcmun_id"]
GEO_NUMERIC_COLS = ["ward_id"]

CATEGORICAL_COLS = [
    "land_surface_condition",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "position",
    "plan_configuration",
    "superstructure",
] + GEO_CATEGORICAL_COLS

NUMERIC_COLS = [
    "count_floors_pre_eq",
    "age_building",
    "plinth_area_sq_ft",
    "height_ft_pre_eq",
] + GEO_NUMERIC_COLS

# Nilai khusus pada age_building yang berarti "umur tidak diketahui".
AGE_UNKNOWN_VALUE = 999

# ---- Eksperimen ----
RANDOM_STATE = 42
N_FOLDS = 3
