# ============================================================
# config.py — Global Configuration for HematoAI v3.0
# ============================================================

# ── LLM Backend ──────────────────────────────────────────────
# Change to 'ollama' to run fully offline.
# Can also be changed at runtime from: Main Menu → S (Settings)
LLM_BACKEND = 'groq'

# ── CBC Feature Columns ──────────────────────────────────────
# Raw CBC features (input from user / CSV / PDF / image)
RAW_FEATURES = ['Hb', 'MCV', 'MCH', 'MCHC', 'RBC', 'RDW', 'Hematocrit']

# All features used by the ML model (raw + derived indices)
FEATURES = [
    'Hb', 'MCV', 'MCH', 'MCHC', 'RBC', 'RDW', 'Hematocrit',
    'Mentzer_Index', 'Shine_Lal', 'England_Fraser',
    'RDW_MCV_Ratio', 'Hb_RBC_Ratio'
]

# ── Target Classes ───────────────────────────────────────────
TARGET_CLASSES = ['IDA', 'Thalassemia_Trait', 'Anemia_of_Chronic_Disease']
TARGET_COL = 'label'

# ── Biologic Plausibility Ranges ─────────────────────────────
# Used for input validation — values outside these are flagged
BIOLOGIC_RANGES = {
    'Hb':          [3,   20],
    'MCV':         [40,  130],
    'MCH':         [10,  45],
    'MCHC':        [20,  40],
    'RBC':         [1.0, 8.0],
    'RDW':         [8,   30],
    'Hematocrit':  [10,  65],
}

# ── Derived Index Formulas (informational) ───────────────────
# Mentzer Index:    MCV / RBC            (>13 → IDA, <13 → Thal)
# Shine-Lal:       MCV² × MCH / 100
# England-Fraser:  MCV - RBC - (5 × Hb) (negative → Thal)
# RDW/MCV Ratio:   RDW / MCV
# Hb/RBC Ratio:    Hb / RBC

# ── Paths ─────────────────────────────────────────────────────
MODEL_DIR   = 'models/'
DATA_DIR    = 'data/'
OUTPUT_DIR  = 'outputs/'
UPLOAD_DIR  = 'uploads/'

SYNTHETIC_CSV = 'data/synthetic/cbc_synthetic.csv'
ACTIVE_CSV    = 'data/active/cbc_active.csv'
MODEL_PATH    = 'models/best_model.pkl'
SCALER_PATH   = 'models/scaler.pkl'
ENCODER_PATH  = 'models/label_encoder.pkl'
METADATA_PATH = 'models/model_metadata.json'
SETTINGS_PATH = '.settings.json'

# ── Training Hyperparameters ─────────────────────────────────
OPTUNA_TRIALS = 50
CV_FOLDS      = 5
TEST_SIZE     = 0.2
RANDOM_STATE  = 42

# ── LLM Defaults ─────────────────────────────────────────────
DEFAULT_TEMPERATURE = 0.2
GROQ_TEXT_MODEL     = 'llama-3.3-70b-versatile'
GROQ_VISION_MODEL   = 'meta-llama/llama-4-scout-17b-16e-instruct'
OLLAMA_TEXT_MODEL   = 'llama3.2'
OLLAMA_VISION_MODEL = 'llava'
OLLAMA_BASE_URL     = 'http://localhost:11434'
