"""Central settings. Change numbers here, not all over the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, MODEL_DIR = ROOT / "data", ROOT / "models"
OUT_DIR, IMG_DIR = ROOT / "outputs", ROOT / "images"

RAW_CSV = DATA_DIR / "sensor_data.csv"
PROC_CSV = DATA_DIR / "processed_features.csv"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
PRED_CSV = OUT_DIR / "predictions.csv"
ALERT_CSV = OUT_DIR / "alerts.csv"
METRICS_JSON = OUT_DIR / "metrics.json"

SENSORS = ["temperature", "vibration", "pressure", "rpm", "current"]
N_MACHINES = 20          # virtual machines
N_STEPS = 2000           # hourly readings per machine
HORIZON = 24             # label = 1 if failure happens within next 24 hours
WINDOWS = (6, 24)        # rolling-window sizes (hours)
TEST_FRACTION = 0.2      # last 20% of time is the test set
RANDOM_STATE = 42

WARN_THRESHOLD = 0.40    # risk >= 40%  -> WARNING
CRIT_THRESHOLD = 0.70    # risk >= 70%  -> CRITICAL

# Industrial "control-room" colour theme (used by plots AND the web dashboard)
THEME = {
    "bg": "#0E1318", "panel": "#18212B", "border": "#2A3745",
    "text": "#E6EDF3", "muted": "#8B98A5",
    "amber": "#FFB020",   # primary accent (warning / brand)
    "red": "#FF4D4F",     # failure / critical
    "green": "#2ECC71",   # healthy
    "blue": "#4DA3FF",    # data lines
}

def ensure_dirs():
    for d in (DATA_DIR, MODEL_DIR, OUT_DIR, IMG_DIR):
        d.mkdir(parents=True, exist_ok=True)
