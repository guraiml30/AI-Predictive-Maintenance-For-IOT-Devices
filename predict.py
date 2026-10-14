"""Load the saved model and predict risk (used by the API and by you in a notebook)."""
import joblib, pandas as pd
from src.config import *
from src.features import add_features

_bundle = None
def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle

def predict_steady(readings: dict) -> float:
    """What-if: assume the machine has been running at these readings for the last 24 h."""
    b = _load()
    df = pd.DataFrame([{"machine_id": "X", **readings}] * 24)
    X = add_features(df)[b["features"]].tail(1)
    return float(b["model"].predict_proba(X)[0, 1])
