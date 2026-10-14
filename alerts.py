"""Turn risk scores into human-readable maintenance alerts."""
import pandas as pd
from src.config import *

def level(r):
    return "CRITICAL" if r >= CRIT_THRESHOLD else "WARNING" if r >= WARN_THRESHOLD else "OK"

def generate(pred):
    a = pred.copy(); a["level"] = a["risk"].apply(level)
    a = a[a.level != "OK"].copy()
    a["message"] = a.apply(lambda r: f"{r.machine_id}: {r.risk:.0%} failure risk in next {HORIZON}h – "
                           + ("stop & inspect now" if r.level == "CRITICAL" else "schedule maintenance"), axis=1)
    a.to_csv(ALERT_CSV, index=False)
    print(f"  {len(a)} alerts saved -> outputs/alerts.csv  (critical: {(a.level=='CRITICAL').sum()})")
    print(a.sort_values("risk", ascending=False).head(5)[["timestamp", "message"]].to_string(index=False))
    return a
