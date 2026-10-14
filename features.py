"""Cleaning + feature engineering."""
import pandas as pd
from src.config import *

def clean(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(["machine_id", "timestamp"]).sort_values(["machine_id", "timestamp"])
    df = df.reset_index(drop=True)
    # fill missing sensor readings from neighbouring readings of the SAME machine
    df[SENSORS] = df.groupby("machine_id")[SENSORS].transform(
        lambda s: s.interpolate(limit_direction="both"))
    return df

def add_features(df):
    """Rolling mean/std + change-since-last-hour, computed per machine."""
    df = df.copy()
    for s in SENSORS:
        g = df.groupby("machine_id")[s]
        df[f"{s}_diff"] = g.diff().fillna(0)
        for w in WINDOWS:
            r = g.rolling(w, min_periods=1)
            df[f"{s}_mean{w}"] = r.mean().reset_index(level=0, drop=True)
            df[f"{s}_std{w}"] = r.std().reset_index(level=0, drop=True).fillna(0)
    return df

def feature_columns():
    cols = list(SENSORS)
    for s in SENSORS:
        cols.append(f"{s}_diff")
        for w in WINDOWS:
            cols += [f"{s}_mean{w}", f"{s}_std{w}"]
    return cols

def build(df_raw, save=True):
    df = add_features(clean(df_raw))
    if save:
        ensure_dirs(); df.to_csv(PROC_CSV, index=False)
    return df
