"""Virtual IoT sensors: generates realistic machine data with degradation before failures."""
import numpy as np
import pandas as pd
from src.config import *

def simulate_machine(machine_id, n_steps, rng):
    """One machine = repeated life cycles: healthy -> degrading -> failure -> repaired."""
    base = dict(  # each machine has slightly different 'normal' values
        temperature=70 + rng.normal(0, 3), vibration=2.0 + rng.normal(0, .2),
        pressure=100 + rng.normal(0, 4), rpm=1500 + rng.normal(0, 30),
        current=12 + rng.normal(0, .6))
    frames, t = [], 0
    start = pd.Timestamp("2025-01-01")
    while t < n_steps:
        life = int(rng.integers(250, 600))
        fails = t + life <= n_steps
        if not fails:
            life = n_steps - t
        steps = np.arange(life)
        ttf = (life - steps) if fails else np.full(life, 10**6)   # hours to failure
        d = np.clip(1 - ttf / 72, 0, 1) ** 1.5                    # degradation 0..1 (last 72h)
        daily = np.sin(2 * np.pi * (t + steps) / 24)              # day/night load swing
        df = pd.DataFrame({
            "machine_id": f"M{machine_id:02d}",
            "timestamp": start + pd.to_timedelta(t + steps, unit="h"),
            "temperature": base["temperature"] + 2 * daily + rng.normal(0, 2.0, life) + 22 * d,
            "vibration": base["vibration"] + .15 * daily + rng.normal(0, .35, life) + 2.8 * d,
            "pressure": base["pressure"] + rng.normal(0, 4, life) - 16 * d,
            "rpm": base["rpm"] + rng.normal(0, 25, life) - 110 * d,
            "current": base["current"] + .3 * daily + rng.normal(0, .6, life) + 3.5 * d,
            "failure_within_24h": ((ttf <= HORIZON) & fails).astype(int),
            "failure_event": 0,
        })
        if fails:
            df.loc[df.index[-1], "failure_event"] = 1
        frames.append(df)
        t += life
    return pd.concat(frames, ignore_index=True)

def generate(save=True):
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.concat([simulate_machine(i + 1, N_STEPS, rng) for i in range(N_MACHINES)],
                   ignore_index=True)
    # real sensors drop readings -> add ~1% missing values so cleaning has a job
    mask = rng.random((len(df), len(SENSORS))) < 0.01
    df[SENSORS] = df[SENSORS].mask(mask)
    if save:
        ensure_dirs(); df.to_csv(RAW_CSV, index=False)
    return df

if __name__ == "__main__":
    d = generate()
    print(d.head()); print("rows:", len(d), "| failure-window rate:", d.failure_within_24h.mean().round(3))
