"""ONE command runs everything:   python runall.py
Steps: simulate -> clean+features -> train -> charts -> alerts -> (launch dashboard API)
Options:  --no-api   skip the dashboard      --port 5000   change port
"""
import argparse, time
from src import simulate_data, features, train, evaluate, alerts
from src.config import *

def step(n, title):
    print(f"\n[{n}/5] {title}"); return time.time()

def run_pipeline():
    ensure_dirs()
    t = step(1, "Simulating IoT sensor data"); df_raw = simulate_data.generate()
    print(f"  {len(df_raw):,} readings from {N_MACHINES} machines -> data/sensor_data.csv ({time.time()-t:.1f}s)")
    t = step(2, "Cleaning + feature engineering"); df = features.build(df_raw)
    print(f"  {df.shape[1]} columns -> data/processed_features.csv")
    t = step(3, "Training models"); best, results, models, test, risk = train.run(df)
    t = step(4, "Creating charts"); evaluate.run(df, test, best, results, models, risk)
    t = step(5, "Generating alerts"); alerts.generate(test.assign(risk=risk)[["machine_id", "timestamp", "risk"]])
    print("\n✅ Pipeline finished. See images/, outputs/, models/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--no-api", action="store_true"); ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    run_pipeline()
    if not a.no_api:
        from src.api import app
        print(f"\n🚀 Dashboard: http://127.0.0.1:{a.port}   (Ctrl+C to stop)")
        app.run(host="127.0.0.1", port=a.port)
