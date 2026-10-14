# Project Guide (read this first)

## 1. What is AI predictive maintenance?
**Simple:** Like a car warning light that tells you the engine will have a problem *next week*, not after it breaks. Sensors on a machine send readings; AI spots the pattern that usually appears before a breakdown.
**Technical:** Supervised binary classification on multivariate time-series. Features (rolling mean/std, deltas) are computed per machine; a classifier estimates P(failure within H hours).
**Value:** less downtime, cheaper repairs, longer machine life, safer plants (manufacturing, power, automotive, aviation).

## 2. Tech stack options
| | A – Easiest (chosen) | B – Intermediate | C – Advanced |
|---|---|---|---|
| Tools | Pandas, Scikit-learn, Matplotlib, Flask | + XGBoost, SHAP, Streamlit | + PyTorch/TensorFlow, MLflow, Docker, MQTT |
| Data | Simulated sensors / AI4I 2020 | NASA C-MAPSS | Streaming sensor data |
| Models | Logistic Regression, Random Forest | XGBoost, RUL regression | LSTM / Transformers |
| Difficulty | Easy | Medium | Hard |
| GPU | No | No | Recommended |
Chosen: **A** — runs in under a minute on any laptop, strong proof, easy to explain in interviews.

## 3. Pipeline phases (what / why / verify)
1. **Setup** – venv + requirements. Verify: `python -c "import sklearn"`.
2. **Dataset** – `simulate_data.py`. Verify: `data/sensor_data.csv` has 40,000 rows.
3. **Cleaning** – sort, remove duplicates, interpolate missing values *per machine*. Mistake: filling across machines.
4. **Features** – rolling mean/std (6h, 24h) + hourly change. Mistake: computing across machines (mix-ups).
5. **Model** – Logistic Regression vs Random Forest with `class_weight="balanced"` (failures are rare).
6. **Evaluation** – precision, recall, F1, ROC-AUC. Mistake: trusting accuracy alone.
7. **Prediction** – risk score 0–1, thresholds 0.4 / 0.7 -> WARNING / CRITICAL.
8. **Visualization** – 6 charts in `images/` + dashboard.
9. **GitHub** – commits below.
10. **Final output** – README with screenshots.
Key rule: **split by time** (train on past, test on future); a random split leaks future information.

## 4. Simulation = how failure is represented
1. Each machine gets its own "normal" sensor values.
2. A life cycle lasts 250–600 hours, ending in a failure.
3. In the last 72 h a degradation factor rises 0 -> 1 and shifts the sensors.
4. Noise + day/night swings hide the pattern a little.
5. Rows in the final 24 h are labelled `failure_within_24h = 1`.
6. The model never sees the label's logic — only sensor trends — and learns to predict it.

## 5. GitHub steps
```bash
git init
git add .gitignore requirements.txt README.md
git commit -m "chore: project skeleton, requirements, README draft"
git branch -M main
git remote add origin https://github.com/<your-username>/AI-Predictive-Maintenance-IoT.git
git push -u origin main
```
**Repo name:** `AI-Predictive-Maintenance-IoT`
**Description:** Predicting machine failures 24h ahead from simulated IoT sensor data with Random Forest + Flask dashboard
**Topics:** `predictive-maintenance` `iot` `machine-learning` `scikit-learn` `time-series` `python` `flask` `industry-4-0` `data-science`

## 6. 7-day commit plan
| Day | Work | Commit message | Proof to capture |
|---|---|---|---|
| 1 | Setup, folders, requirements | `chore: add project structure and requirements` | terminal: pip install |
| 2 | Dataset simulator | `feat: add IoT sensor data simulator` | `data` preview (`df.head()`) |
| 3 | Cleaning + features | `feat: add preprocessing and feature engineering` | processed columns output |
| 4 | Model training | `feat: train logistic regression and random forest` | training log |
| 5 | Evaluation | `feat: add metrics, confusion matrix, ROC` | images 02–05 |
| 6 | Alerts, charts, API | `feat: add alerts, timeline chart, Flask dashboard` | dashboard screenshot |
| 7 | runall.py, README, polish | `docs: finalize README and add runall.py` | repo preview |

## 7. Proof checklist (name files like this, store in `images/`)
- [ ] `00_dataset_preview.png` – `head()` of sensor_data.csv
- [ ] `00_preprocessing_output.png` – terminal after step 2
- [ ] `00_training_log.png` – terminal after step 3
- [ ] `01_sensor_degradation.png`, `03_confusion_matrix.png`, `06_failure_prediction_timeline.png` (auto-generated)
- [ ] `07_dashboard.png` – browser screenshot
- [ ] `08_github_repo.png` – repo home page
- [ ] README shows at least 4 images; `outputs/metrics.json` committed
- [ ] LinkedIn post: problem -> approach -> result (F1 0.90) -> GitHub link
