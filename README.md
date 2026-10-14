# ⚙️ AI-Powered Predictive Maintenance System for IoT Devices

Predicts machine failures **24 hours in advance** from simulated IoT sensor data, raises WARNING/CRITICAL alerts, and shows everything on a dark "control-room" web dashboard.

![Failure prediction timeline](images/06_failure_prediction_timeline.png)

## Problem statement
Unplanned machine breakdowns stop production and cost far more than planned repairs. Fixed-schedule maintenance wastes money on healthy machines. This project learns failure patterns from sensor history (temperature, vibration, pressure, RPM, current) and predicts *which machine will fail soon*, so maintenance can be scheduled just in time.

## Industry relevance
Manufacturing plants, power plants, automotive lines, aviation (engine/component health) and any fleet of IoT-connected machines use predictive maintenance to cut downtime, repair cost and safety risk.

## Tech stack
Python 3.10+, Pandas, NumPy, Scikit-learn (Logistic Regression, Random Forest), Matplotlib, Flask (API + dashboard).

## Dataset
No hardware needed: `src/simulate_data.py` creates **40,000 hourly readings from 20 virtual machines**. Each machine runs repeated life cycles; in the last ~72 h before a failure, temperature, vibration and current rise while pressure and RPM drop. Noise, day/night load swings and ~1% missing values make it realistic. Label `failure_within_24h = 1` when failure happens in the next 24 h.
*(You can swap in a real dataset such as NASA C-MAPSS or AI4I 2020 by keeping the same column names.)*

## Architecture
```
Virtual IoT sensors -> Cleaning -> Feature engineering -> Model training -> Risk score
   (simulate_data)     (features)    (rolling stats)       (train)           |
                                                                    Alerts + Charts
                                                                   (alerts, evaluate)
                                                                            |
                                                              Flask API + Dashboard (api)
```

## Installation
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python runall.py            # everything + opens dashboard at http://127.0.0.1:5000
python runall.py --no-api   # pipeline only
python main.py              # same as --no-api
python -m src.api           # dashboard only (after the pipeline has run once)
```
API: `GET /api/machines`, `GET /api/alerts`, `POST /api/predict` (JSON with temperature, vibration, pressure, rpm, current), `GET /api/health`.

## Results (time-based test set = last 20% of the timeline)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.984 | 0.780 | 0.977 | 0.867 |
| **Random Forest** | **0.989** | **0.853** | **0.956** | **0.902** |

Results are on *simulated* data, so they look cleaner than real-world data would. Recall matters most here: a missed failure costs more than a false alarm.

## Screenshots
| | |
|---|---|
| ![](images/01_sensor_degradation.png) | ![](images/03_confusion_matrix.png) |
| ![](images/05_feature_importance.png) | ![](images/04_roc_curve.png) |

*(Add a dashboard screenshot as `images/07_dashboard.png`.)*

## Folder structure
`data/` datasets · `src/` code · `models/` saved model · `outputs/` metrics, predictions, alerts · `images/` charts · `docs/` guide · `notebooks/` experiments · `runall.py` one-command runner

## Learning outcomes
Time-series feature engineering, leakage-free time-based splitting, class imbalance handling, model comparison, alert thresholds, REST API building, GitHub project presentation.

## Limitations & future work
Simulated data; add LSTM, Remaining-Useful-Life regression, real datasets, live streaming (MQTT), Docker.
