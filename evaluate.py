"""All charts. Dark industrial theme to match the dashboard."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve
from src.config import *
from src.features import feature_columns

T = THEME
plt.rcParams.update({"figure.facecolor": T["bg"], "axes.facecolor": T["panel"], "axes.edgecolor": T["border"],
    "axes.labelcolor": T["text"], "xtick.color": T["muted"], "ytick.color": T["muted"], "text.color": T["text"],
    "axes.titlecolor": T["amber"], "grid.color": T["border"], "axes.grid": True, "font.size": 10})

def _save(name):
    plt.tight_layout(); plt.savefig(IMG_DIR / name, dpi=150, facecolor=T["bg"]); plt.close()
    print(f"   saved images/{name}")

def sensor_degradation(df):
    m = df[df.machine_id == "M01"]
    fail_idx = m.index[m.failure_event == 1][0]
    w = m.loc[fail_idx - 150: fail_idx]
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(w.timestamp, w.temperature, color=T["amber"]); ax[0].set_ylabel("Temperature (°C)")
    ax[1].plot(w.timestamp, w.vibration, color=T["blue"]); ax[1].set_ylabel("Vibration (mm/s)")
    for a in ax:
        a.axvspan(w.timestamp.iloc[-HORIZON], w.timestamp.iloc[-1], color=T["red"], alpha=.2)
        a.axvline(w.timestamp.iloc[-1], color=T["red"], ls="--")
    ax[0].set_title("Sensor drift before a failure (red zone = last 24 h, machine M01)")
    _save("01_sensor_degradation.png")

def model_comparison(results):
    df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1"]]
    ax = df.plot(kind="bar", figsize=(8, 4.5), color=[T["blue"], T["green"], T["amber"], T["red"]], rot=0)
    ax.set_ylim(0, 1.05); ax.set_title("Model comparison"); ax.legend(facecolor=T["panel"], edgecolor=T["border"])
    _save("02_model_comparison.png")

def confusion(y, p):
    cm = confusion_matrix(y, (p >= .5).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4.5)); ax.imshow(cm, cmap="YlOrBr"); ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center", fontsize=16, color="black", weight="bold")
    ax.set_xticks([0, 1], ["Normal", "Failure"]); ax.set_yticks([0, 1], ["Normal", "Failure"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    _save("03_confusion_matrix.png")

def roc(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(5.5, 4.5)); plt.plot(fpr, tpr, color=T["amber"], lw=2)
    plt.plot([0, 1], [0, 1], color=T["muted"], ls="--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curve")
    _save("04_roc_curve.png")

def importance(rf):
    imp = pd.Series(rf.feature_importances_, index=feature_columns()).nlargest(12)[::-1]
    plt.figure(figsize=(8, 5)); plt.barh(imp.index, imp.values, color=T["amber"])
    plt.title("Top 12 features (Random Forest)"); _save("05_feature_importance.png")

def timeline(test, risk):
    t = test.assign(risk=risk)
    mid = t[t.failure_event == 1].machine_id.iloc[0] if (t.failure_event == 1).any() else t.machine_id.iloc[0]
    m = t[t.machine_id == mid]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(m.timestamp, m.risk, color=T["blue"], lw=1.5, label="Predicted failure risk")
    ax.axhline(WARN_THRESHOLD, color=T["amber"], ls="--", label="Warning"); ax.axhline(CRIT_THRESHOLD, color=T["red"], ls="--", label="Critical")
    ax.fill_between(m.timestamp, 0, 1, where=m.failure_within_24h == 1, color=T["red"], alpha=.2, label="Actual 24 h danger zone")
    ax.set_ylim(0, 1.02); ax.set_ylabel("Risk"); ax.set_title(f"Failure prediction timeline – {mid}")
    ax.legend(facecolor=T["panel"], edgecolor=T["border"], loc="upper left"); _save("06_failure_prediction_timeline.png")

def run(df, test, best, results, models, risk):
    y = test["failure_within_24h"]
    sensor_degradation(df); model_comparison(results); confusion(y, risk); roc(y, risk)
    importance(models["Random Forest"]); timeline(test, risk)
