"""Train two models, compare, keep the best."""
import json, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.config import *
from src.features import feature_columns

def time_split(df):
    """Chronological split (past -> train, future -> test). Random split would leak the future."""
    cutoff = df["timestamp"].quantile(1 - TEST_FRACTION)
    return df[df.timestamp <= cutoff], df[df.timestamp > cutoff]

def run(df):
    feats = feature_columns()
    train, test = time_split(df)
    Xtr, ytr, Xte, yte = train[feats], train["failure_within_24h"], test[feats], test["failure_within_24h"]
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
    }
    results, probas = {}, {}
    for name, m in models.items():
        print(f"  training {name} ...")
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]; pred = (p >= 0.5).astype(int)
        probas[name] = p
        results[name] = dict(accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred, zero_division=0),
                             recall=recall_score(yte, pred), f1=f1_score(yte, pred), roc_auc=roc_auc_score(yte, p))
        print("   ", {k: round(v, 3) for k, v in results[name].items()})
    best = max(results, key=lambda n: results[n]["f1"])
    print(f"  ✔ best model: {best}")
    ensure_dirs()
    joblib.dump({"model": models[best], "features": feats, "name": best}, MODEL_PATH)
    test = test.copy(); test["risk"] = probas[best]
    test[["machine_id", "timestamp", "risk", "failure_within_24h"]].to_csv(PRED_CSV, index=False)
    METRICS_JSON.write_text(json.dumps({"best": best, "results": results,
                                        "train_rows": len(train), "test_rows": len(test)}, indent=2))
    return best, results, models, test, probas[best]
