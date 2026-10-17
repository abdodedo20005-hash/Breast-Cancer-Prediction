"""Honest evaluation: 5-fold stratified CV, scaler fitted inside each fold.

Run from the project root:   pip install matplotlib   (once)
                             python evaluate.py

It saves docs/confusion_matrix.png + docs/roc_curve.png and fills the results
table in README.md (between the RESULTS markers).
"""
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURES

MODEL = "models/final_breast_cancer_model.pkl"
SEED = 42

raw = pd.read_csv("data/data.csv")
raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
X = raw[FEATURES].astype(float).values
y = (raw["diagnosis"].astype(str).str.strip() == "M").astype(int).values  # 1 = malignant

# Same hyper-parameters as the trained model, but refit inside every fold.
pipe = make_pipeline(StandardScaler(), clone(joblib.load(MODEL)))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


def metrics(th):
    p = (proba >= th).astype(int)
    return (
        accuracy_score(y, p),
        recall_score(y, p),
        precision_score(y, p),
        f1_score(y, p),
    )


acc, rec, prec, f1 = metrics(0.5)
auc = roc_auc_score(y, proba)
print(f"Accuracy {acc:.3f} | Recall {rec:.3f} | Precision {prec:.3f} | F1 {f1:.3f} | ROC-AUC {auc:.3f}")

Path("docs").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay.from_predictions(
    y, (proba >= 0.5).astype(int), display_labels=["Benign", "Malignant"], cmap="Blues", ax=ax
)
ax.set_title("Confusion matrix (5-fold CV)")
fig.tight_layout()
fig.savefig("docs/confusion_matrix.png", dpi=150)

fig, ax = plt.subplots(figsize=(4.5, 4))
RocCurveDisplay.from_predictions(y, proba, name=f"RandomForest (AUC={auc:.3f})", ax=ax)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_title("ROC curve (5-fold CV)")
fig.tight_layout()
fig.savefig("docs/roc_curve.png", dpi=150)

rows = "\n".join(
    "| %.1f | %.1f%% | %.1f%% |" % (th, metrics(th)[1] * 100, metrics(th)[2] * 100)
    for th in (0.5, 0.4, 0.3)
)
block = f"""<!-- RESULTS:START -->
| Metric | Value |
|---|---|
| Accuracy | {acc:.1%} |
| Recall (malignant) | {rec:.1%} |
| Precision (malignant) | {prec:.1%} |
| F1 (malignant) | {f1:.1%} |
| ROC-AUC | {auc:.3f} |

Lowering the decision threshold trades precision for recall:

| Threshold | Recall (malignant) | Precision (malignant) |
|---|---|---|
{rows}
<!-- RESULTS:END -->"""

readme = Path("README.md")
text = readme.read_text(encoding="utf-8")
new, n = re.subn(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda m: block, text, flags=re.S)
if n:
    readme.write_text(new, encoding="utf-8")
    print("README.md results table updated.")
else:
    print("RESULTS markers not found in README.md; paste this block manually:\n" + block)
