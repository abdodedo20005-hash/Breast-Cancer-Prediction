"""Rebuild the missing StandardScaler and verify the model + class mapping.

Run from the project root (with the .venv active):  python fix_scaler.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features import FEATURES


def norm(c):
    return str(c).strip().lower().replace(" ", "_")


model = joblib.load("models/final_breast_cancer_model.pkl")

raw = pd.read_csv("data/data.csv")
raw.columns = [norm(c) for c in raw.columns]
clean = pd.read_csv("data/breast_cancer_cleaned.csv")
clean.columns = [norm(c) for c in clean.columns]

print("cleaned csv has same columns/order as FEATURES:", list(clean.columns) == FEATURES)
print("cleaned csv  mean(max abs):", round(float(clean.mean().abs().max()), 4),
      "| std(avg):", round(float(clean.std(ddof=0).mean()), 4))
print("raw csv      mean(avg):", round(float(raw[FEATURES].mean().mean()), 2))

X_raw = raw[FEATURES].astype(float).values
y = (raw["diagnosis"].astype(str).str.strip() == "M").astype(int).values  # 1 = malignant

scaler = StandardScaler().fit(X_raw)
scaled = scaler.transform(X_raw)
diff = float(np.abs(scaled - clean.values).max())
print("max |scaler(raw) - cleaned csv|:", round(diff, 6), "->", "MATCH" if diff < 1e-3 else "NO MATCH")

pred_raw = model.predict(X_raw)
pred_scaled = model.predict(scaled)
print("\nAccuracy WITHOUT scaling : (1=malignant) %.3f | (0=malignant) %.3f"
      % ((pred_raw == y).mean(), (pred_raw == 1 - y).mean()))
acc1 = (pred_scaled == y).mean()
acc0 = (pred_scaled == 1 - y).mean()
print("Accuracy WITH scaling    : (1=malignant) %.3f | (0=malignant) %.3f" % (acc1, acc0))

best = max(acc1, acc0)
if best >= 0.9:
    joblib.dump(scaler, "models/scaler.joblib")
    print("\nSaved models/scaler.joblib")
    print("MALIGNANT_CLASS should be:", "1" if acc1 >= acc0 else "0")
else:
    print("\nAccuracy is still low, scaler NOT saved. Send me this output.")
