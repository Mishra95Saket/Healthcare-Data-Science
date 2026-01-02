import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("data/heart.csv")
X = df.drop(columns=["target"])
y = df["target"]

model = joblib.load("data/rf_model.joblib")
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:,1]

print(classification_report(y, y_pred))
print("ROC‑AUC:", roc_auc_score(y, y_proba))
