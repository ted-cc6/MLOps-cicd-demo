import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

THRESHOLD = 0.80

df = pd.read_csv("data/sample.csv")
X = df.drop("label", axis=1)
y = df["label"]

model = joblib.load("model.joblib")
preds = model.predict(X)

acc = accuracy_score(y, preds)
print(f"Model accuracy: {acc}")

if acc < THRESHOLD:
    raise ValueError(f"Model accuracy {acc} below threshold {THRESHOLD}")
