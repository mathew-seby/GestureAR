import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay)

warnings.filterwarnings("ignore", category=FutureWarning)
os.makedirs("models", exist_ok=True)

BLOCK = 60
rng = np.random.default_rng(42)

df = pd.read_csv("data/gestures.csv")
X = df.drop(columns="label").values
y = df["label"].values
print(df["label"].value_counts(), "\n")

train_idx, test_idx = [], []
for label in df["label"].unique():
    idx = df.index[df["label"] == label].to_numpy()
    blocks = [idx[i:i + BLOCK] for i in range(0, len(idx), BLOCK)]
    order = rng.permutation(len(blocks))
    n_test = max(1, int(len(blocks) * 0.2))
    for rank, b in enumerate(order):
        (test_idx if rank < n_test else train_idx).extend(blocks[b])

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples\n")

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced",
            probability=True, random_state=42)),
    "Neural Network": make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                      random_state=42)),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = acc
    print(f"{name}: {acc * 100:.2f}%")

best_name = max(results, key=results.get)
pred = models[best_name].predict(X_test)

print(f"\nBest model: {best_name}\n")
print(classification_report(y_test, pred))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, cmap="Blues")
ax.set_title(f"Confusion matrix: {best_name}")
plt.tight_layout()
plt.savefig(os.path.join("models", "confusion_matrix.png"), dpi=150)

final_model = clone(models[best_name]).fit(X, y)
joblib.dump(final_model, os.path.join("models", "gesture_model.pkl"))
print(f"Final {best_name} trained on all {len(X)} samples and saved.")