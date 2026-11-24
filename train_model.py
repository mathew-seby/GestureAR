import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay)

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/gestures.csv")
X = df.drop(columns="label").values
y = df["label"].values

train_idx, test_idx = [], []
for label in df["label"].unique():
    idx = df.index[df["label"] == label].to_numpy()
    cut = int(len(idx) * 0.8)
    train_idx.extend(idx[:cut])
    test_idx.extend(idx[cut:])

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
best_model = models[best_name]
pred = best_model.predict(X_test)

print(f"\nBest model: {best_name}\n")
print(classification_report(y_test, pred))

joblib.dump(best_model, os.path.join("models", "gesture_model.pkl"))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, cmap="Blues")
ax.set_title(f"Confusion matrix: {best_name}")
plt.tight_layout()
plt.savefig(os.path.join("models", "confusion_matrix.png"), dpi=150)
print("Saved models/gesture_model.pkl and models/confusion_matrix.png")