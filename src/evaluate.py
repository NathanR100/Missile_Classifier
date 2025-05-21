import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix

# === Load Validation Data with labels for threshold tuning ===
X_val = np.load("models/X_val.npy")  # validation input sequences
y_val = np.load("models/y_val.npy")  # validation labels (known reentry_phase)

# === Load Test Data (no labels) and Model ===
X_test = np.load("models/X_test.npy")
model = load_model("models/final_model.keras")

# === Function to evaluate baseline on labeled data ===
altitude_index = 3

def evaluate_baseline(y_true, X, threshold):
    altitudes = X[:, -1, altitude_index]
    preds = (altitudes < threshold).astype(int)
    acc = accuracy_score(y_true.flatten(), preds.flatten())
    prec = precision_score(y_true.flatten(), preds.flatten())
    rec = recall_score(y_true.flatten(), preds.flatten())
    f1 = f1_score(y_true.flatten(), preds.flatten())
    return acc, prec, rec, f1

# === Find best threshold on Validation ===
thresholds = [0.6, 0.7, 0.75, 0.8, 0.85]
best_f1 = 0
best_threshold = None

print("Baseline Threshold Tuning on Validation Set:")
for threshold in thresholds:
    acc, prec, rec, f1 = evaluate_baseline(y_val, X_val, threshold)
    print(f"Threshold: {threshold}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print("----")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best Threshold chosen: {best_threshold} with F1: {best_f1:.4f}\n")

# === Predict on Test Data with Model ===
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)
y_pred_flat = y_pred.flatten()

# === Baseline predictions on Test Data using best threshold ===
altitudes_test = X_test[:, -1, altitude_index]
baseline_preds = (altitudes_test < best_threshold).astype(int)
baseline_preds_flat = baseline_preds.flatten()

# === Compare model and baseline predictions on Test ===
agreement = accuracy_score(y_pred_flat, baseline_preds_flat)
kappa = cohen_kappa_score(y_pred_flat, baseline_preds_flat)
cm = confusion_matrix(y_pred_flat, baseline_preds_flat)

print("Comparison of Model vs Baseline on Test Set (no labels):")
print(f"Prediction Agreement: {agreement:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Confusion Matrix:\n{cm}")
print("----")

# === Save predictions for later analysis ===
import os
os.makedirs("models", exist_ok=True)
np.save("models/y_pred.npy", y_pred)
np.save("models/baseline_preds.npy", baseline_preds)

print("Saved model predictions to models/y_pred.npy")
print("Saved baseline predictions to models/baseline_preds.npy")
