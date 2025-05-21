# Feature index for altitude in your input (0=relative_time, 1=lat, 2=lon, 3=altitude, 4=rad)
altitude_index = 3
altitudes = X_val[:, -1, altitude_index]

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_baseline(threshold):
    preds = (altitudes < threshold).astype(int)
    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds)
    rec = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    return acc, prec, rec, f1

#MinMaxScaling
thresholds = [0.6, 0.7, 0.75, 0.8, 0.85]

for threshold in thresholds:
    acc, prec, rec, f1 = evaluate_baseline(threshold)
    print(f"Threshold: {threshold}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print("----")
