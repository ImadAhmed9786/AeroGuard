import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

DATASET_FILE = "data/real_training_dataset_10.csv"

FEATURES = [
    "altitude_mean", "altitude_std", "altitude_change",
    "velocity_mean", "velocity_std", "velocity_change",
    "heading_mean", "heading_std",
    "vertical_rate_mean", "vertical_rate_std",
    "vertical_rate_min", "vertical_rate_max"
]

print("Loading dataset...")
df = pd.read_csv(DATASET_FILE)
X = df[FEATURES]
y = df["label"]
groups = df["group_id"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

print(f"Training rows: {len(train_idx)}")
print(f"Testing rows:  {len(test_idx)}")
print(f"Training groups: {len(train_groups)}")
print(f"Testing groups:  {len(test_groups)}")
print(f"Group overlap: {len(train_groups & test_groups)}")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

tn, fp, fn, tp = confusion_matrix(
    y_test, pred, labels=[0, 1]
).ravel()

print("\nGROUP-BASED EVALUATION")
print("=" * 45)
print(f"Accuracy:          {accuracy_score(y_test, pred) * 100:.2f}%")
print(f"Emergency precision: {tp / (tp + fp) * 100 if tp + fp else 0:.2f}%")
print(f"Emergency recall:    {tp / (tp + fn) * 100 if tp + fn else 0:.2f}%")
print(f"Normal false-positive rate: {fp / (fp + tn) * 100 if fp + tn else 0:.2f}%")
print(f"True negatives: {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives: {tp}")
print("\nClassification report:")
print(classification_report(
    y_test, pred, labels=[0, 1],
    target_names=["Normal", "Emergency"],
    zero_division=0
))
print("Saved production model was not modified.")
