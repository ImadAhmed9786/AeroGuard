from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

DATA_FILE = Path("data/real_training_dataset_10_clean.csv")
MODEL_FILE = Path("aeroguard_model_clean_candidate.pkl")

FEATURES = [
    "altitude_mean", "altitude_std", "altitude_change",
    "velocity_mean", "velocity_std", "velocity_change",
    "heading_mean", "heading_std",
    "vertical_rate_mean", "vertical_rate_std",
    "vertical_rate_min", "vertical_rate_max",
]

df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=FEATURES + ["label", "group_id"])

X = df[FEATURES]
y = df["label"].astype(int)
groups = df["group_id"].astype(str)

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

print("Rows:", len(df))
print("Training windows:", len(train_idx))
print("Testing windows:", len(test_idx))
print("Training groups:", len(train_groups))
print("Testing groups:", len(test_groups))
print("Group overlap:", len(train_groups & test_groups))

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nCandidate model evaluation")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(
    y_test, predictions,
    labels=[0, 1],
    target_names=["Normal", "Emergency"],
    zero_division=0,
))
print("Confusion matrix [normal, emergency]:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# Save to a new filename, never overwrite the live model.
joblib.dump(model, MODEL_FILE)
print("\nCandidate model saved:", MODEL_FILE)
print("Existing aeroguard_model_10.pkl was not changed.")
