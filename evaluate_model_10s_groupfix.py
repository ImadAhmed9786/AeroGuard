from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA = Path("data/real_training_dataset_10s.csv")
MODEL_OUT = Path("aeroguard_model_10s_groupfix_candidate.pkl")

FEATURES = [
    "altitude_mean", "altitude_std", "altitude_change",
    "velocity_mean", "velocity_std", "velocity_change",
    "heading_mean", "heading_std",
    "vertical_rate_mean", "vertical_rate_std",
    "vertical_rate_min", "vertical_rate_max",
]

df = pd.read_csv(DATA).dropna(
    subset=FEATURES + ["label", "group_id"]
).copy()

# Recover the ID after removing the old label prefix.
df["source_id"] = df["group_id"].str.replace(
    r"^(normal_|emergency_)", "", regex=True
)

# Any ID with an emergency-prefixed group is an emergency-flight ID.
emergency_ids = set(
    df.loc[
        df["group_id"].str.startswith("emergency_"),
        "source_id",
    ]
)

# Keep pre- and post-declaration windows from the same emergency flight
# in the same split. Keep ordinary normal-source groups separate.
df["split_group"] = [
    f"emergency:{sid}" if sid in emergency_ids
    else f"normal:{sid}"
    for sid in df["source_id"]
]

X = df[FEATURES]
y = df["label"].astype(int)
groups = df["split_group"]

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

print("Total windows:", len(df))
print("Underlying emergency IDs with both labels:", len(
    set(df.loc[df["group_id"].str.startswith("emergency_"), "source_id"])
    & set(df.loc[df["group_id"].str.startswith("normal_"), "source_id"])
))
print("Training windows:", len(train_idx))
print("Testing windows:", len(test_idx))
print("Training groups:", len(train_groups))
print("Testing groups:", len(test_groups))
print("Group overlap:", len(train_groups & test_groups))
print("\nTest class counts:")
print(y_test.value_counts().sort_index().to_string())

if len(train_groups & test_groups) != 0:
    raise RuntimeError("Flight-group leakage detected.")
if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise RuntimeError("Train or test split does not contain both classes.")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nCorrected grouped evaluation")
print("Accuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%")
print(classification_report(
    y_test,
    pred,
    labels=[0, 1],
    target_names=["Normal", "Emergency"],
    zero_division=0,
))
print("Confusion matrix [normal, emergency]:")
print(confusion_matrix(y_test, pred, labels=[0, 1]))

joblib.dump(model, MODEL_OUT)
print("\nSaved separate candidate:", MODEL_OUT)
print("Live model and previous candidates remain unchanged.")
