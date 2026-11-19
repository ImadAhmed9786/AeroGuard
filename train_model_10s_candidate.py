from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA = Path("data/real_training_dataset_10s.csv")
MODEL_OUT = Path("aeroguard_model_10s_candidate.pkl")

FEATURES = [
    "altitude_mean", "altitude_std", "altitude_change",
    "velocity_mean", "velocity_std", "velocity_change",
    "heading_mean", "heading_std",
    "vertical_rate_mean", "vertical_rate_std",
    "vertical_rate_min", "vertical_rate_max",
]

df = pd.read_csv(DATA).dropna(subset=FEATURES + ["label", "group_id"])
X = df[FEATURES]
y = df["label"].astype(int)
groups = df["group_id"].astype(str)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

print("Rows:", len(df))
print("Train windows:", len(train_idx))
print("Test windows:", len(test_idx))
print("Train groups:", len(train_groups))
print("Test groups:", len(test_groups))
print("Group overlap:", len(train_groups & test_groups))
print("\nTest label counts:")
print(y_test.value_counts().sort_index().to_string())

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%")
print(classification_report(
    y_test, pred, labels=[0, 1],
    target_names=["Normal", "Emergency"],
    zero_division=0,
))
print("Confusion matrix [normal, emergency]:")
print(confusion_matrix(y_test, pred, labels=[0, 1]))

joblib.dump(model, MODEL_OUT)
print("\nCandidate saved:", MODEL_OUT)
print("Live model unchanged.")
