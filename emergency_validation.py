import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("Loading AeroGuard training data...")

df = pd.read_csv("data/real_training_dataset.csv")

features = [
    "altitude_mean",
    "altitude_std",
    "altitude_change",
    "velocity_mean",
    "velocity_std",
    "velocity_change",
    "heading_mean",
    "heading_std",
    "vertical_rate_mean",
    "vertical_rate_std",
    "vertical_rate_min",
    "vertical_rate_max"
]

X = df[features]
y = df["label"]
groups = df["group_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("Training groups:", groups_train.nunique())
print("Testing groups:", groups_test.nunique())

print("\nTraining AeroGuard AI...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    pos_label=1
)

recall = recall_score(
    y_test,
    predictions,
    pos_label=1
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label=1
)

matrix = confusion_matrix(
    y_test,
    predictions
)

print("\n======================================")
print("AEROGUARD EMERGENCY VALIDATION")
print("======================================")

print(f"Overall Accuracy : {accuracy * 100:.2f}%")
print(f"Emergency Precision: {precision * 100:.2f}%")
print(f"Emergency Recall   : {recall * 100:.2f}%")
print(f"Emergency F1 Score : {f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(matrix)

print("\n======================================")
print("EMERGENCY TEST RESULTS")
print("======================================")

tn, fp, fn, tp = matrix.ravel()

print(f"Normal correctly identified : {tn}")
print(f"Normal falsely flagged      : {fp}")
print(f"Emergency missed            : {fn}")
print(f"Emergency detected          : {tp}")

print("\n======================================")
print("VALIDATION COMPLETE")
print("======================================")