import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/real_training_dataset.csv")

features = [
    "altitude_mean",
    "altitude_std",
    "altitude_change",
    "velocity_mean",
    "velocity_std",
    "velocity_change",
    "heading_mean",
    "heading_std"
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

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

train_accuracy = accuracy_score(
    y_train,
    model.predict(X_train)
)

test_accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)

print("\n======================================")
print("VERTICAL RATE ABLATION TEST")
print("======================================")

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

print("======================================")