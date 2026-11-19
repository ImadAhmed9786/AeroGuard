import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_FILE = "data/real_training_dataset_v2.csv"
MODEL_FILE = "aeroguard_model_v2.pkl"

FEATURES = [
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

print("Loading AeroGuard V2 training data...")

df = pd.read_csv(DATA_FILE)

print("Total samples:", len(df))

X = df[FEATURES]
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

print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training groups:", groups.iloc[train_idx].nunique())
print("Testing groups:", groups.iloc[test_idx].nunique())

print()
print("Training AeroGuard V2 AI...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

joblib.dump(model, MODEL_FILE)

print("Model saved as:", MODEL_FILE)

# --------------------------------------------------
# Independent test
# --------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print()
print("======================================")
print("AEROGUARD V2 INDEPENDENT TEST")
print("======================================")

print(f"Training Accuracy: {model.score(X_train, y_train) * 100:.2f}%")
print(f"Testing Accuracy: {accuracy * 100:.2f}%")

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Emergency"]
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# --------------------------------------------------
# Group cross validation
# --------------------------------------------------

print()
print("======================================")
print("AEROGUARD V2 GROUP CROSS VALIDATION")
print("======================================")

group_kfold = GroupKFold(n_splits=5)

scores = cross_val_score(
    model,
    X,
    y,
    groups=groups,
    cv=group_kfold,
    scoring="accuracy",
    n_jobs=-1
)

for i, score in enumerate(scores, 1):
    print(f"Fold {i}: {score * 100:.2f}%")

print(f"Average: {scores.mean() * 100:.2f}%")
print(f"Std: {scores.std() * 100:.2f}%")

# --------------------------------------------------
# Feature importance
# --------------------------------------------------

print()
print("======================================")
print("FEATURE IMPORTANCE")
print("======================================")

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

for _, row in importance.iterrows():
    print(
        f"{row['feature']:<25} "
        f"{row['importance'] * 100:.2f}%"
    )