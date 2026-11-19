import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# settings

FILE = "data/real_training_dataset.csv"


# load data

print("Loading AeroGuard training data...")

df = pd.read_csv(FILE)

print("Total samples:", len(df))


# features

X = df.drop(columns=["label", "group_id"])
y = df["label"]
groups = df["group_id"]


# split unseen groups

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print(
    "Training groups:",
    groups.iloc[train_index].nunique()
)

print(
    "Testing groups:",
    groups.iloc[test_index].nunique()
)


# model

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# train

print("\nTraining AeroGuard AI...")

model.fit(X_train, y_train)
import joblib

joblib.dump(model, "aeroguard_model.pkl")

print("Model saved as: aeroguard_model.pkl")


# predictions

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)


# accuracy

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)


# result

print("\n======================================")
print("AEROGUARD INDEPENDENT TEST")
print("======================================")

print(
    f"Training Accuracy: {train_accuracy * 100:.2f}%"
)

print(
    f"Testing Accuracy: {test_accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Normal",
            "Emergency"
        ]
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)