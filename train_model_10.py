import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# ==========================================
# SETTINGS
# ==========================================

DATASET_FILE = "data/real_training_dataset_10.csv"
MODEL_FILE = "aeroguard_model_10.pkl"


# ==========================================
# LOAD DATA
# ==========================================

print("Loading 10-observation training dataset...")

df = pd.read_csv(DATASET_FILE)

print("Dataset shape:", df.shape)


# ==========================================
# FEATURES
# ==========================================

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

X = df[FEATURES]
y = df["label"]

groups = df["group_id"]


# ==========================================
# GROUP-BASED TRAIN / TEST SPLIT
# ==========================================

print("\nSplitting data by flight/group...")

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


print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==========================================
# TRAIN RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)


# ==========================================
# TEST MODEL
# ==========================================

print("\nTesting model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n======================================")
print("AEROGUARD V1 - 10 OBSERVATION MODEL")
print("======================================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Normal",
            "Emergency"
        ]
    )
)


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

print("\nFeature Importance:")

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(
    ascending=False
)

print(importance)


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    MODEL_FILE
)

print("\n======================================")
print("MODEL SAVED")
print("======================================")

print(MODEL_FILE)
