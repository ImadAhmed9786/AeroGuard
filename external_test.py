
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==================================================
# AEROGUARD EXTERNAL EVALUATION
# ==================================================

NPZ_FILE = "DASHlink_full_fourclass_raw_comp.npz"
MODEL_FILE = "aeroguard_model_10.pkl"

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

print("=" * 55)
print("AEROGUARD EXTERNAL MODEL EVALUATION")
print("=" * 55)

# Load the actual live model
print("\nLoading deployed model...")
model = joblib.load(MODEL_FILE)

if model.n_features_in_ != len(FEATURES):
    raise ValueError(
        f"Expected {model.n_features_in_} features, "
        f"but evaluation defines {len(FEATURES)}."
    )

if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != FEATURES:
        raise ValueError(
            "Feature names/order do not match the trained model."
        )

print("Model:", MODEL_FILE)
print("Features:", model.n_features_in_)
print("Classes:", model.classes_)

# Load the complete NASA dataset
print("\nLoading NASA dataset...")
data = np.load(NPZ_FILE)

X_nasa = data["data.npy"]
y_nasa = data["label.npy"]

print("Dataset shape:", X_nasa.shape)
print("Total sequences:", len(y_nasa))

print("\nNASA label distribution:")
print(pd.Series(y_nasa).value_counts().sort_index())

# NASA labels:
# 0 = nominal
# 1, 2, 3 = anomaly-associated classes
true_labels = (y_nasa != 0).astype(int)

# NASA data is sampled at approximately 1-second intervals.
# Sample every 10 points to approximate AeroGuard's cadence.
sampled = X_nasa[:, ::10, :]

# Use the latest 10 sampled observations, matching the
# live model's 10-observation feature-window length.
window = sampled[:, -10:, :]

# Extract the same telemetry variables as the previous test.
altitude = window[:, :, 3] * 0.3048
velocity = window[:, :, 4] * 0.514444
heading = window[:, :, 17]

# Calculate vertical rate in metres per second.
vertical_rate = np.diff(altitude, axis=1) / 10.0

# Match the live feature names and population standard deviation.
features = pd.DataFrame({
    "altitude_mean": np.mean(altitude, axis=1),
    "altitude_std": np.std(altitude, axis=1, ddof=0),
    "altitude_change": altitude[:, -1] - altitude[:, 0],

    "velocity_mean": np.mean(velocity, axis=1),
    "velocity_std": np.std(velocity, axis=1, ddof=0),
    "velocity_change": velocity[:, -1] - velocity[:, 0],

    "heading_mean": np.mean(heading, axis=1),
    "heading_std": np.std(heading, axis=1, ddof=0),

    "vertical_rate_mean": np.mean(vertical_rate, axis=1),
    "vertical_rate_std": np.std(vertical_rate, axis=1, ddof=0),
    "vertical_rate_min": np.min(vertical_rate, axis=1),
    "vertical_rate_max": np.max(vertical_rate, axis=1)
}, columns=FEATURES)

# Reject invalid rows instead of silently evaluating them.
valid = (
    np.isfinite(features.to_numpy()).all(axis=1)
    & np.isfinite(true_labels)
)

features = features.loc[valid].reset_index(drop=True)
true_labels = true_labels[valid]

print("\nValid sequences:", len(features))
print("Excluded invalid sequences:", len(y_nasa) - len(features))

if len(features) == 0:
    raise ValueError("No valid NASA sequences remain.")

print("\nRunning deployed model...")
predictions = model.predict(features).astype(int)

# Metrics
accuracy = accuracy_score(true_labels, predictions)
tn, fp, fn, tp = confusion_matrix(
    true_labels, predictions, labels=[0, 1]
).ravel()

fpr = fp / (fp + tn) if (fp + tn) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
precision = tp / (tp + fp) if (tp + fp) else 0.0

print("\n" + "=" * 55)
print("AEROGUARD EVALUATION RESULTS")
print("=" * 55)

print(f"Model tested       : {MODEL_FILE}")
print(f"Sequences tested   : {len(features)}")
print(f"Accuracy            : {accuracy * 100:.2f}%")
print(f"Normal false alarms : {fp}")
print(f"False-positive rate : {fpr * 100:.2f}%")
print(f"Anomaly detections  : {tp}")
print(f"Anomaly misses      : {fn}")
print(f"Anomaly recall      : {recall * 100:.2f}%")
print(f"Anomaly precision   : {precision * 100:.2f}%")

print("\nConfusion matrix:")
print("[[True normal, False alarms],")
print(" [Missed anomalies, Detected anomalies]]")
print(confusion_matrix(true_labels, predictions, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    true_labels,
    predictions,
    labels=[0, 1],
    target_names=["Normal", "Anomaly-associated"],
    zero_division=0
))

print("\nEvaluation complete.")
print(
    "Note: NASA anomaly labels do not necessarily mean "
    "confirmed aircraft emergencies."
)
print(
    "This evaluates one 10-observation window per sequence; "
    "it does not establish certified aviation performance."
)
