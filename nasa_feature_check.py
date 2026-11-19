import numpy as np
import pandas as pd

NPZ_FILE = "DASHlink_full_fourclass_raw_comp.npz"

print("Loading NASA dataset...")
data = np.load(NPZ_FILE)

X = data["data.npy"]
y = data["label.npy"]

# use 5000 normal + all anomalies
normal = np.where(y == 0)[0][:5000]
anomaly = np.where(y != 0)[0]

indices = np.concatenate([normal, anomaly])

X = X[indices]
y = y[indices]

# NASA variables
altitude = X[:, :, 3] * 0.3048
velocity = X[:, :, 4] * 0.514444
heading = X[:, :, 17]

# 10-second sampling
altitude = altitude[:, ::10]
velocity = velocity[:, ::10]
heading = heading[:, ::10]

# vertical rate
vertical_rate = np.diff(altitude, axis=1) / 10.0

features = pd.DataFrame({
    "altitude_mean": np.mean(altitude, axis=1),
    "altitude_std": np.std(altitude, axis=1),
    "altitude_change": altitude[:, -1] - altitude[:, 0],

    "velocity_mean": np.mean(velocity, axis=1),
    "velocity_std": np.std(velocity, axis=1),
    "velocity_change": velocity[:, -1] - velocity[:, 0],

    "heading_mean": np.mean(heading, axis=1),
    "heading_std": np.std(heading, axis=1),

    "vertical_rate_mean": np.mean(vertical_rate, axis=1),
    "vertical_rate_std": np.std(vertical_rate, axis=1),
    "vertical_rate_min": np.min(vertical_rate, axis=1),
    "vertical_rate_max": np.max(vertical_rate, axis=1)
})

# training ranges
training = pd.read_csv("data/real_training_dataset.csv")

feature_names = list(features.columns)

print("\n======================================")
print("FEATURE RANGE COMPARISON")
print("======================================")

for feature in feature_names:
    train_min = training[feature].min()
    train_max = training[feature].max()

    nasa_min = features[feature].min()
    nasa_max = features[feature].max()

    inside = (
        (features[feature] >= train_min) &
        (features[feature] <= train_max)
    ).mean() * 100

    print(f"\n{feature}")
    print(f"Training : {train_min:.2f} to {train_max:.2f}")
    print(f"NASA     : {nasa_min:.2f} to {nasa_max:.2f}")
    print(f"NASA inside training range: {inside:.2f}%")

print("\n======================================")
print("DIAGNOSTIC COMPLETE")
print("======================================")