import pandas as pd
import numpy as np

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

print("\n======================================")
print("AEROGUARD TRAINING FEATURE RANGES")
print("======================================")

for feature in features:
    print(
        f"{feature:25} "
        f"min={df[feature].min():10.2f} "
        f"max={df[feature].max():10.2f} "
        f"mean={df[feature].mean():10.2f}"
    )

print("\n======================================")
print("DONE")
print("======================================")