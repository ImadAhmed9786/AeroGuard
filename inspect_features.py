import pandas as pd

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
    "vertical_rate_max",
    "label"
]

print("======================================")
print("NORMAL SAMPLES")
print("======================================")

print(
    df[df["label"] == 0][features]
    .head(10)
    .to_string(index=False)
)

print("\n======================================")
print("EMERGENCY SAMPLES")
print("======================================")

print(
    df[df["label"] == 1][features]
    .head(10)
    .to_string(index=False)
)