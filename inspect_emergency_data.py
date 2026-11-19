


import pandas as pd

FILE = "real_emergency_10sec.csv"

df = pd.read_csv(FILE)

print("======================================")
print("EMERGENCY DATA CHECK")
print("======================================")

print("\nRows:", len(df))

print("\nVertical rate statistics:")
print(df["vertical_rate"].describe())

print("\nMost extreme vertical-rate values:")

print("\nLowest 20:")
print(
    df[["vertical_rate", "altitude", "groundspeed", "track"]]
    .sort_values("vertical_rate")
    .head(20)
    .to_string(index=False)
)

print("\nHighest 20:")
print(
    df[["vertical_rate", "altitude", "groundspeed", "track"]]
    .sort_values("vertical_rate", ascending=False)
    .head(20)
    .to_string(index=False)
)