from pathlib import Path
import numpy as np
import pandas as pd

NORMAL_FILE = Path("data/normal_flight_data.csv")
EMERGENCY_FILE = Path("real_emergency_10sec_clean_v3.csv")
OUTPUT_FILE = Path("data/real_training_dataset_10_clean.csv")
WINDOW_SIZE = 10
MAX_GAP_SECONDS = 30

FEATURES = [
    "altitude_mean", "altitude_std", "altitude_change",
    "velocity_mean", "velocity_std", "velocity_change",
    "heading_mean", "heading_std",
    "vertical_rate_mean", "vertical_rate_std",
    "vertical_rate_min", "vertical_rate_max",
]

def make_windows(frame, group_col, label, declaration_col=None):
    results = []

    for group_id, flight in frame.groupby(group_col, sort=False):
        flight = flight.sort_values("timestamp").copy()
        flight["timestamp"] = pd.to_datetime(flight["timestamp"], utc=True)

        # Break a flight into continuous segments; never bridge large gaps.
        gap = flight["timestamp"].diff().dt.total_seconds()
        flight["_segment"] = (gap.isna() | gap.gt(MAX_GAP_SECONDS)).cumsum()

        for _, segment in flight.groupby("_segment", sort=False):
            segment = segment.reset_index(drop=True)
            if len(segment) < WINDOW_SIZE:
                continue

            for start in range(0, len(segment) - WINDOW_SIZE + 1, WINDOW_SIZE):
                w = segment.iloc[start:start + WINDOW_SIZE]

                # For emergency flights, do not label a window that crosses
                # the first recorded 7700 declaration.
                if declaration_col is not None:
                    declared = w[declaration_col].astype(bool)
                    if declared.any() and not declared.all():
                        continue
                    window_label = int(declared.all())
                else:
                    window_label = label

                if window_label is None:
                    continue

                row = {
                    "group_id": str(group_id),
                    "label": window_label,
                    "window_start": w["timestamp"].iloc[0],
                    "window_end": w["timestamp"].iloc[-1],
                }

                for col, out in [
                    ("baro_altitude", "altitude"),
                    ("velocity", "velocity"),
                    ("heading", "heading"),
                    ("vertical_rate", "vertical_rate"),
                ]:
                    values = w[col].to_numpy(dtype=float)
                    row[f"{out}_mean"] = float(np.mean(values))
                    row[f"{out}_std"] = float(np.std(values, ddof=0))
                    if out in ("altitude", "velocity"):
                        row[f"{out}_change"] = float(values[-1] - values[0])

                vr = w["vertical_rate"].to_numpy(dtype=float)
                row["vertical_rate_min"] = float(np.min(vr))
                row["vertical_rate_max"] = float(np.max(vr))
                results.append(row)

    return pd.DataFrame(results)

print("Loading normal flight data...")
normal = pd.read_csv(NORMAL_FILE)
normal["timestamp"] = pd.to_datetime(normal["timestamp"], utc=True)
normal = normal.dropna(
    subset=["icao24", "timestamp", "baro_altitude", "velocity",
            "heading", "vertical_rate"]
)

print("Loading cleaned emergency flight data...")
emergency = pd.read_csv(EMERGENCY_FILE)
emergency["timestamp"] = pd.to_datetime(emergency["timestamp"], utc=True)
emergency["squawk"] = (
    emergency["squawk"].astype(str).str.replace(r"\.0$", "", regex=True)
)
emergency = emergency.dropna(
    subset=["flight_id", "timestamp", "altitude", "groundspeed",
            "track", "vertical_rate", "squawk"]
).copy()

# Convert emergency columns to the names used by the model.
emergency = emergency.rename(columns={
    "altitude": "baro_altitude",
    "groundspeed": "velocity",
    "track": "heading",
})

# A declaration proxy: first recorded 7700 observation for each flight.
emergency["_declared"] = emergency["squawk"].eq("7700")
emergency["_after_declaration"] = (
    emergency.groupby("flight_id")["_declared"].cummax()
)

print("Building normal windows...")
normal_windows = make_windows(normal, "icao24", 0)

print("Building emergency-flight windows...")
emergency_windows = make_windows(
    emergency,
    "flight_id",
    None,
    declaration_col="_after_declaration",
)

if normal_windows.empty or emergency_windows.empty:
    raise RuntimeError("No windows created for one or both classes.")

normal_windows["group_id"] = "normal_" + normal_windows["group_id"]
emergency_windows["group_id"] = "emergency_" + emergency_windows["group_id"]

print("\nBefore balancing:")
print("Normal windows:", int((normal_windows["label"] == 0).sum()))
print("Emergency windows:", int((emergency_windows["label"] == 1).sum()))
print("Pre-declaration normal windows:", int((emergency_windows["label"] == 0).sum()))

# Combine ordinary normal flights with pre-declaration windows.
normal_pool = pd.concat([
    normal_windows[normal_windows["label"] == 0],
    emergency_windows[emergency_windows["label"] == 0],
], ignore_index=True)

emergency_pool = emergency_windows[
    emergency_windows["label"] == 1
].copy()

# Balance classes without overwriting the previous dataset.
n = min(len(normal_pool), len(emergency_pool))
if n == 0:
    raise RuntimeError("One class has no windows after declaration-based labelling.")

normal_pool = normal_pool.sample(n=n, random_state=42)
emergency_pool = emergency_pool.sample(n=n, random_state=42)

dataset = pd.concat([normal_pool, emergency_pool], ignore_index=True)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
dataset.to_csv(OUTPUT_FILE, index=False)

print("\nCandidate dataset saved:", OUTPUT_FILE)
print("Total windows:", len(dataset))
print("Class counts:")
print(dataset["label"].value_counts().sort_index().to_string())
print("Unique groups:", dataset["group_id"].nunique())
print("Feature columns:", FEATURES)
