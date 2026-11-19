from pathlib import Path
import numpy as np
import pandas as pd

NORMAL_FILE = Path("data/normal_flight_data.csv")
EMERGENCY_FILE = Path("real_emergency_10sec_clean_v3.csv")
OUTPUT_FILE = Path("data/real_training_dataset_10s.csv")

STEP_SECONDS = 10
MAX_GAP_SECONDS = 30
WINDOW_SIZE = 10
MIN_REAL_POINTS = 8

FEATURES = [
    "altitude_mean", "altitude_std", "altitude_change",
    "velocity_mean", "velocity_std", "velocity_change",
    "heading_mean", "heading_std",
    "vertical_rate_mean", "vertical_rate_std",
    "vertical_rate_min", "vertical_rate_max",
]

def make_windows(df, group_col, label, emergency=False):
    rows = []

    for group_id, flight in df.groupby(group_col, sort=False):
        flight = flight.sort_values("timestamp").copy()
        flight["timestamp"] = pd.to_datetime(flight["timestamp"], utc=True)
        flight = flight.drop_duplicates("timestamp", keep="first")

        # Split at large gaps before resampling.
        gaps = flight["timestamp"].diff().dt.total_seconds()
        flight["_segment"] = (gaps.isna() | gaps.gt(MAX_GAP_SECONDS)).cumsum()

        for _, seg in flight.groupby("_segment", sort=False):
            seg = seg.set_index("timestamp").sort_index()
            if len(seg) < MIN_REAL_POINTS:
                continue

            # 10-second bins: use actual observations, not interpolation.
            numeric = ["baro_altitude", "velocity", "heading", "vertical_rate"]
            binned = seg[numeric].resample("10s").median()
            counts = seg[numeric[0]].resample("10s").count()
            binned["_real"] = counts.gt(0)

            # Only use continuous runs of bins with an actual observation.
            binned["_run"] = (
                (~binned["_real"]) | (~binned["_real"].shift(fill_value=False))
            ).cumsum()

            for _, run in binned[binned["_real"]].groupby("_run", sort=False):
                if len(run) < WINDOW_SIZE:
                    continue

                # Require at least 8 of 10 bins to contain actual observations.
                for start in range(0, len(run) - WINDOW_SIZE + 1, WINDOW_SIZE):
                    w = run.iloc[start:start + WINDOW_SIZE]
                    if len(w) != WINDOW_SIZE or int(w["_real"].sum()) < MIN_REAL_POINTS:
                        continue

                    row = {
                        "group_id": str(group_id),
                        "label": int(label),
                        "window_start": w.index[0],
                        "window_end": w.index[-1],
                    }

                    for col, prefix in [
                        ("baro_altitude", "altitude"),
                        ("velocity", "velocity"),
                        ("heading", "heading"),
                        ("vertical_rate", "vertical_rate"),
                    ]:
                        values = w[col].dropna().to_numpy(dtype=float)
                        if len(values) < MIN_REAL_POINTS:
                            break
                        row[f"{prefix}_mean"] = float(np.mean(values))
                        row[f"{prefix}_std"] = float(np.std(values, ddof=0))
                        if prefix in ("altitude", "velocity"):
                            row[f"{prefix}_change"] = float(values[-1] - values[0])
                    else:
                        vr = w["vertical_rate"].dropna().to_numpy(dtype=float)
                        row["vertical_rate_min"] = float(np.min(vr))
                        row["vertical_rate_max"] = float(np.max(vr))
                        rows.append(row)

    return pd.DataFrame(rows)

print("Loading normal data...")
normal = pd.read_csv(NORMAL_FILE)
normal["timestamp"] = pd.to_datetime(normal["timestamp"], utc=True)
normal = normal.dropna(subset=[
    "icao24", "timestamp", "baro_altitude",
    "velocity", "heading", "vertical_rate"
])
normal = normal.rename(columns={"icao24": "_group"})

print("Loading cleaned emergency data...")
emergency = pd.read_csv(EMERGENCY_FILE)
emergency["timestamp"] = pd.to_datetime(emergency["timestamp"], utc=True)
emergency["squawk"] = (
    emergency["squawk"].astype(str).str.replace(r"\.0$", "", regex=True)
)
emergency = emergency.dropna(subset=[
    "flight_id", "timestamp", "altitude",
    "groundspeed", "track", "vertical_rate", "squawk"
]).copy()

emergency = emergency.rename(columns={
    "flight_id": "_group",
    "altitude": "baro_altitude",
    "groundspeed": "velocity",
    "track": "heading",
})

# The first recorded 7700 is a declaration proxy, not verified onset.
emergency["_declared"] = emergency["squawk"].eq("7700")
emergency["_after_declaration"] = (
    emergency.groupby("_group")["_declared"].cummax()
)

# Split emergency source into normal-proxy and emergency-labelled segments.
# We do not let a window cross the declaration boundary.
normal_parts = []
emergency_parts = []

for _, flight in emergency.groupby("_group", sort=False):
    flight = flight.sort_values("timestamp").copy()
    flight["_label"] = flight["_after_declaration"].astype(int)
    for label, part in flight.groupby("_label", sort=False):
        if len(part):
            normal_parts.append(part) if label == 0 else emergency_parts.append(part)

print("Creating windows...")
normal_windows = make_windows(normal, "_group", 0)
pre_windows = pd.concat(
    [make_windows(part, "_group", 0) for part in normal_parts],
    ignore_index=True
) if normal_parts else pd.DataFrame()

emergency_windows = pd.concat(
    [make_windows(part, "_group", 1) for part in emergency_parts],
    ignore_index=True
) if emergency_parts else pd.DataFrame()

normal_pool = pd.concat([normal_windows, pre_windows], ignore_index=True)
if normal_pool.empty or emergency_windows.empty:
    raise RuntimeError("No usable windows in one or both classes.")

# Keep source groups distinct and balance classes.
normal_pool["group_id"] = "normal_" + normal_pool["group_id"].astype(str)
emergency_windows["group_id"] = "emergency_" + emergency_windows["group_id"].astype(str)

n = min(len(normal_pool), len(emergency_windows))
normal_pool = normal_pool.sample(n=n, random_state=42)
emergency_windows = emergency_windows.sample(n=n, random_state=42)

dataset = pd.concat([normal_pool, emergency_windows], ignore_index=True)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
dataset.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
print("Rows:", len(dataset))
print("Label counts:")
print(dataset["label"].value_counts().sort_index().to_string())
print("Unique groups:", dataset["group_id"].nunique())
print("Feature list:", FEATURES)
