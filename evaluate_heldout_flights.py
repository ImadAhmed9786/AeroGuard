
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# AEROGUARD: PROPER TRAIN / VALIDATION / TEST EVALUATION
# ============================================================

NORMAL_FILE = Path("data/normal_flight_data.csv")
EMERGENCY_FILE = Path("real_emergency_10sec_clean_v3.csv")

CANDIDATE_FILE = Path("data/aeroguard_temporal_candidate.pkl")
RESULTS_FILE = Path("data/aeroguard_temporal_test_results.csv")
SUMMARY_FILE = Path("data/aeroguard_temporal_test_summary.csv")

WINDOW_SIZE = 10
MAX_GAP_SECONDS = 30
MAX_VALIDATION_FPR = 0.20
RANDOM_STATE = 42

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
    "vertical_rate_max",
    "altitude_slope",
    "velocity_slope",
    "vertical_rate_slope",
    "vertical_rate_abs_mean",
    "vertical_rate_nonzero_fraction",
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def heading_statistics(values):
    radians = np.deg2rad(np.asarray(values, dtype=float))
    sin_mean = np.mean(np.sin(radians))
    cos_mean = np.mean(np.cos(radians))

    mean = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360
    resultant = np.clip(
        np.sqrt(sin_mean ** 2 + cos_mean ** 2),
        1e-12,
        1.0,
    )
    std = np.rad2deg(np.sqrt(-2 * np.log(resultant)))

    return float(mean), float(std)


def calculate_slope(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    elapsed = times - times[0]

    if len(values) < 2 or elapsed[-1] <= 0:
        return 0.0

    return float(np.polyfit(elapsed, values, 1)[0])


def extract_features(window):
    altitude = window["altitude"].to_numpy(dtype=float)
    velocity = window["groundspeed"].to_numpy(dtype=float)
    heading = window["track"].to_numpy(dtype=float)
    vertical = window["vertical_rate"].to_numpy(dtype=float)

    times = (
        window["timestamp"].astype("int64").to_numpy(dtype=float)
        / 1e9
    )

    heading_mean, heading_std = heading_statistics(heading)

    return {
        "altitude_mean": float(np.mean(altitude)),
        "altitude_std": float(np.std(altitude, ddof=1)),
        "altitude_change": float(altitude[-1] - altitude[0]),

        "velocity_mean": float(np.mean(velocity)),
        "velocity_std": float(np.std(velocity, ddof=1)),
        "velocity_change": float(velocity[-1] - velocity[0]),

        "heading_mean": heading_mean,
        "heading_std": heading_std,

        "vertical_rate_mean": float(np.mean(vertical)),
        "vertical_rate_std": float(np.std(vertical, ddof=1)),
        "vertical_rate_min": float(np.min(vertical)),
        "vertical_rate_max": float(np.max(vertical)),

        "altitude_slope": calculate_slope(altitude, times),
        "velocity_slope": calculate_slope(velocity, times),
        "vertical_rate_slope": calculate_slope(vertical, times),
        "vertical_rate_abs_mean": float(np.mean(np.abs(vertical))),
        "vertical_rate_nonzero_fraction": float(
            np.mean(np.abs(vertical) > 0.5)
        ),
    }



def split_gaps(group):
    """Split a flight timeline at gaps longer than MAX_GAP_SECONDS."""
    group = group.sort_values("timestamp").copy()

    seconds = (
        group["timestamp"].astype("int64").to_numpy(dtype=float)
        / 1e9
    )

    breaks = np.where(np.diff(seconds) > MAX_GAP_SECONDS)[0] + 1

    # np.split returns NumPy arrays of row positions.
    # Convert each segment back into a pandas DataFrame.
    positions = np.split(np.arange(len(group)), breaks)

    return [
        group.iloc[position].copy()
        for position in positions
        if len(position) >= WINDOW_SIZE
    ]


# ============================================================
# BUILD GAP-AWARE WINDOWS
# ============================================================

def build_dataset(normal, emergency):
    rows = []

    # Normal aircraft. All windows for an aircraft stay together
    # during the eventual train/validation/test split.
    for aircraft, aircraft_df in normal.groupby("icao24"):
        group_id = f"NORMAL_{aircraft}"

        for segment in split_gaps(aircraft_df):
            for start in range(
                0, len(segment) - WINDOW_SIZE + 1, WINDOW_SIZE
            ):
                window = segment.iloc[start:start + WINDOW_SIZE]

                row = extract_features(window)
                row.update({
                    "label": 0,
                    "group_id": group_id,
                    "source": "normal",
                    "window_start": window["timestamp"].iloc[0],
                    "window_end": window["timestamp"].iloc[-1],
                })
                rows.append(row)

    # Emergency source. First observed squawk 7700 is used as a
    # proxy boundary, not as proof of a mechanical failure.
    for flight_id, flight_df in emergency.groupby("flight_id"):
        flight_df = flight_df.sort_values("timestamp").copy()

        declaration_times = flight_df.loc[
            flight_df["squawk"].astype(str).str.strip() == "7700",
            "timestamp",
        ]

        if declaration_times.empty:
            continue

        first_declaration = declaration_times.min()
        group_id = f"EMERGENCY_{flight_id}"

        for segment in split_gaps(flight_df):
            for start in range(
                0, len(segment) - WINDOW_SIZE + 1, WINDOW_SIZE
            ):
                window = segment.iloc[start:start + WINDOW_SIZE]

                start_time = window["timestamp"].iloc[0]
                end_time = window["timestamp"].iloc[-1]

                # Never label a window that crosses the declaration.
                if start_time < first_declaration <= end_time:
                    continue

                label = int(start_time >= first_declaration)

                row = extract_features(window)
                row.update({
                    "label": label,
                    "group_id": group_id,
                    "source": "emergency_source",
                    "window_start": start_time,
                    "window_end": end_time,
                })
                rows.append(row)

    dataset = pd.DataFrame(rows)

    if dataset.empty:
        raise RuntimeError("No windows generated. Check the input CSV files.")

    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset.dropna(subset=FEATURES + ["label", "group_id"], inplace=True)
    dataset["label"] = dataset["label"].astype(int)

    return dataset.reset_index(drop=True)


# ============================================================
# GROUPED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_by_flight(dataset):
    # Test groups are held out first.
    # Validation groups are then held out from the remaining groups.
    for seed in range(42, 142):
        first_split = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=seed,
        )

        train_val_idx, test_idx = next(
            first_split.split(
                dataset[FEATURES],
                dataset["label"],
                dataset["group_id"],
            )
        )

        train_val = dataset.iloc[train_val_idx]
        test = dataset.iloc[test_idx]

        second_split = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=seed + 1000,
        )

        train_idx, val_idx = next(
            second_split.split(
                train_val[FEATURES],
                train_val["label"],
                train_val["group_id"],
            )
        )

        train = train_val.iloc[train_idx].copy()
        validation = train_val.iloc[val_idx].copy()
        test = test.copy()

        # Require both labels in every partition.
        labels_valid = all(
            set(part["label"]) == {0, 1}
            for part in (train, validation, test)
        )

        groups = [
            set(part["group_id"])
            for part in (train, validation, test)
        ]

        no_overlap = not (
            groups[0] & groups[1]
            or groups[0] & groups[2]
            or groups[1] & groups[2]
        )

        if labels_valid and no_overlap:
            return train, validation, test, seed

    raise RuntimeError(
        "Could not create three valid group-separated partitions. "
        "Check the number of distinct aircraft and flights."
    )


# ============================================================
# METRICS AND VALIDATION-ONLY THRESHOLD SELECTION
# ============================================================

def calculate_metrics(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "miss_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(y_true, scores):
    # This function receives VALIDATION scores only.
    candidates = np.arange(0.10, 0.951, 0.01)

    results = [
        calculate_metrics(y_true, scores, threshold)
        for threshold in candidates
    ]

    acceptable = [
        result for result in results
        if result["false_positive_rate"] <= MAX_VALIDATION_FPR
    ]

    if acceptable:
        # Maximise recall while keeping validation FPR at or below 20%.
        chosen = max(
            acceptable,
            key=lambda result: (
                result["recall"],
                -result["false_positive_rate"],
                result["threshold"],
            ),
        )
        rule = "Maximum validation recall with FPR <= 20%"
    else:
        # Safe fallback if no candidate meets the configured constraint.
        chosen = min(
            results,
            key=lambda result: (
                result["false_positive_rate"],
                -result["recall"],
            ),
        )
        rule = "No threshold met validation FPR limit; minimum validation FPR"

    return chosen, rule


# ============================================================
# MAIN
# ============================================================

def main():
    Path("data").mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("AEROGUARD: GROUPED MODEL EVALUATION")
    print("=" * 65)

    normal = pd.read_csv(NORMAL_FILE)
    emergency = pd.read_csv(EMERGENCY_FILE)

    required_normal = {
        "timestamp", "icao24", "baro_altitude",
        "velocity", "heading", "vertical_rate",
    }

    required_emergency = {
        "timestamp", "flight_id", "altitude",
        "groundspeed", "track", "vertical_rate", "squawk",
    }

    if not required_normal.issubset(normal.columns):
        raise ValueError(
            f"Normal CSV missing columns: {required_normal - set(normal.columns)}"
        )

    if not required_emergency.issubset(emergency.columns):
        raise ValueError(
            f"Emergency CSV missing columns: "
            f"{required_emergency - set(emergency.columns)}"
        )

    normal.rename(columns={
        "baro_altitude": "altitude",
        "velocity": "groundspeed",
        "heading": "track",
    }, inplace=True)

    for frame in (normal, emergency):
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], utc=True, errors="coerce"
        )

        for column in (
            "altitude", "groundspeed", "track", "vertical_rate"
        ):
            frame[column] = pd.to_numeric(
                frame[column], errors="coerce"
            )

        frame.dropna(
            subset=[
                "timestamp", "altitude", "groundspeed",
                "track", "vertical_rate",
            ],
            inplace=True,
        )

    normal["icao24"] = normal["icao24"].astype(str).str.strip().str.lower()
    emergency["flight_id"] = emergency["flight_id"].astype(str).str.strip()
    emergency["squawk"] = (
        emergency["squawk"].astype(str).str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    print("Building windows...")
    dataset = build_dataset(normal, emergency)

    print(f"Total windows: {len(dataset):,}")
    print(f"Normal-labelled windows: {(dataset['label'] == 0).sum():,}")
    print(f"Post-declaration windows: {(dataset['label'] == 1).sum():,}")
    print(f"Flight/aircraft groups: {dataset['group_id'].nunique():,}")

    train, validation, test, split_seed = split_by_flight(dataset)

    for name, part in (
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ):
        print(
            f"{name:10} | windows={len(part):,} | "
            f"groups={part['group_id'].nunique():,} | "
            f"labels={part['label'].value_counts().sort_index().to_dict()}"
        )

    assert not set(train.group_id) & set(validation.group_id)
    assert not set(train.group_id) & set(test.group_id)
    assert not set(validation.group_id) & set(test.group_id)

    print("Flight/aircraft overlap: 0")
    print("\nTraining Random Forest using TRAIN only...")

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(train[FEATURES], train["label"])

    emergency_index = list(model.classes_).index(1)

    # Choose threshold using VALIDATION only.
    validation_scores = model.predict_proba(
        validation[FEATURES]
    )[:, emergency_index]

    selected, selection_rule = select_threshold(
        validation["label"].to_numpy(),
        validation_scores,
    )

    threshold = selected["threshold"]

    print("\nVALIDATION-SELECTED THRESHOLD")
    print("-" * 65)
    print("Rule:", selection_rule)
    print(f"Threshold: {threshold:.2f}")
    print(f"Validation recall: {selected['recall']:.2%}")
    print(f"Validation FPR: {selected['false_positive_rate']:.2%}")
    print(f"Validation miss rate: {selected['miss_rate']:.2%}")

    # Threshold is now locked. Evaluate the untouched test flights once.
    print("\nFINAL UNTOUCHED TEST EVALUATION")
    print("-" * 65)

    test_scores = model.predict_proba(
        test[FEATURES]
    )[:, emergency_index]

    final = calculate_metrics(
        test["label"].to_numpy(),
        test_scores,
        threshold,
    )

    predictions = (test_scores >= threshold).astype(int)

    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy: {final['accuracy']:.2%}")
    print(f"Emergency precision: {final['precision']:.2%}")
    print(f"Emergency recall: {final['recall']:.2%}")
    print(f"Emergency F1: {final['f1']:.2%}")
    print(f"False-positive rate: {final['false_positive_rate']:.2%}")
    print(f"Emergency miss rate: {final['miss_rate']:.2%}")

    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array([
        [final["tn"], final["fp"]],
        [final["fn"], final["tp"]],
    ]))

    print("\nClassification report:")
    print(classification_report(
        test["label"],
        predictions,
        labels=[0, 1],
        target_names=["Normal-labelled", "Post-declaration"],
        zero_division=0,
    ))

    results = test[
        ["group_id", "source", "window_start", "window_end", "label"]
    ].copy()

    results["emergency_score"] = test_scores
    results["threshold"] = threshold
    results["prediction"] = predictions
    results.to_csv(RESULTS_FILE, index=False)

    summary = {
        **final,
        "split_seed": split_seed,
        "train_windows": len(train),
        "validation_windows": len(validation),
        "test_windows": len(test),
        "train_groups": train["group_id"].nunique(),
        "validation_groups": validation["group_id"].nunique(),
        "test_groups": test["group_id"].nunique(),
        "threshold_selection_rule": selection_rule,
        "label_definition": (
            "Post first observed squawk 7700 declaration; "
            "proxy label, not verified mechanical failure."
        ),
    }

    pd.DataFrame([summary]).to_csv(SUMMARY_FILE, index=False)

    candidate = {
        "model": model,
        "features": FEATURES,
        "window_size": WINDOW_SIZE,
        "max_gap_seconds": MAX_GAP_SECONDS,
        "threshold": threshold,
        "threshold_selection": selection_rule,
        "validation_max_false_positive_rate": MAX_VALIDATION_FPR,
        "label_definition": summary["label_definition"],
        "test_metrics": final,
        "status": "candidate_not_deployed",
    }

    joblib.dump(candidate, CANDIDATE_FILE)

    print("\nSAVED")
    print("Candidate:", CANDIDATE_FILE)
    print("Test predictions:", RESULTS_FILE)
    print("Test summary:", SUMMARY_FILE)
    print("Live aeroguard_model_10.pkl was NOT modified.")
    print("\nAeroGuard project completion: 95%")
    print("Research evaluation only, not certified aviation equipment.")


if __name__ == "__main__":
    main()
