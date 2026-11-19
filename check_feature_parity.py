import ast
from pathlib import Path

import numpy as np
import pandas as pd

TRAINING_FILE = Path("evaluate_heldout_flights.py")
CANDIDATE_FILE = Path("aeroguard_single_candidate.py")

FEATURE_NAMES = [
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

NAMESPACE = {
    "np": np,
    "pd": pd,
    "AI_WINDOW": 10,
    "MAX_GAP_SECONDS": 30,
    "FEATURE_NAMES": FEATURE_NAMES,
}


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))

    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
    ]

    found = {node.name for node in functions}
    missing = set(names) - found

    if missing:
        raise RuntimeError(
            f"{path}: missing functions {sorted(missing)}"
        )

    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


def as_feature_dict(result):
    if isinstance(result, pd.DataFrame):
        return result.iloc[0].to_dict()

    if isinstance(result, pd.Series):
        return result.to_dict()

    return dict(result)


def main():
    print("AEROGUARD FEATURE PARITY TEST")
    print("=" * 70)

    # Load training functions first.
    load_functions(
        TRAINING_FILE,
        {
            "extract_features",
            "heading_statistics",
            "calculate_slope",
        },
        NAMESPACE,
    )

    training_extract = NAMESPACE["extract_features"]

    # Load candidate functions separately.
    load_functions(
        CANDIDATE_FILE,
        {
            "create_features",
            "heading_statistics",
            "calculate_slope",
        },
        NAMESPACE,
    )

    # Use realistic datetime timestamps, as the training pipeline expects.
    timestamps = pd.date_range(
        start="2026-01-01T00:00:00Z",
        periods=10,
        freq="10s",
    )

    training_window = pd.DataFrame({
        "altitude": [10000 + i * 12 for i in range(10)],
        "groundspeed": [220 + i * 0.8 for i in range(10)],
        "track": [350, 354, 358, 2, 6, 10, 14, 18, 22, 26],
        "vertical_rate": [
            0, 0.2, 0.7, 1.1, 0.3,
            -0.8, -1.2, 0, 0.6, 0.9
        ],
        "timestamp": timestamps,
    })

    candidate_window = pd.DataFrame({
        "baro_altitude": training_window["altitude"],
        "velocity": training_window["groundspeed"],
        "heading": training_window["track"],
        "vertical_rate": training_window["vertical_rate"],
        "timestamp_unix": (
            training_window["timestamp"].astype("int64") / 1e9
        ),
    }).to_dict("records")

    expected = as_feature_dict(
        training_extract(training_window)
    )

    actual = as_feature_dict(
        NAMESPACE["create_features"](candidate_window)
    )

    mismatches = []

    for name in FEATURE_NAMES:
        if name not in expected or name not in actual:
            print(f"{name:35} MISSING")
            mismatches.append(name)
            continue

        train_value = float(expected[name])
        candidate_value = float(actual[name])

        passed = np.isclose(
            train_value,
            candidate_value,
            rtol=1e-7,
            atol=1e-7,
        )

        print(
            f"{name:35} "
            f"training={train_value:12.6f} "
            f"candidate={candidate_value:12.6f} "
            f"{'OK' if passed else 'MISMATCH'}"
        )

        if not passed:
            mismatches.append(name)

    print("=" * 70)

    if mismatches:
        print("FAILED: feature calculations differ.")
        print("Features:", ", ".join(mismatches))
        raise SystemExit(1)

    print("PASSED: all 17 features match.")
    print("Synthetic test only; no deployment validation performed.")


if __name__ == "__main__":
    main()