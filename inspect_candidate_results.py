from pathlib import Path

import pandas as pd

RESULTS = Path("data/aeroguard_temporal_test_results.csv")
SUMMARY = Path("data/aeroguard_temporal_test_summary.csv")


def main():
    if not RESULTS.exists():
        raise FileNotFoundError(f"Missing file: {RESULTS}")

    df = pd.read_csv(RESULTS)

    print("\nTEST RESULT FILE")
    print("=" * 65)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    print("\nSUMMARY FILE")
    print("=" * 65)

    if SUMMARY.exists():
        print(pd.read_csv(SUMMARY).to_string(index=False))
    else:
        print("Summary file not found.")

    required = {"label", "prediction"}
    if not required.issubset(df.columns):
        print(
            "\nCannot classify errors automatically. "
            "The result file uses different column names."
        )
        print("Available columns:", list(df.columns))
        return

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["prediction"] = pd.to_numeric(
        df["prediction"], errors="coerce"
    )
    df = df.dropna(subset=["label", "prediction"])

    false_negatives = df[
        (df["label"] == 1) & (df["prediction"] == 0)
    ]
    false_positives = df[
        (df["label"] == 0) & (df["prediction"] == 1)
    ]
    true_positives = df[
        (df["label"] == 1) & (df["prediction"] == 1)
    ]
    true_negatives = df[
        (df["label"] == 0) & (df["prediction"] == 0)
    ]

    print("\nCONFUSION COUNTS")
    print("=" * 65)
    print("True positives :", len(true_positives))
    print("True negatives :", len(true_negatives))
    print("False positives:", len(false_positives))
    print("False negatives:", len(false_negatives))

    if "source" in df.columns:
        print("\nERRORS BY DATA SOURCE")
        print("=" * 65)

        errors = df[
            (df["label"] != df["prediction"])
        ]

        print(
            errors.groupby(["source", "label", "prediction"])
            .size()
            .rename("count")
            .to_string()
        )

    if "group_id" in false_negatives.columns:
        print("\nMOST FREQUENTLY MISSED TEST GROUPS")
        print("=" * 65)

        print(
            false_negatives.groupby("group_id")
            .size()
            .sort_values(ascending=False)
            .head(15)
            .rename("missed_windows")
            .to_string()
        )

    print("\nInspection complete. Original model unchanged.")


if __name__ == "__main__":
    main()