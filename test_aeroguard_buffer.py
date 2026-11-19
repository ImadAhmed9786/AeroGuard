from collections import deque
from datetime import datetime, timezone
import csv
import io
import unittest

RECORDING_SECONDS = 45 * 60


def create_record(timestamp):
    return {
        "timestamp_unix": float(timestamp),
        "timestamp_utc": datetime.fromtimestamp(
            float(timestamp), tz=timezone.utc
        ).isoformat(),
        "altitude_m": 1000.0,
        "velocity_mps": 200.0,
    }


def trim_recording(records, observation_time):
    """Remove observations older than the rolling 45-minute window."""
    cutoff = float(observation_time) - RECORDING_SECONDS

    while records and float(records[0]["timestamp_unix"]) < cutoff:
        records.popleft()

    return records


def validate_recording(
    records,
    trigger_time,
    required_coverage_seconds=RECORDING_SECONDS,
):
    """Check whether the buffer contains a complete, ordered time window."""
    reasons = []

    if not records:
        return False, ["Buffer is empty"]

    try:
        timestamps = [
            float(record["timestamp_unix"])
            for record in records
        ]
    except (KeyError, TypeError, ValueError):
        return False, ["One or more records have invalid timestamps"]

    if timestamps != sorted(timestamps):
        reasons.append("Timestamps are not in chronological order")

    if timestamps[-1] > float(trigger_time):
        reasons.append("Buffer contains data later than the trigger time")

    span = max(timestamps) - min(timestamps)

    if span < required_coverage_seconds:
        reasons.append(
            f"Partial history: {span:.1f} seconds available; "
            f"{required_coverage_seconds:.1f} seconds required"
        )

    cutoff = float(trigger_time) - required_coverage_seconds

    if min(timestamps) > cutoff:
        reasons.append("Buffer starts later than the required cutoff")

    return not reasons, reasons


class TestAeroGuardBuffer(unittest.TestCase):

    def test_exact_cutoff_is_retained(self):
        now = 10000.0

        records = deque(
            create_record(t)
            for t in [7299, 7300, 8000, 9990]
        )

        trim_recording(records, now)

        self.assertEqual(
            [r["timestamp_unix"] for r in records],
            [7300.0, 8000.0, 9990.0],
        )

    def test_old_records_are_removed(self):
        now = 10000.0

        records = deque(
            create_record(t)
            for t in [100, 200, 7300, 9000, 9990]
        )

        trim_recording(records, now)

        self.assertTrue(
            all(r["timestamp_unix"] >= 7300 for r in records)
        )

    def test_empty_buffer_is_rejected(self):
        records = deque()

        valid, reasons = validate_recording(records, 10000)

        self.assertFalse(valid)
        self.assertIn("Buffer is empty", reasons)

    def test_partial_history_is_rejected(self):
        trigger = 10000.0

        records = deque(
            create_record(t)
            for t in [8000, 9000, 9990]
        )

        valid, reasons = validate_recording(records, trigger)

        self.assertFalse(valid)
        self.assertTrue(
            any("Partial history" in reason for reason in reasons)
        )

    def test_complete_45_minute_window_passes(self):
        trigger = 10000.0

        records = deque(
            create_record(trigger - RECORDING_SECONDS + i * 10)
            for i in range(271)
        )

        valid, reasons = validate_recording(records, trigger)

        self.assertTrue(valid, reasons)

    def test_future_records_are_rejected(self):
        trigger = 10000.0

        records = deque([
            create_record(trigger - RECORDING_SECONDS),
            create_record(trigger),
            create_record(trigger + 1),
        ])

        valid, reasons = validate_recording(records, trigger)

        self.assertFalse(valid)
        self.assertIn(
            "Buffer contains data later than the trigger time",
            reasons,
        )

    def test_out_of_order_records_are_rejected(self):
        trigger = 10000.0

        records = deque([
            create_record(trigger - RECORDING_SECONDS),
            create_record(trigger - 100),
            create_record(trigger - 200),
            create_record(trigger),
        ])

        valid, reasons = validate_recording(records, trigger)

        self.assertFalse(valid)
        self.assertIn(
            "Timestamps are not in chronological order",
            reasons,
        )

    def test_csv_preserves_timestamps(self):
        output = io.StringIO()

        records = [
            create_record(1700000000),
            create_record(1700000010),
        ]

        writer = csv.DictWriter(
            output,
            fieldnames=list(records[0].keys()),
        )

        writer.writeheader()
        writer.writerows(records)

        output.seek(0)

        loaded = list(csv.DictReader(output))

        self.assertEqual(len(loaded), 2)
        self.assertIn("timestamp_utc", loaded[0])
        self.assertIn("timestamp_unix", loaded[0])

    def test_duplicate_timestamps_do_not_fake_coverage(self):
        trigger = 10000.0

        records = deque(
            create_record(t)
            for t in [7300, 7300, 7300, 9990]
        )

        valid, reasons = validate_recording(records, trigger)

        self.assertFalse(valid)
        self.assertTrue(
            any("Partial history" in reason for reason in reasons)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)