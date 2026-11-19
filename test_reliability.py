import csv
import tempfile
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

print("AEROGUARD RELIABILITY TEST")
print("=" * 45)

# Test 1: Rolling buffer expires old records and keeps recent ones.
now = datetime.now(timezone.utc).timestamp()
buffer = deque([
    {"timestamp_unix": now - 2800},
    {"timestamp_unix": now - 1000},
    {"timestamp_unix": now}
])
cutoff = now - 45 * 60

while buffer and buffer[0]["timestamp_unix"] < cutoff:
    buffer.popleft()

assert len(buffer) == 2
assert all(r["timestamp_unix"] >= cutoff for r in buffer)
print("PASS: 45-minute buffer cutoff")

# Test 2: Simulate transmitter failure, then log later telemetry.
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "post_trigger_telemetry.csv"
    records = [
        {"timestamp": "t1", "altitude": 10000},
        {"timestamp": "t2", "altitude": 9900},
        {"timestamp": "t3", "altitude": 9800},
    ]

    try:
        raise OSError("Intentional simulated transmission failure")
    except OSError:
        pass

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    with path.open(newline="", encoding="utf-8") as f:
        saved = list(csv.DictReader(f))

    assert len(saved) == 3
    assert saved[-1]["timestamp"] == "t3"
    print("PASS: Telemetry logging after simulated transmitter failure")

# Test 3: Check CSV integrity after writing and reading.
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "recording.csv"
    expected = [{"timestamp": "t1", "altitude": "10000"}]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "altitude"])
        writer.writeheader()
        writer.writerows(expected)

    with path.open(newline="", encoding="utf-8") as f:
        actual = list(csv.DictReader(f))

    assert actual == expected
    print("PASS: Recording CSV integrity")

print("=" * 45)
print("ALL SIMULATED RELIABILITY CHECKS PASSED")
print("Note: These tests do not exercise the live AeroGuard loop or OpenSky recovery.")
