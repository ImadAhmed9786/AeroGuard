import ast
import csv
import os
import tempfile
from pathlib import Path
from collections import deque
from unittest.mock import patch
import pandas as pd

SOURCE = Path("aeroguard_single.py")
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))

wanted = {
    "write_recording",
    "save_previous_recording",
    "trigger_emergency",
    "append_post_trigger_telemetry",
    "run_ai",
}
functions = {
    n.name: n for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name in wanted
}
if set(functions) != wanted:
    raise SystemExit(f"Missing functions: {wanted - set(functions)}")

module = ast.Module(body=list(functions.values()), type_ignores=[])
ast.fix_missing_locations(module)

class FakeModel:
    classes_ = [0, 1]
    def predict(self, features):
        return [1]
    def predict_proba(self, features):
        return [[0.01, 0.99]]

class FakeWindow:
    def __init__(self, t):
        self.rows = [
            {"timestamp_unix": t - 90 + i * 10}
            for i in range(10)
        ]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        return self.rows[i]

calls = {"transmit": 0}
clock = {"t": 1_800_000_000.0}
recording = deque()
ai_history = deque(maxlen=30)

def make_record(t):
    return {
        "timestamp": pd.Timestamp(t, unit="s", tz="UTC").isoformat(),
        "timestamp_unix": t,
        "icao24": "test123",
        "callsign": "SIMTEST",
        "latitude": 12.9,
        "longitude": 77.6,
        "baro_altitude": 10000.0,
        "geo_altitude": 10000.0,
        "velocity": 220.0,
        "heading": 180.0,
        "vertical_rate": 0.0,
        "on_ground": False,
        "squawk": "0000",
    }

def fail_transmitter(**kwargs):
    calls["transmit"] += 1
    raise OSError("Intentional simulated transmission failure")

env = {
    "os": os, "csv": csv, "pd": pd,
    "recording": recording,
    "ai_history": ai_history,
    "AIRCRAFT_ID": "SIMTEST01",
    "CALLSIGN": "SIMTEST",
    "RECORDING_SIZE": 270,
    "RECORDING_MINUTES": 45,
    "RECORDING_SECONDS": 2700,
    "AI_WINDOW": 10,
    "AI_HISTORY_SIZE": 30,
    "ANOMALY_REQUIRED": 5,
    "EMERGENCY_CONFIDENCE": 85.0,
    "anomaly_count": 0,
    "emergency_triggered": False,
    "last_ai_window_end": None,
    "model": FakeModel(),
    "get_ai_window": lambda: FakeWindow(clock["t"]),
    "create_features": lambda window: pd.DataFrame([{}]),
    "activate_transmitter": fail_transmitter,
    "print": print,
    "int": int, "float": float,
}

exec(compile(module, str(SOURCE), "exec"), env)
save_recording = env["save_previous_recording"]
append_post = env["append_post_trigger_telemetry"]
run_ai = env["run_ai"]

with tempfile.TemporaryDirectory(prefix="aeroguard_e2e_") as temp:
    old_cwd = os.getcwd()
    os.chdir(temp)
    try:
        # Seed a full synthetic 45-minute rolling buffer.
        for i in range(270):
            recording.append(make_record(clock["t"] - (269 - i) * 10))

        for step in range(1, 51):
            clock["t"] += 10
            rec = make_record(clock["t"])
            recording.append(rec)
            ai_history.append(rec)

            cutoff = clock["t"] - env["RECORDING_SECONDS"]
            while recording and recording[0]["timestamp_unix"] < cutoff:
                recording.popleft()

            try:
                run_ai()
            finally:
                append_post(rec)

        assert env["emergency_triggered"], "Emergency did not trigger"
        assert calls["transmit"] == 1, "Unexpected transmitter call count"
        print("PASS: Five qualifying windows triggered emergency")
        print("PASS: Pre-trigger recording save attempted")
        print("PASS: Simulated transmitter failure handled")

        pre = Path("data/emergencies/SIMTEST01/previous_45_minutes.csv")
        assert pre.is_file(), "Pre-trigger recording missing"
        pre_df = pd.read_csv(pre)
        assert len(pre_df) in (270, 271), f"Unexpected snapshot size: {len(pre_df)}"
        assert "timestamp" in pre_df.columns
        assert pre_df["timestamp"].notna().all()
        print("PASS: Saved pre-trigger recording contains 270 observations")

        post = Path("data/ground_station/SIMTEST01/post_trigger_telemetry.csv")
        assert post.is_file(), "Post-trigger telemetry missing"
        post_df = pd.read_csv(post)
        assert len(post_df) == 1, f"Expected triggering row, got {len(post_df)}"
        assert float(post_df.iloc[0]["timestamp_unix"]) == clock["t"]
        print("PASS: Triggering observation appears in post-trigger telemetry")

        # Simulate later observations after the failed transmission.
        for _ in range(3):
            clock["t"] += 10
            rec = make_record(clock["t"])
            recording.append(rec)
            ai_history.append(rec)
            cutoff = clock["t"] - env["RECORDING_SECONDS"]
            while recording and recording[0]["timestamp_unix"] < cutoff:
                recording.popleft()
            try:
                run_ai()
            finally:
                append_post(rec)

        post_df = pd.read_csv(post)
        assert len(post_df) == 4, f"Expected 4 post-trigger rows, got {len(post_df)}"
        assert calls["transmit"] == 1, "Failed transmission was repeated"
        assert env["emergency_triggered"], "Emergency latch unexpectedly reset"
        print("PASS: Three later observations recorded after transmission failure")
        print("PASS: No duplicate transmission attempt")
        print("\nEND-TO-END SIMULATION PASSED")

    finally:
        os.chdir(old_cwd)

