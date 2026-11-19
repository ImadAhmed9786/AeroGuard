import os
import shutil
import tempfile
import importlib.util
from collections import deque
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# Load the real source without executing its live-monitor startup.
source = Path("aeroguard_single.py").read_text(encoding="utf-8")
tree = __import__("ast").parse(source)

wanted = {
    "save_previous_recording",
    "append_post_trigger_telemetry",
}

functions = [
    node for node in tree.body
    if isinstance(node, (__import__("ast").FunctionDef,
                         __import__("ast").AsyncFunctionDef))
    and node.name in wanted
]

found = {node.name for node in functions}
if found != wanted:
    raise SystemExit(f"Could not find expected functions: {wanted - found}")

module = __import__("ast").Module(body=functions, type_ignores=[])
namespace = {
    "os": os,
    "pd": pd,
    "csv": __import__("csv"),
    "recording": deque(),
    "AIRCRAFT_ID": "TEST123",
    "RECORDING_SIZE": 270,
    "emergency_triggered": False,
    "print": print,
}
exec(compile(module, "aeroguard_single.py", "exec"), namespace)

save_recording = namespace["save_previous_recording"]
append_post = namespace["append_post_trigger_telemetry"]

def make_record(seconds):
    return {
        "timestamp": f"2026-10-18T12:{seconds // 60:02d}:{seconds % 60:02d}+00:00",
        "timestamp_unix": 1_800_000_000 + seconds,
        "icao24": "test123",
        "latitude": 12.9,
        "longitude": 77.6,
        "baro_altitude": 10000.0,
        "velocity": 220.0,
        "heading": 180.0,
        "vertical_rate": 0.0,
    }

temp_root = Path(tempfile.mkdtemp(prefix="aeroguard_pipeline_"))
try:
    # Redirect output files to an isolated temporary directory.
    namespace["os"].makedirs = os.makedirs

    namespace["recording"] = deque(
        make_record(i * 10) for i in range(270)
    )

    with patch.dict(os.environ, {"AEROGUARD_TEST": "1"}):
        # Test the actual save function in a temporary working directory.
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        try:
            saved = save_recording()
            assert saved and os.path.isfile(saved), "Recording file not saved"

            saved_df = pd.read_csv(saved)
            assert len(saved_df) == 270, f"Expected 270 rows, got {len(saved_df)}"
            assert "timestamp_unix" not in saved_df.columns
            print("PASS: 270-observation pre-trigger recording saved")

            # Test post-trigger logging with the actual function.
            namespace["emergency_triggered"] = False
            post_record = make_record(2700)
            append_post(post_record)
            folder = Path("data/ground_station/TEST123")
            post_file = folder / "post_trigger_telemetry.csv"
            assert not post_file.exists(), "Logged before trigger"

            namespace["emergency_triggered"] = True
            append_post(post_record)
            assert post_file.exists(), "Post-trigger file missing"
            post_df = pd.read_csv(post_file)
            assert len(post_df) == 1, "Unexpected post-trigger row count"
            assert post_df.iloc[0]["icao24"] == "test123"
            print("PASS: Post-trigger telemetry logs only after trigger")
            print("PASS: Triggering observation can be logged")

        finally:
            os.chdir(original_cwd)

    print("\nRecording-function tests passed.")
    print("NOTE: Rolling-buffer trimming and transmitter-failure behaviour")
    print("still require separate tests of the live-loop integration.")

finally:
    shutil.rmtree(temp_root, ignore_errors=True)
