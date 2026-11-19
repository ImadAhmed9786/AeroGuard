from collections import deque
from datetime import datetime, timezone
from unittest.mock import Mock
import time

namespace = {
    "deque": deque,
    "datetime": datetime,
    "timezone": timezone,
    "time": time,
    "recording": deque(),
    "ai_history": deque(),
    "AIRCRAFT_ID": "test123",
    "RECORDING_SECONDS": 2700,
    "SNAPSHOT_INTERVAL_SECONDS": 60,
    "last_observation_timestamp": None,
    "last_snapshot_time": time.monotonic(),
    "save_rolling_snapshot": lambda: None,
    "run_ai": lambda: None,
    "append_post_trigger_telemetry": lambda record: None,
}

exec(open("process_response_test_extract.py", encoding="utf-8").read(), namespace)

result = namespace["process_response"](Mock(states=[]))

assert result is False, f"Expected False for empty response, got {result!r}"
assert len(namespace["recording"]) == 0
assert len(namespace["ai_history"]) == 0

print("PASS: Empty OpenSky response is handled without recording fake telemetry.")
