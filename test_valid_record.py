from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
from collections import deque
import time

source = Path("response_processing_section.txt").read_text(encoding="utf-8")

start = source.index("        timestamp = datetime.fromtimestamp(")
end = source.index("        print(", start)

# Use the actual production record-building and append statements.
block = source[start:end]
block = "\n".join(line[8:] if line.startswith("        ") else line for line in block.splitlines())

now = datetime.now(timezone.utc).timestamp()
state = SimpleNamespace(
    icao24="test123", callsign="TEST123 ",
    latitude=12.9, longitude=77.6,
    baro_altitude=10000.0, geo_altitude=10100.0,
    velocity=220.0, true_track=90.0,
    vertical_rate=0.0, on_ground=False, squawk="1200"
)
observation_time = now
recording = deque()
ai_history = deque()

env = {
    "datetime": datetime, "timezone": timezone,
    "observation_time": observation_time, "state": state,
    "recording": recording, "ai_history": ai_history,
    "RECORDING_SECONDS": 45 * 60,
    "time": time, "print": print,
}

exec(compile(block, "<production-record-building>", "exec"), env)

assert len(recording) == 1, f"Expected 1 recording, got {len(recording)}"
assert len(ai_history) == 1, f"Expected 1 AI-history entry, got {len(ai_history)}"
assert recording[0]["icao24"] == "test123"
assert recording[0]["velocity"] == 220.0

print("PASS: Valid telemetry converted to a production-format record.")
print("PASS: Record appended to rolling recording and AI history.")
print("Note: Record-building was tested, not the full live polling loop.")
