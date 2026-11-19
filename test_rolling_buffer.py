
from collections import deque
from datetime import datetime, timedelta, timezone

BUFFER = timedelta(minutes=45)
buffer = deque()
start = datetime(2026, 10, 25, 7, 0, tzinfo=timezone.utc)

for minute in range(60):
    now = start + timedelta(minutes=minute)
    buffer.append({"timestamp": now, "minute": minute})

    cutoff = now - BUFFER
    while buffer and buffer[0]["timestamp"] < cutoff:
        buffer.popleft()

first = buffer[0]["timestamp"]
last = buffer[-1]["timestamp"]
coverage = (last - first).total_seconds() / 60

print("AEROGUARD BUFFER TEST")
print("-" * 30)
print("Observations received:", 60)
print("Observations retained:", len(buffer))
print("First retained minute:", buffer[0]["minute"])
print("Last retained minute:", buffer[-1]["minute"])
print("Time coverage:", coverage, "minutes")

assert len(buffer) == 46
assert coverage == 45
assert buffer[0]["minute"] == 14
assert buffer[-1]["minute"] == 59

print("\nSUCCESS: 45-minute rolling buffer logic passed!")
