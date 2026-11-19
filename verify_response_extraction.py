from pathlib import Path

text = Path("response_processing_section.txt").read_text(encoding="utf-8")

checks = {
    "Empty response handling": "response is None or not response.states" in text,
    "Aircraft matching": "item.icao24.lower() == AIRCRAFT_ID" in text,
    "Stale data rejection": "age > 60 or age < -10" in text,
    "Duplicate rejection": "observation_time <= last_observation_timestamp" in text,
    "Recording append": "recording.append(record)" in text,
    "Rolling buffer trimming": "recording.popleft()" in text,
    "AI evaluation": "run_ai()" in text,
    "Post-trigger logging": "append_post_trigger_telemetry(record)" in text,
}

for name, passed in checks.items():
    print(f'{"PASS" if passed else "FAIL"}: {name}')

if not all(checks.values()):
    raise SystemExit("Extraction verification failed.")

print("\nPASS: Response-processing block verified.")
