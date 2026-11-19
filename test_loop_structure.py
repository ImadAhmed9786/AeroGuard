from pathlib import Path

source = Path("monitoring_loop_source.txt").read_text(encoding="utf-8")

checks = {
    "OpenSky request exception handling": 'except Exception as exc' in source and 'api.get_states(icao24=[AIRCRAFT_ID])' in source,
    "Empty response handling": 'response is None or not response.states' in source,
    "Aircraft selection": 'item.icao24.lower() == AIRCRAFT_ID' in source,
    "Incomplete telemetry rejection": 'Incomplete telemetry. Observation skipped.' in source,
    "Stale observation rejection": 'age > 60 or age < -10' in source,
    "Duplicate observation rejection": 'observation_time <= last_observation_timestamp' in source,
    "Rolling recording": 'recording.append(record)' in source and 'recording.popleft()' in source,
    "Periodic snapshot": 'save_rolling_snapshot()' in source,
    "AI evaluation": 'run_ai()' in source,
    "Post-trigger telemetry in finally": 'finally:' in source and 'append_post_trigger_telemetry(record)' in source,
    "Graceful Ctrl+C handling": 'except KeyboardInterrupt:' in source,
}

failed = False
for name, passed in checks.items():
    print(f'{"PASS" if passed else "FAIL"}: {name}')
    failed |= not passed

if failed:
    raise SystemExit("Static safety check failed.")

print("\nALL STATIC SAFETY CHECKS PASSED")
print("Note: This checks source structure, not runtime behaviour.")
