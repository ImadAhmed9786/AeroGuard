def process_response(response):
    global last_observation_timestamp, last_snapshot_time
    if response is None or not response.states:
        print('No OpenSky observation. Existing recording retained.')
        return False
    state = next((item for item in response.states if item.icao24 and item.icao24.lower() == AIRCRAFT_ID), None)
    if state is None:
        print('Selected aircraft not present in response.')
        return False
    required = (state.latitude, state.longitude, state.baro_altitude, state.velocity, state.true_track, state.vertical_rate, state.time_position)
    if any((value is None for value in required)):
        print('Incomplete telemetry. Observation skipped.')
        return False
    observation_time = float(state.time_position)
    current_time = datetime.now(timezone.utc).timestamp()
    age = current_time - observation_time
    if age > 60 or age < -10:
        print(f'Stale/future observation skipped. Age: {age:.1f}s')
        return False
    if last_observation_timestamp is not None and observation_time <= last_observation_timestamp:
        print('Duplicate or older observation skipped.')
        return False
    last_observation_timestamp = observation_time
    timestamp = datetime.fromtimestamp(observation_time, timezone.utc).isoformat()
    record = {'timestamp': timestamp, 'timestamp_unix': observation_time, 'icao24': state.icao24, 'callsign': (state.callsign or '').strip(), 'latitude': state.latitude, 'longitude': state.longitude, 'baro_altitude': state.baro_altitude, 'geo_altitude': state.geo_altitude, 'velocity': float(state.velocity), 'heading': float(state.true_track), 'vertical_rate': float(state.vertical_rate), 'on_ground': state.on_ground, 'squawk': state.squawk}
    recording.append(record)
    ai_history.append(record)
    cutoff = observation_time - RECORDING_SECONDS
    while recording and recording[0]['timestamp_unix'] < cutoff:
        recording.popleft()
    print(f'{timestamp[11:19]} UTC | Recorded {len(recording)} observations | AI batch {len(ai_history)}/{AI_WINDOW} | ALT {state.baro_altitude:.1f} m | SPEED {state.velocity:.1f} m/s')
    snapshot_now = time.monotonic()
    if snapshot_now - last_snapshot_time >= SNAPSHOT_INTERVAL_SECONDS:
        save_rolling_snapshot()
        last_snapshot_time = snapshot_now
    try:
        run_ai()
    finally:
        append_post_trigger_telemetry(record)
    return True
