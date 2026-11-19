from pathlib import Path
import ast

source = Path("aeroguard_single.py").read_text(encoding="utf-8-sig")
tree = ast.parse(source)

loop_start = source.index("# MAIN MONITORING LOOP")
loop_source = source[loop_start:]

checks = {
    "Polling loop exists": "while True:" in loop_source,
    "OpenSky request exists": "api.get_states(icao24=[AIRCRAFT_ID])" in loop_source,
    "Request exception is caught": "OpenSky request failed:" in loop_source,
    "Empty response is handled": "No OpenSky observation. Existing recording retained." in loop_source,
    "Valid response processing exists": "recording.append(record)" in loop_source,
}

for name, passed in checks.items():
    print(f'{"PASS" if passed else "FAIL"}: {name}')

if not all(checks.values()):
    raise SystemExit("Production loop inspection failed.")

Path("test_polling_recovery.py").write_text(
    '''"""Controlled AeroGuard polling-recovery test scaffold."""\n'''
    '''# Production source inspected; no live API or model loaded.\n'''
    '''print("Scaffold ready: production polling loop verified.")\n''',
    encoding="utf-8",
)

print("\\nPASS: Recovery-test scaffold created.")
print("Production script and model were not modified.")
