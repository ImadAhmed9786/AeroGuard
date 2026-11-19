from pathlib import Path

path = Path("aeroguard_single.py")
source = path.read_text(encoding="utf-8")

old = '''        transmission_ok = activate_transmitter(
            aircraft_id=AIRCRAFT_ID,
            callsign=CALLSIGN,
            issues=issues,
            recording=recording_file,
        )

        print(
            "Emergency report operation: "
            f"{'SUCCESS' if transmission_ok else 'FAILED'}"
        )
'''

new = '''        try:
            transmission_ok = activate_transmitter(
                aircraft_id=AIRCRAFT_ID,
                callsign=CALLSIGN,
                issues=issues,
                recording=recording_file,
            )
        except Exception as error:
            transmission_ok = False
            print(f"TRANSMITTER ERROR: {error}")

        print(
            "Emergency report operation: "
            f"{'SUCCESS' if transmission_ok else 'FAILED'}"
        )
'''

if source.count(old) != 1:
    raise SystemExit(
        "PATCH STOPPED: Expected transmitter block not found exactly once. "
        "No changes made."
    )

path.write_text(source.replace(old, new, 1), encoding="utf-8")
print("PATCH APPLIED: AI-triggered transmitter errors are handled.")
