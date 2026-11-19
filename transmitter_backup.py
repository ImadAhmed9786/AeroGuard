
from pathlib import Path
from datetime import datetime, timezone
import shutil
import csv


GROUND_STATION_DIR = Path("data/ground_station")


def activate_transmitter(aircraft_id, callsign, issues, recording):
    """
    Simulates transmitting an emergency report and recording
    to the AeroGuard ground station.

    Returns True only when the report and requested recording
    have both been saved successfully.
    """

    aircraft_id = str(aircraft_id)
    callsign = str(callsign)
    aircraft_dir = GROUND_STATION_DIR / aircraft_id
    aircraft_dir.mkdir(parents=True, exist_ok=True)

    report_path = aircraft_dir / "emergency_report.txt"
    recording_path = aircraft_dir / "previous_45_minutes.csv"

    detection_time = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    if isinstance(issues, str):
        issues = [issues]
    elif issues is None:
        issues = []

    print("\n" + "=" * 40)
    print("        AEROGUARD TRANSMITTER")
    print("=" * 40)
    print("\nEMERGENCY TRANSMISSION")
    print(f"\nAircraft ID: {aircraft_id}")
    print(f"Callsign: {callsign}")
    print(f"Detection Time: {detection_time}")
    print("\nStatus: CRITICAL")
    print("Transmission: ACTIVE")
    print("\nDetected Issues:")

    if not issues:
        print("  No issue details supplied.")

    for issue in issues:
        if isinstance(issue, dict):
            description = issue.get("issue", "Unspecified issue")
            severity = issue.get("severity")

            if severity:
                print(f"\n  Issue: {description}")
                print(f"  Severity: {severity}")
            else:
                print(f"\n  Issue: {description}")
        else:
            print(f"\n  Issue: {issue}")

    # Step 1: Save the emergency report.
    report_saved = False

    try:
        with report_path.open("w", encoding="utf-8") as file:
            file.write("AEROGUARD EMERGENCY REPORT\n")
            file.write("=" * 27 + "\n\n")
            file.write(f"Aircraft ID: {aircraft_id}\n")
            file.write(f"Callsign: {callsign}\n")
            file.write(f"Detection Time: {detection_time}\n")
            file.write("Status: CRITICAL\n")
            file.write("Transmission: ACTIVE\n\n")
            file.write("Detected Issues:\n")

            for issue in issues:
                if isinstance(issue, dict):
                    description = issue.get(
                        "issue", "Unspecified issue"
                    )
                    severity = issue.get("severity", "")

                    if severity:
                        file.write(
                            f"- {description} "
                            f"(Severity: {severity})\n"
                        )
                    else:
                        file.write(f"- {description}\n")
                else:
                    file.write(f"- {issue}\n")

            file.write("\n")

        report_saved = True
        print(f"\nEmergency report saved:\n{report_path}")

    except OSError as error:
        print(f"\nREPORT SAVE FAILED: {error}")

    # Step 2: Transfer the requested recording.
    recording_transferred = recording is None

    if recording is not None:
        try:
            if isinstance(recording, (str, Path)):
                source_path = Path(recording)

                if not source_path.is_file():
                    raise FileNotFoundError(
                        f"Recording file not found: {source_path}"
                    )

                # Avoid copying a file onto itself.
                if source_path.resolve() != recording_path.resolve():
                    shutil.copy2(source_path, recording_path)

                recording_transferred = recording_path.is_file()

            else:
                # Supports iterable recording rows, including
                # lists of dictionaries and DataFrame-like objects.
                if hasattr(recording, "to_csv"):
                    recording.to_csv(recording_path, index=False)
                else:
                    rows = list(recording)

                    if not rows:
                        raise ValueError("Recording contains no rows.")

                    if isinstance(rows[0], dict):
                        with recording_path.open(
                            "w", newline="", encoding="utf-8"
                        ) as file:
                            writer = csv.DictWriter(
                                file, fieldnames=list(rows[0].keys())
                            )
                            writer.writeheader()
                            writer.writerows(rows)
                    else:
                        with recording_path.open(
                            "w", newline="", encoding="utf-8"
                        ) as file:
                            writer = csv.writer(file)
                            writer.writerows(rows)

                recording_transferred = recording_path.is_file()

            if recording_transferred:
                print(
                    f"\nRecording transferred successfully:\n"
                    f"{recording_path}"
                )
            else:
                print("\nRECORDING TRANSFER FAILED.")

        except (OSError, ValueError, TypeError) as error:
            recording_transferred = False
            print(f"\nRECORDING TRANSFER FAILED: {error}")

    else:
        print("\nNo recording supplied; recording transfer skipped.")

    # Step 3: Report the actual result.
    print("\n" + "-" * 40)
    print("GROUND STATION STATUS")
    print(f"Emergency report saved: {report_saved}")

    if recording is None:
        print("Recording transfer: NOT REQUESTED")
    else:
        print(f"Recording transferred: {recording_transferred}")

    success = report_saved and recording_transferred

    if success:
        print("\nSIMULATION RESULT: SUCCESS")
        print("All requested files were saved.")
    else:
        print("\nSIMULATION RESULT: FAILED")
        print("One or more required operations failed.")

    print("-" * 40)

    return success
