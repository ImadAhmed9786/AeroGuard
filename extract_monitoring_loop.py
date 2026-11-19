from pathlib import Path

source = Path("aeroguard_single.py").read_text(encoding="utf-8-sig")
marker = "# MAIN MONITORING LOOP"

if marker not in source:
    raise SystemExit("ERROR: Main-loop marker not found.")

loop = source.split(marker, 1)[1]

Path("monitoring_loop_source.txt").write_text(
    marker + loop,
    encoding="utf-8"
)

print("Actual production monitoring-loop source extracted.")
print("Lines:", len(loop.splitlines()))
print("Production script unchanged.")
