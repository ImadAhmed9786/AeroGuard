from pathlib import Path

source = Path("aeroguard_single.py").read_text(encoding="utf-8-sig")
loop_start = source.index("# MAIN MONITORING LOOP")
start = source.index(
    "        if response is None or not response.states:",
    loop_start,
)
end_marker = "        try:\n            run_ai()\n        finally:\n            append_post_trigger_telemetry(record)"
end = source.index(end_marker, start) + len(end_marker)

lines = source[start:end].splitlines()
print("Extracted lines:", len(lines))
print("First line indentation:", len(lines[0]) - len(lines[0].lstrip()))
print("Continue statements:", sum(line.strip() == "continue" for line in lines))
print("Last lines:")
for line in lines[-8:]:
    print(repr(line))

Path("response_processing_section.txt").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
print("PASS: Clean extraction refreshed. Production source unchanged.")
