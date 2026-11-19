from pathlib import Path
import ast
import textwrap

path = Path("aeroguard_single.py")
source = path.read_text(encoding="utf-8")

loop_marker = "# MAIN MONITORING LOOP"
loop_start = source.find(loop_marker)
if loop_start < 0:
    raise SystemExit("STOP: Main loop marker not found.")

start_marker = "        if response is None or not response.states:"
start = source.find(start_marker, loop_start)
end_marker = "        try:\n            run_ai()\n        finally:\n            append_post_trigger_telemetry(record)"
end = source.find(end_marker, start)

if start < 0 or end < 0:
    raise SystemExit("STOP: Response-processing boundaries not found.")

end += len(end_marker)
block = source[start:end]

# Convert the loop's response-processing block into a reusable function.
dedented = textwrap.dedent(block)
dedented = "\n".join(
    line.replace("continue", "return False", 1)
    if line.strip() == "continue" else line
    for line in dedented.splitlines()
)
function = (
    "def process_response(response):\n"
    "    global last_observation_timestamp, last_snapshot_time\n"
    + textwrap.indent(dedented, "    ")
    + "\n    return True\n\n\n"
)

replacement = "        process_response(response)"
updated = source[:loop_start] + function + source[loop_start:start] + replacement + source[end:]

# Parse before saving anything.
ast.parse(updated)
Path("aeroguard_single_refactored_preview.py").write_text(
    updated, encoding="utf-8"
)

print("PASS: Preview generated and Python syntax is valid.")
print(f"Original response block: {len(block.splitlines())} lines")
print("Preview saved as aeroguard_single_refactored_preview.py")
print("Production script remains unchanged.")
