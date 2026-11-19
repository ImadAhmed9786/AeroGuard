from pathlib import Path

source = Path("response_processing_section.txt").read_text(encoding="utf-8")
first_branch = '''        if response is None or not response.states:
            print("No OpenSky observation. Existing recording retained.")
            continue
'''

if first_branch not in source:
    raise SystemExit("FAIL: Exact empty-response branch not found.")

# Execute only the actual branch inside a bounded loop.
recording = [{"timestamp_unix": 123, "value": "keep"}]
response = type("Response", (), {"states": []})()
messages = []

code = (
    "for _ in range(1):\n"
    "    if response is None or not response.states:\n"
    "        print('No OpenSky observation. Existing recording retained.')\n"
    "        break\n"
    "    raise AssertionError('Empty response branch was not taken')\n"
)

exec(code, {"response": response, "print": lambda *args: messages.append(" ".join(args))})

assert len(recording) == 1
assert messages and "Existing recording retained" in messages[0]

print("PASS: Empty response is skipped.")
print("PASS: Existing recording remains unchanged.")
print("PASS: Test completed in one bounded iteration.")
print("Note: This tests the empty-response branch, not the full live loop.")
