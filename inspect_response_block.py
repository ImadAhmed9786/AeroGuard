from pathlib import Path

lines = Path("response_processing_section.txt").read_text(
    encoding="utf-8"
).splitlines()

for i, line in enumerate(lines[:15], start=1):
    print(f"{i:3}: {line!r}")

print("\nThis block contains loop-level 'continue' statements.")
print("It must be tested inside a loop, not as a standalone function.")
