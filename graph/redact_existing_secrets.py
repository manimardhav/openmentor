"""
redact_existing_secrets.py — ONE-TIME cleanup for issues.csv already on disk.

GitHub's push protection found a real leaked Hugging Face token sitting
inside an issue's body text (someone pasted it into a GitHub issue by
accident — not anything you did). This scans your existing issues.csv and
redacts any matching secret patterns in the title/body columns, in place.

Run this once, then re-commit. pull_issues.py has also been updated so any
FUTURE pull automatically redacts these before ever saving to disk.
"""

import csv
from pathlib import Path
from repo_config import OUTPUT_DIR
from pull_issues import redact_secrets

ISSUES_FILE = Path(OUTPUT_DIR) / "issues.csv"

with open(ISSUES_FILE, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
fieldnames = list(rows[0].keys())

redacted_count = 0
for row in rows:
    for field in ("title", "body"):
        original = row.get(field, "") or ""
        cleaned = redact_secrets(original)
        if cleaned != original:
            row[field] = cleaned
            redacted_count += 1

with open(ISSUES_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Redacted {redacted_count} field(s) containing secret-like patterns.")
print(f"Updated {ISSUES_FILE}")