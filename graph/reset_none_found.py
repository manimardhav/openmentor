"""
reset_none_found.py — ONE-TIME helper to run after fixing the cross-repo
matching bug in enrich_issue_links.py.

Your previous enrich_issue_links.py run marked some issues "NONE_FOUND"
that were actually caused by the bug (grabbing a same-numbered PR from a
totally different repository, which then 404'd). This resets those rows
back to blank, so the next run of enrich_issue_links.py will recheck them
with the corrected logic — without touching rows that already matched
successfully (no need to redo those).

Run this ONCE, then run enrich_issue_links.py again as normal.
"""

import csv
from pathlib import Path
from repo_config import OUTPUT_DIR

ISSUES_FILE = Path(OUTPUT_DIR) / "issues.csv"
NOT_FOUND = "NONE_FOUND"

with open(ISSUES_FILE, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
fieldnames = list(rows[0].keys())

reset_count = 0
for row in rows:
    if row["affected_files"] == NOT_FOUND:
        row["linked_pr"] = ""
        row["affected_files"] = ""
        row["centrality_of_affected_files"] = ""
        reset_count += 1

with open(ISSUES_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Reset {reset_count} previously-unmatched rows — they'll be rechecked "
      f"next time you run enrich_issue_links.py.")