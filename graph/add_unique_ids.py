"""
add_unique_ids.py — ONE-TIME fix for issue_id collisions across repos.

issue_id is just the raw GitHub issue number, which is only unique WITHIN
one repository — issue #6144 exists independently in both networkx and
huggingface/datasets, for example. This adds a genuinely unique column,
`unique_id` = "<repo_name>#<issue_id>", to every data file that has both
repo_name and issue_id, so the whole team uses the same composite key
instead of each person re-deriving it their own way downstream.

Run this once. Future runs of pull_issues.py and build_ground_truth.py
already include this column automatically (see the code changes there).
"""

import csv
from pathlib import Path
from repo_config import OUTPUT_DIR

FILES_TO_FIX = ["issues.csv", "ground_truth.csv", "baseline_gfi.csv", "baseline_naive.csv"]


def add_unique_id_column(path: Path):
    if not path.exists():
        print(f"Skipping {path.name} — file not found.")
        return

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"Skipping {path.name} — empty file.")
        return

    if "unique_id" in rows[0]:
        print(f"Skipping {path.name} — already has unique_id.")
        return

    for row in rows:
        row["unique_id"] = f"{row['repo_name']}#{row['issue_id']}"

    # put unique_id right after issue_id, for readability
    old_fieldnames = list(rows[0].keys())
    old_fieldnames.remove("unique_id")
    idx = old_fieldnames.index("issue_id") + 1
    fieldnames = old_fieldnames[:idx] + ["unique_id"] + old_fieldnames[idx:]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Added unique_id to {len(rows)} rows in {path.name}")


if __name__ == "__main__":
    for filename in FILES_TO_FIX:
        add_unique_id_column(Path(OUTPUT_DIR) / filename)