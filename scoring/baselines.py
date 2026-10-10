"""
baselines.py — comparison baselines.

- gfi_label_baseline / single_feature_baseline read Member 3's precomputed OPEN-issue
  files (baseline_gfi.csv, baseline_naive.csv) for "what to recommend today". Both
  already use unique_id as the key.
- age_baseline_from_issues / random_baseline work on whatever issue list is passed in
  (used for historical evaluation, which ranks closed issues).
"""

import random
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASELINE_GFI_PATH = DATA_DIR / "baseline_gfi.csv"
BASELINE_NAIVE_PATH = DATA_DIR / "baseline_naive.csv"


def gfi_label_baseline(path: Path = BASELINE_GFI_PATH) -> list:
    """unique_ids of currently-open GFI-labeled issues, in the file's own order (only ~25 across all repos)."""
    df = pd.read_csv(path)
    return df["unique_id"].tolist()


def single_feature_baseline(path: Path = BASELINE_NAIVE_PATH) -> list:
    """unique_ids of open issues ranked oldest-first, using Member 3's precomputed rank column."""
    df = pd.read_csv(path)
    return df.sort_values("rank")["unique_id"].tolist()


def age_baseline_from_issues(issues: list) -> list:
    """
    Age ranking (oldest first) over WHATEVER issue population is passed in.
    Used by historical evaluation, which ranks closed issues; the open-issues-only
    baseline_naive.csv would be wrong there, because an open issue can never be
    "already resolved".
    """
    rows = [(issue["id"], issue.get("created_at")) for issue in issues]
    rows = [(uid, pd.to_datetime(created, utc=True)) for uid, created in rows if pd.notna(created)]
    rows.sort(key=lambda x: x[1])  # oldest first
    return [uid for uid, _ in rows]


def random_baseline(issues: list, seed: int = 42) -> list:
    """issues: list of issue dicts from issue_loader.py."""
    ids = [issue["id"] for issue in issues]
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids


if __name__ == "__main__":
    from issue_loader import load_issues

    issues = load_issues()
    open_issues = [i for i in issues if i.get("state") == "open"]
    print(f"Open issues (for random baseline): {len(open_issues)}")

    gfi_ranked = gfi_label_baseline()
    print(f"\nGFI-label baseline: {len(gfi_ranked)} issues")
    print(f"Sample: {gfi_ranked[:5]}")

    naive_ranked = single_feature_baseline()
    print(f"\nIssue-age baseline: {len(naive_ranked)} issues")
    print(f"Sample (oldest first): {naive_ranked[:5]}")

    random_ranked = random_baseline(open_issues)
    print(f"\nRandom baseline: {len(random_ranked)} issues")
    print(f"Sample: {random_ranked[:5]}")