"""
baselines.py — comparison baselines, rebuilt for real data.

- gfi_label_baseline: uses ground_truth.csv's authoritative had_gfi_label.
- single_feature_baseline: comment count doesn't exist in this data —
  switched to issue age (from created_at).
"""

import random
import pandas as pd
from ground_truth_loader import load_ground_truth


def gfi_label_baseline(ground_truth_df: pd.DataFrame = None) -> pd.DataFrame:
    ground_truth_df = ground_truth_df if ground_truth_df is not None else load_ground_truth()
    return (ground_truth_df[["issue_id", "had_gfi_label"]]
            .sort_values("had_gfi_label", ascending=False, kind="stable")
            .reset_index(drop=True))


def single_feature_baseline(issues: list, feature: str = "issue_age_days") -> pd.DataFrame:
    rows = []
    for issue in issues:
        created = issue.get("created_at")
        rows.append({"issue_id": issue["id"], "created_at": created})
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df.sort_values("created_at", ascending=True, kind="stable").reset_index(drop=True)


def random_baseline(issues: list, seed: int = 42) -> pd.DataFrame:
    ids = [issue["id"] for issue in issues]
    rng = random.Random(seed)
    rng.shuffle(ids)
    return pd.DataFrame({"issue_id": ids})


if __name__ == "__main__":
    from issue_loader import load_issues
    issues = load_issues()

    print("Real GFI label baseline ranking:")
    print(gfi_label_baseline().to_string(index=False))

    print("\nIssue-age baseline ranking:")
    print(single_feature_baseline(issues).to_string(index=False))

    print("\nRandom baseline ranking:")
    print(random_baseline(issues).to_string(index=False))