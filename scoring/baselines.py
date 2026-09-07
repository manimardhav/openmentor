"""
baselines.py — the two comparison baselines defined in the plan:
  1. GitHub's native "good first issue"/"help wanted" label.
  2. A naive single-feature baseline (comment count only).
A random baseline is included too, as a cheap floor check.
"""

import random
import pandas as pd
from features import has_help_wanted_label


def gfi_label_baseline(issues: list) -> pd.DataFrame:
    """Ranks issues by whether they carry a "good first issue" label."""
    rows = []
    for issue in issues:
        has_label = has_help_wanted_label(issue.get("labels", []))
        rows.append({"issue_id": issue["id"], "gfi_labeled": has_label})
    df = pd.DataFrame(rows)
    return df.sort_values("gfi_labeled", ascending=False, kind="stable").reset_index(drop=True)


def single_feature_baseline(issues: list, feature: str = "comments") -> pd.DataFrame:
    """Naive baseline: rank by ONE raw feature only, no model."""
    rows = []
    for issue in issues:
        rows.append({"issue_id": issue["id"], feature: issue.get(feature, 0)})
    df = pd.DataFrame(rows)
    return df.sort_values(feature, ascending=True, kind="stable").reset_index(drop=True)


def random_baseline(issues: list, seed: int = 42) -> pd.DataFrame:
    """Floor baseline — if the real system can't beat random, nothing else matters."""
    ids = [issue["id"] for issue in issues]
    rng = random.Random(seed)
    rng.shuffle(ids)
    return pd.DataFrame({"issue_id": ids})


if __name__ == "__main__":
    from sanity_check import REAL_ISSUES

    print("GFI label baseline ranking:")
    print(gfi_label_baseline(REAL_ISSUES).to_string(index=False))

    print("\nSingle-feature (comment count) baseline ranking:")
    print(single_feature_baseline(REAL_ISSUES).to_string(index=False))

    print("\nRandom baseline ranking:")
    print(random_baseline(REAL_ISSUES).to_string(index=False))