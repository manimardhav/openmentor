"""
ground_truth_loader.py — replaces mock_relevance.py entirely.

data/ground_truth.csv gives REAL outcomes per issue: whether it had a GFI
label, whether it was actually resolved by a first-time contributor, and
how many days it took to close.

Confirmed real columns: issue_id, repo_name, title, had_gfi_label,
resolved_by_newcomer, days_to_close.
"""

import pandas as pd

GROUND_TRUTH_PATH = "../data/ground_truth.csv"


def load_ground_truth(path: str = GROUND_TRUTH_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def build_real_relevant_set(ground_truth_df: pd.DataFrame = None) -> set:
    """
    Real replacement for mock_relevance.py's guessed set: an issue counts
    as "relevant" (a genuinely good newcomer match) if it was ACTUALLY
    resolved by a first-time contributor.

    NOTE: this is a repo-wide relevant set, not personalized to one
    contributor's specific skills. Personalizing further would need
    resolver skill data, which isn't in this file — ask Person C if that
    could be added later.
    """
    ground_truth_df = ground_truth_df if ground_truth_df is not None else load_ground_truth()
    relevant = ground_truth_df[ground_truth_df["resolved_by_newcomer"] == True]
    return set(relevant["issue_id"])


def real_gfi_baseline_ranking(ground_truth_df: pd.DataFrame = None) -> pd.DataFrame:
    ground_truth_df = ground_truth_df if ground_truth_df is not None else load_ground_truth()
    return (ground_truth_df[["issue_id", "had_gfi_label"]]
            .sort_values("had_gfi_label", ascending=False, kind="stable")
            .reset_index(drop=True))


if __name__ == "__main__":
    gt = load_ground_truth()
    print(f"Loaded {len(gt)} ground-truth records")
    print(f"Resolved by newcomer: {gt['resolved_by_newcomer'].sum()} / {len(gt)}")
    print(f"Had GFI label: {gt['had_gfi_label'].sum()} / {len(gt)}")

    relevant = build_real_relevant_set(gt)
    print(f"\nReal relevant set size: {len(relevant)}")