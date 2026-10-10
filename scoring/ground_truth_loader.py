"""
ground_truth_loader.py — loads data/ground_truth.csv: real outcomes per closed
issue (had a GFI label, resolved by a first-time contributor, days to close).

9-repo update: path anchored to this file's location, and everything keys on
`unique_id` (not the cross-repo-colliding raw `issue_id`).
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.csv"


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def build_real_relevant_set(ground_truth_df: pd.DataFrame = None) -> set:
    """
    An issue counts as "relevant" (a genuinely good newcomer match) if it
    was ACTUALLY resolved by a first-time contributor. Returns unique_id
    values, matching issue_loader.py's issue["id"]. Issues whose resolver is
    unknown (blank) are not counted as relevant.

    NOTE: this is a repo-wide relevant set, not personalized to one
    contributor's specific skills.
    """
    ground_truth_df = ground_truth_df if ground_truth_df is not None else load_ground_truth()
    relevant = ground_truth_df[ground_truth_df["resolved_by_newcomer"] == True]
    return set(relevant["unique_id"])


def real_gfi_baseline_ranking(ground_truth_df: pd.DataFrame = None) -> pd.DataFrame:
    """Kept for compatibility. Closed issues only; baselines.py uses baseline_gfi.csv for open issues."""
    ground_truth_df = ground_truth_df if ground_truth_df is not None else load_ground_truth()
    return (ground_truth_df[["unique_id", "had_gfi_label"]]
            .sort_values("had_gfi_label", ascending=False, kind="stable")
            .reset_index(drop=True))


if __name__ == "__main__":
    gt = load_ground_truth()
    print(f"Loaded {len(gt)} ground-truth records")
    print(f"Resolved by newcomer: {gt['resolved_by_newcomer'].sum()} / {len(gt)} "
          f"({gt['resolved_by_newcomer'].mean():.1%})")
    print(f"Had GFI label: {gt['had_gfi_label'].sum()} / {len(gt)}")

    relevant = build_real_relevant_set(gt)
    print(f"\nReal relevant set size: {len(relevant)}")
    print(f"Sample: {list(relevant)[:3]}")