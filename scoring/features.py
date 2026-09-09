"""
features.py — feature extraction for the difficulty-scoring model, rebuilt
for real matplotlib issue data.

Changes from the mock-data version:
- centrality: now reads the REAL precomputed centrality_of_affected_files
  from issues.csv directly. Falls back to graph_metrics_loader.py's lookup
  only when that value is missing.
- referenced_files: now counts REAL affected_files, not a regex guess.
- first_contribution_label: matplotlib's real GFI-equivalent tag is
  "first-contribution" (confirmed from real label data).
- comment_count is GONE — no comment-count column exists in this data.
  Replaced by issue_age_days (from created_at), which IS real data.
"""

import pandas as pd
from graph_metrics_loader import load_graph_metrics, lookup_centrality

FEATURE_COLUMNS = ["centrality", "text_length", "issue_age_days", "first_contribution_label", "referenced_files"]


def centrality_feature(issue: dict, graph_metrics_df: pd.DataFrame = None, cap: float = 0.05) -> float:
    value = issue.get("centrality_of_affected_files")
    if value is None and graph_metrics_df is not None:
        value = lookup_centrality(issue.get("affected_files", []), graph_metrics_df)
    value = value or 0.0
    return min(value, cap) / cap


def text_length_feature(body: str, cap: int = 2000) -> float:
    length = len(body or "")
    return min(length, cap) / cap


def issue_age_feature(created_at, reference_date, cap_days: int = 365) -> float:
    if pd.isna(created_at):
        return 0.0
    created = pd.to_datetime(created_at, utc=True)
    reference = pd.to_datetime(reference_date, utc=True)
    age_days = max((reference - created).days, 0)
    return min(age_days, cap_days) / cap_days


def first_contribution_label_feature(labels: list) -> float:
    labels_lower = [l.lower() for l in (labels or [])]
    return 1.0 if any(
        "first-contribution" in l or "good first issue" in l or "help wanted" in l
        for l in labels_lower
    ) else 0.0


def referenced_files_feature(affected_files: list, cap: int = 10) -> float:
    return min(len(affected_files or []), cap) / cap


def build_feature_row(issue: dict, reference_date, graph_metrics_df: pd.DataFrame = None) -> dict:
    return {
        "issue_id": issue["id"],
        "centrality": centrality_feature(issue, graph_metrics_df),
        "text_length": text_length_feature(issue.get("body", "")),
        "issue_age_days": issue_age_feature(issue.get("created_at"), reference_date),
        "first_contribution_label": first_contribution_label_feature(issue.get("labels", [])),
        "referenced_files": referenced_files_feature(issue.get("affected_files", [])),
    }


def build_feature_dataframe(issues: list, graph_metrics_df: pd.DataFrame = None) -> pd.DataFrame:
    created_dates = [i.get("created_at") for i in issues if pd.notna(i.get("created_at"))]
    reference_date = max(pd.to_datetime(created_dates, utc=True)) if created_dates else pd.Timestamp.now(tz="UTC")

    rows = [build_feature_row(issue, reference_date, graph_metrics_df) for issue in issues]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from issue_loader import load_issues
    issues = load_issues()
    try:
        metrics = load_graph_metrics()
    except FileNotFoundError:
        metrics = None
        print("graph_metrics.csv not found — proceeding without fallback lookup")

    df = build_feature_dataframe(issues, metrics)
    print(df)