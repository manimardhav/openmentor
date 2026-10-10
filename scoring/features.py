"""
features.py — feature extraction for the difficulty-scoring model.

9-repo update:
1. MISSING CENTRALITY is no longer treated as 0. About 70% of issues have no
   centrality (no resolving PR traced). Unknown values are imputed with the median of
   the KNOWN values, and a binary centrality_known feature is added so the model can
   learn how much to trust the centrality column.
2. Raw values (centrality_raw is NaN when unknown, raw age in days, raw file count) are
   exposed as extra columns for truthful explanations. No model uses them.
3. The issue-age cap was raised from 365 to 2000 days: the old cap put 74.7% of issues
   exactly at the cap, so the feature carried almost no information.
4. lookup_centrality() takes repo_name (file_path alone collides across repos).

NOTE: centrality, centrality_known and referenced_files come from the PR that resolved
the issue, so they are unavailable for every OPEN issue. See ablation_study.py.
"""

import statistics
import pandas as pd
from graph_metrics_loader import load_graph_metrics, lookup_centrality

FEATURE_COLUMNS = [
    "centrality", "centrality_known", "text_length",
    "issue_age_days", "first_contribution_label", "referenced_files",
]

CENTRALITY_CAP = 0.05  # observed real values are small
AGE_CAP_DAYS = 2000


def _raw_centrality(issue: dict, graph_metrics_df: pd.DataFrame = None):
    """Returns the real centrality value, or None if genuinely unknown. Never fabricates a 0.0."""
    value = issue.get("centrality_of_affected_files")
    if value is None and graph_metrics_df is not None:
        value = lookup_centrality(issue.get("repo_name"), issue.get("affected_files", []), graph_metrics_df)
    return value


def text_length_feature(body: str, cap: int = 2000) -> float:
    length = len(body or "")
    return min(length, cap) / cap


def issue_age_feature(created_at, reference_date, cap_days: int = AGE_CAP_DAYS) -> float:
    """Age in days, scaled to [0, 1]."""
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


def build_feature_dataframe(issues: list, graph_metrics_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Two passes, deliberately: centrality needs the dataset's own median of
    KNOWN values before any row can be finalized.
    """
    created_dates = [i.get("created_at") for i in issues if pd.notna(i.get("created_at"))]
    reference_date = max(pd.to_datetime(created_dates, utc=True)) if created_dates else pd.Timestamp.now(tz="UTC")

    raw_centralities = [_raw_centrality(issue, graph_metrics_df) for issue in issues]
    known_values = [v for v in raw_centralities if v is not None]
    median_known = statistics.median(known_values) if known_values else 0.0

    rows = []
    for issue, raw_c in zip(issues, raw_centralities):
        centrality_known = raw_c is not None
        c = raw_c if centrality_known else median_known
        created = issue.get("created_at")
        age_raw = max((pd.to_datetime(reference_date, utc=True) - pd.to_datetime(created, utc=True)).days, 0) \
            if pd.notna(created) else float("nan")
        rows.append({
            "issue_id": issue["id"],
            # ---- model features (FEATURE_COLUMNS) ----
            "centrality": min(c, CENTRALITY_CAP) / CENTRALITY_CAP,
            "centrality_known": 1.0 if centrality_known else 0.0,
            "text_length": text_length_feature(issue.get("body", "")),
            "issue_age_days": issue_age_feature(created, reference_date),
            "first_contribution_label": first_contribution_label_feature(issue.get("labels", [])),
            "referenced_files": referenced_files_feature(issue.get("affected_files", [])),
            # ---- raw values, NOT used by any model; for truthful explanations ----
            # centrality_raw is NaN (not 0, not the imputed median) when unknown, so
            # an explanation can say "unknown" instead of inventing a centrality.
            "centrality_raw": raw_c if centrality_known else float("nan"),
            "issue_age_days_raw": age_raw,
            "referenced_files_count": len(issue.get("affected_files") or []),
        })
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
    print(f"Built features for {len(df)} issues")
    print(f"centrality_known rate: {df['centrality_known'].mean():.1%}")
    print(df.head())