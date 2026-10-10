"""
issue_loader.py — loads real issues from data/issues.csv.

9-repo update:
1. PATH FIX: anchored to this file's own location, so it works from any directory.
2. COLLISION FIX: the raw issue_id collides across repos (1,901 of 9,000 rows share
   an id with an issue in another repo). issue["id"] now holds unique_id
   ("<repo_name>#<issue_id>"), which is unique, so every dict key / join in scoring/
   is collision-safe. The raw number is kept as issue["raw_issue_number"].

Data quirks handled: 'NONE_FOUND' sentinel strings in numeric columns; missing (NaN)
title/body (note: `value or ""` does NOT work, because NaN is truthy in Python).
labels and affected_files are semicolon-separated.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ISSUES_PATH = DATA_DIR / "issues.csv"

NULL_SENTINELS = {"none_found", "n/a", "na", "none", "null", ""}


def _safe_float(value):
    """Converts a value to float, treating known sentinel strings (and actual NaN) as missing."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in NULL_SENTINELS:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_str(value) -> str:
    """Converts a value to a clean string, treating NaN as empty (NOT `value or ""`)."""
    if pd.isna(value):
        return ""
    return str(value)


def _split_semicolon(value) -> list:
    """Splits a semicolon-delimited string into a clean list; handles NaN/empty/sentinel values safely."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in NULL_SENTINELS:
        return []
    items = [item.strip() for item in text.split(";") if item.strip()]
    return [item for item in items if item.lower() not in NULL_SENTINELS]


def load_issues(path: Path = ISSUES_PATH) -> list:
    """
    Returns a list of issue dicts:
      id                   -> unique_id (collision-free primary key; "<repo>#<issue_id>")
      raw_issue_number     -> the original per-repo issue_id, for display/linking only
      repo_name
      title, body (title+body combined)
      labels (list), affected_files (list)
      centrality_of_affected_files (float or None)
      resolver_is_first_time, state, created_at, days_to_close
    """
    df = pd.read_csv(path)

    if not df["unique_id"].is_unique:
        dupes = df[df.duplicated(subset=["unique_id"], keep=False)]
        raise ValueError(
            f"data/issues.csv's unique_id column is not actually unique "
            f"({len(dupes)} colliding rows) — check with Member 3 before proceeding, "
            f"since the whole point of this column is to be collision-free."
        )

    issues = []
    for _, row in df.iterrows():
        title = _safe_str(row.get("title"))
        body_text = _safe_str(row.get("body"))

        issues.append({
            "id": row["unique_id"],
            "raw_issue_number": row["issue_id"],
            "repo_name": _safe_str(row.get("repo_name")),
            "title": title,
            "body": f"{title} {body_text}".strip(),
            "labels": _split_semicolon(row.get("labels")),
            "affected_files": _split_semicolon(row.get("affected_files")),
            "centrality_of_affected_files": _safe_float(row.get("centrality_of_affected_files")),
            "resolver_is_first_time": bool(row.get("resolver_is_first_time")) if pd.notna(row.get("resolver_is_first_time")) else None,
            "state": row.get("state"),
            "created_at": row.get("created_at"),
            "days_to_close": _safe_float(row.get("days_to_close")),
        })
    return issues


def get_issues(use_real_data: bool = True) -> list:
    if not use_real_data:
        raise RuntimeError("Mock data path has been retired now that real data is available.")
    return load_issues()


if __name__ == "__main__":
    issues = load_issues()
    print(f"Loaded {len(issues)} real issues across {len(set(i['repo_name'] for i in issues))} repos")
    for i in issues[:3]:
        print(f"  {i['id']} (raw #{i['raw_issue_number']}): labels={i['labels'][:2]}, "
              f"centrality={i['centrality_of_affected_files']}")