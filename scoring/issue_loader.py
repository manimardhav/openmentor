"""
issue_loader.py — loads real issues from data/issues.csv (matplotlib/matplotlib).

Confirmed real columns: issue_id, repo_name, title, body, labels, linked_pr,
resolver, close_date, resolver_is_first_time, affected_files,
centrality_of_affected_files, state, created_at, days_to_close.

Confirmed delimiter: semicolon (;) for both `labels` and `affected_files`.

DATA QUIRKS handled:
1. The real export uses the literal string 'NONE_FOUND' as a sentinel for
   missing values in numeric columns, instead of leaving cells blank.
   pd.notna() does NOT catch this — it's a real, non-null string.
2. Some rows have a genuinely missing (NaN) title/body. `value or ""` does
   NOT safely handle this — NaN is truthy in Python — so it silently lets
   NaN floats through, which later crashes anything that slices or
   concatenates them as if they were text.
"""

import pandas as pd

ISSUES_PATH = "../data/issues.csv"

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
    """Converts a value to a clean string, treating NaN as empty (NOT `value or ""` — see module docstring)."""
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


def load_issues(path: str = ISSUES_PATH) -> list:
    """
    Returns a list of issue dicts: id, title, body (title+body combined),
    labels (list), affected_files (list), centrality_of_affected_files
    (float or None), resolver_is_first_time, state, created_at, days_to_close.
    """
    df = pd.read_csv(path)
    issues = []
    for _, row in df.iterrows():
        title = _safe_str(row.get("title"))
        body_text = _safe_str(row.get("body"))

        issues.append({
            "id": row["issue_id"],
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
        raise RuntimeError(
            "Mock data path has been retired now that real data is available."
        )
    return load_issues()


if __name__ == "__main__":
    issues = load_issues()
    print(f"Loaded {len(issues)} real issues")
    for i in issues[:3]:
        print(f"  #{i['id']}: labels={i['labels']}, affected_files={i['affected_files']}, "
              f"centrality={i['centrality_of_affected_files']}")