"""
issue_loader.py — single entry point for issue data, mirroring the
placeholder/real swap pattern in build_placeholder_graph.py.
"""

import json
from sanity_check import REAL_ISSUES as PLACEHOLDER_ISSUES


def load_real_issues(path: str = "../data/issues.json") -> list:
    """
    Loads Person A's real issue export.
    Assumed shape:
        [{"id": ..., "title": ..., "body": ..., "comments": ...,
          "labels": [...], "files_touched": [...]}, ...]
    """
    with open(path) as f:
        issues = json.load(f)
    return issues


def get_issues(use_real_data: bool = False) -> list:
    """Single entry point — import this everywhere instead of hardcoded lists."""
    if use_real_data:
        return load_real_issues()
    return PLACEHOLDER_ISSUES


if __name__ == "__main__":
    issues = get_issues(use_real_data=False)
    print(f"Loaded {len(issues)} issues (placeholder mode)")
    for i in issues[:3]:
        print(f"  #{i['id']}: {i['body'][:60]}")