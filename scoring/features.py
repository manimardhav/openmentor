"""
features.py — Week 2/3: Feature extraction for the difficulty-scoring model.

Turns a raw GitHub issue (+ the repo's dependency graph) into a numeric
feature vector. Keep every feature normalized to roughly [0, 1] so the
weighted-sum formula (Week 2) and the logistic regression (Week 3) both
behave sensibly without one feature dominating.
"""

import re
import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Graph-centrality features
# ---------------------------------------------------------------------------

def compute_centrality_scores(dependency_graph: nx.Graph) -> dict:
    """
    Precompute centrality once for the whole repo graph (expensive), then
    look values up per-issue. Returns {node_name: centrality_score}.

    Betweenness centrality tends to capture "structurally important" files
    better than degree centrality, but is O(V*E) — for large repos, swap in
    nx.betweenness_centrality(G, k=100) to sample instead of computing exactly.
    """
    if dependency_graph.number_of_nodes() == 0:
        return {}
    betweenness = nx.betweenness_centrality(dependency_graph)
    # Normalize to [0, 1] via min-max (betweenness is already ~bounded, but
    # repo-specific rescaling makes cross-repo comparisons fairer).
    values = list(betweenness.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 0.0 for k in betweenness}
    return {k: (v - lo) / (hi - lo) for k, v in betweenness.items()}


def issue_centrality_feature(issue_files: list, centrality_scores: dict) -> float:
    """
    An issue may touch multiple files. Use the MAX centrality among touched
    files as the feature — an issue that touches one highly-central file is
    harder than one touching several peripheral ones, even if the average
    looks similar.
    """
    if not issue_files:
        return 0.0
    scores = [centrality_scores.get(f, 0.0) for f in issue_files]
    return max(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# 2. Text / metadata features
# ---------------------------------------------------------------------------

def text_length_feature(issue_body: str, cap: int = 2000) -> float:
    """Normalized issue-body length. Cap avoids outlier essays dominating."""
    length = len(issue_body or "")
    return min(length, cap) / cap


def comment_count_feature(num_comments: int, cap: int = 20) -> float:
    """More discussion often (not always) signals more complexity/disagreement."""
    return min(num_comments, cap) / cap


def has_help_wanted_label(labels: list) -> float:
    labels_lower = [l.lower() for l in (labels or [])]
    return 1.0 if any("help wanted" in l or "good first issue" in l for l in labels_lower) else 0.0


def referenced_files_feature(issue_body: str, cap: int = 10) -> float:
    """
    Rough heuristic: count filepath-like tokens (e.g. `src/foo.py`) mentioned
    in the issue body as a proxy for "how much of the codebase is implicated."
    """
    pattern = r"[\w\-/]+\.\w{1,5}"  # crude filepath matcher
    matches = re.findall(pattern, issue_body or "")
    return min(len(matches), cap) / cap


# ---------------------------------------------------------------------------
# 3. Assemble full feature row for one issue
# ---------------------------------------------------------------------------

def build_feature_row(issue: dict, centrality_scores: dict) -> dict:
    """
    issue: dict with keys — id, body, comments, labels, files_touched
    Returns a flat dict ready to append into a DataFrame.
    """
    return {
        "issue_id": issue["id"],
        "centrality": issue_centrality_feature(issue.get("files_touched", []), centrality_scores),
        "text_length": text_length_feature(issue.get("body", "")),
        "comment_count": comment_count_feature(issue.get("comments", 0)),
        "help_wanted_label": has_help_wanted_label(issue.get("labels", [])),
        "referenced_files": referenced_files_feature(issue.get("body", "")),
    }


def build_feature_dataframe(issues: list, dependency_graph: nx.Graph) -> pd.DataFrame:
    """Batch version: run this once per repo pull."""
    centrality_scores = compute_centrality_scores(dependency_graph)
    rows = [build_feature_row(issue, centrality_scores) for issue in issues]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Tiny smoke test with a toy graph + toy issues
    G = nx.Graph()
    G.add_edges_from([("api.py", "auth.py"), ("api.py", "db.py"), ("db.py", "utils.py")])

    toy_issues = [
        {"id": 1, "body": "Fix typo in README.md", "comments": 0, "labels": ["good first issue"], "files_touched": ["README.md"]},
        {"id": 2, "body": "Refactor db.py connection pooling, touches api.py and auth.py too, see src/db.py:120", "comments": 12, "labels": ["bug"], "files_touched": ["db.py", "api.py"]},
    ]

    df = build_feature_dataframe(toy_issues, G)
    print(df)
