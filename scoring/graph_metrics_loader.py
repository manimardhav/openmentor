"""
graph_metrics_loader.py — loads the precomputed per-file centrality metrics
(data/graph_metrics.csv) and looks them up for an issue's affected files.

9-repo update:
1. PATH FIX: anchored to this file's own location.
2. The index is (repo_name, file_path), because file_path alone is not unique across
   repos (e.g. setup.py appears in several repos).
3. Paths are normalized to forward slashes: issues.csv uses forward slashes while
   graph_metrics.csv uses backslashes for nested paths, so exact comparison never matched.
4. A file that is not found returns None, never 0.0 ("unknown" is not "not central").
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GRAPH_METRICS_PATH = DATA_DIR / "graph_metrics.csv"


def _normalize_path(path: str) -> str:
    """Forward slashes, consistently, so paths from different sources compare equal."""
    return str(path).replace("\\", "/").strip()


def load_graph_metrics(path: Path = GRAPH_METRICS_PATH) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by (repo_name, normalized_file_path) with
    betweenness/pagerank/degree columns.
    """
    df = pd.read_csv(path)
    df["file_path"] = df["file_path"].apply(_normalize_path)
    return df.set_index(["repo_name", "file_path"])


def lookup_centrality(repo_name: str, file_paths: list, metrics_df: pd.DataFrame, metric: str = "betweenness"):
    """
    Given a repo and a list of file paths an issue touches, returns the
    MAX centrality among them, or None if genuinely not found (NOT 0.0 —
    0.0 would wrongly claim "confirmed not central" when the real
    situation is "we don't have data for this file").
    """
    if not file_paths:
        return None
    values = []
    for fp in file_paths:
        key = (repo_name, _normalize_path(fp))
        if key in metrics_df.index:
            v = metrics_df.loc[key, metric]
            if isinstance(v, pd.Series):  # defensive: duplicate (repo, path) row, shouldn't happen
                v = v.iloc[0]
            values.append(v)
    return max(values) if values else None


if __name__ == "__main__":
    metrics = load_graph_metrics()
    print(f"Loaded centrality metrics for {len(metrics)} (repo, file) pairs")

    # Sanity check: confirm the slash-normalization fix actually works end to end
    from issue_loader import load_issues
    issues = load_issues()
    checked, found = 0, 0
    for issue in issues[:2000]:
        if issue["centrality_of_affected_files"] is None and issue["affected_files"]:
            checked += 1
            result = lookup_centrality(issue["repo_name"], issue["affected_files"], metrics)
            if result is not None:
                found += 1
    print(f"Fallback lookup sanity check (first 2000 issues missing precomputed centrality): "
          f"{found}/{checked} recovered via graph_metrics.csv fallback")