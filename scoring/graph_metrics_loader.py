"""
graph_metrics_loader.py — replaces build_placeholder_graph.py entirely.

Person A's ingestion pipeline already computed real per-file centrality
metrics (betweenness, pagerank, degree) into data/graph_metrics.csv — we
don't need networkx or a hand-built graph anymore. This file is a lookup
table: file_path -> centrality metrics, used as a fallback for the rare
issue where data/issues.csv's own precomputed `centrality_of_affected_files`
is missing (see features.py's real-data path).
"""

import pandas as pd

GRAPH_METRICS_PATH = "../data/graph_metrics.csv"


def load_graph_metrics(path: str = GRAPH_METRICS_PATH) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by file_path with betweenness/pagerank/degree
    columns. Real columns confirmed from the data: file_path, betweenness,
    pagerank, degree, repo_name.
    """
    df = pd.read_csv(path)
    return df.set_index("file_path")


def lookup_centrality(file_paths: list, metrics_df: pd.DataFrame, metric: str = "betweenness") -> float:
    """
    Given a list of file paths an issue touches, returns the MAX centrality
    among them (same "hardest touched file wins" logic as the old
    features.py, now backed by real precomputed metrics instead of a
    hand-built graph).
    """
    if not file_paths:
        return 0.0
    values = []
    for fp in file_paths:
        if fp in metrics_df.index:
            values.append(metrics_df.loc[fp, metric])
    return max(values) if values else 0.0


if __name__ == "__main__":
    metrics = load_graph_metrics()
    print(f"Loaded centrality metrics for {len(metrics)} files")
    print(metrics.head())