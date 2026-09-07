"""
build_placeholder_graph.py — Week 2 fallback: build a small REAL dependency
graph from httpie/cli's actual imports, so you're not blocked waiting on
Person A's full ingestion pipeline.

*** UPDATE (once Person A's data lands): swap load_real_graph() in as the
    active function — see the bottom of this file. Keep
    build_placeholder_graph() around as a fallback/debugging tool. ***
"""

import json
import networkx as nx

REAL_HTTPIE_EDGES = [
    ("httpie/__main__.py", "httpie/core.py"),
    ("httpie/core.py", "httpie/client.py"),
    ("httpie/core.py", "httpie/cli/definition.py"),
    ("httpie/core.py", "httpie/output/writer.py"),
    ("httpie/client.py", "httpie/adapters.py"),
    ("httpie/client.py", "httpie/sessions.py"),
    ("httpie/cli/definition.py", "httpie/cli/argparser.py"),
    ("httpie/cli/argparser.py", "httpie/cli/argtypes.py"),
    ("httpie/output/writer.py", "httpie/output/formatters/json.py"),
    ("httpie/output/writer.py", "httpie/output/formatters/colors.py"),
    ("httpie/sessions.py", "httpie/config.py"),
    ("httpie/plugins/manager.py", "httpie/plugins/base.py"),
    ("httpie/core.py", "httpie/plugins/manager.py"),
]


def build_placeholder_graph() -> nx.Graph:
    """Fallback graph — use only until Person A's real export is available."""
    G = nx.Graph()
    G.add_edges_from(REAL_HTTPIE_EDGES)
    return G


def load_real_graph(path: str = "../data/dependency_graph.json") -> nx.Graph:
    """
    Loads Person A's real dependency graph export.
    ADJUST to match their actual file format. Assumes JSON shaped like:
        {"edges": [["file_a.py", "file_b.py"], ...]}
    If GraphML instead, use: return nx.read_graphml(path)
    """
    with open(path) as f:
        data = json.load(f)

    G = nx.Graph()
    edges = data.get("edges", data if isinstance(data, list) else [])
    G.add_edges_from([tuple(edge) for edge in edges])
    return G


def get_dependency_graph(use_real_data: bool = False) -> nx.Graph:
    """Single entry point the rest of your pipeline should call."""
    if use_real_data:
        return load_real_graph()
    return build_placeholder_graph()


if __name__ == "__main__":
    USE_REAL_DATA = False

    G = get_dependency_graph(use_real_data=USE_REAL_DATA)
    print(f"Using {'REAL' if USE_REAL_DATA else 'PLACEHOLDER'} graph")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    from features import compute_centrality_scores
    scores = compute_centrality_scores(G)
    print("\nCentrality scores (higher = more structurally central):")
    for node, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {node:<45}{score:.3f}")