import ast
import csv
from pathlib import Path

import networkx as nx

from repo_config import FINAL_REPOS, CLONE_DIR, OUTPUT_DIR

# Folders/files to skip while walking the repo — irrelevant to real code structure.
IGNORE_DIRS = {"tests", "test", "vendor", "node_modules", ".git", "venv", "env",
                "__pycache__", "build", "dist", "docs", "examples"}


def find_python_files(repo_root: Path):
    """Walks the repo, returns every .py file not inside an ignored folder."""
    for path in repo_root.rglob("*.py"):
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        yield path


def module_name_for(file_path: Path, repo_root: Path) -> str:
    """Converts a file path like pkg/utils/helpers.py -> pkg.utils.helpers"""
    rel = file_path.relative_to(repo_root).with_suffix("")
    return ".".join(rel.parts)


def extract_imports(file_path: Path) -> list[str]:
    """Parses one Python file with `ast` and returns the modules it imports."""
    try:
        tree = ast.parse(file_path.read_text(errors="ignore"))
    except SyntaxError:
        return []  # skip files that don't parse cleanly

    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def build_graph_for_repo(repo_root: Path) -> nx.DiGraph:
    """
    Builds a directed graph: one node per file, one edge A -> B if file A
    imports something from file B (matched by module name, only counting
    imports that resolve to a file inside this same repo).
    """
    graph = nx.DiGraph()
    py_files = list(find_python_files(repo_root))

    # map every internal module name -> its file, so we can resolve imports
    module_to_file = {module_name_for(f, repo_root): f for f in py_files}

    for f in py_files:
        node_id = str(f.relative_to(repo_root))
        graph.add_node(node_id)

    for f in py_files:
        source_id = str(f.relative_to(repo_root))
        for imp in extract_imports(f):
            # match "pkg.utils.helpers" or a prefix of it (e.g. "pkg.utils")
            for module_name, target_file in module_to_file.items():
                if imp == module_name or imp.startswith(module_name + "."):
                    target_id = str(target_file.relative_to(repo_root))
                    if target_id != source_id:
                        graph.add_edge(source_id, target_id)
                    break

    return graph


def compute_metrics(graph: nx.DiGraph) -> dict:
    """Returns {file_path: {betweenness, pagerank, degree}}."""
    if graph.number_of_nodes() == 0:
        return {}

    betweenness = nx.betweenness_centrality(graph)
    pagerank = nx.pagerank(graph) if graph.number_of_edges() > 0 else {n: 0 for n in graph.nodes}
    degree = dict(graph.degree())

    return {
        node: {
            "betweenness": betweenness.get(node, 0.0),
            "pagerank": pagerank.get(node, 0.0),
            "degree": degree.get(node, 0),
        }
        for node in graph.nodes
    }


def export_graph_metrics_csv(all_rows: list[dict], output_path: Path):
    fieldnames = ["file_path", "betweenness", "pagerank", "degree", "repo_name"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {output_path}")


def main():
    all_rows = []

    for repo_name in FINAL_REPOS:
        repo_root = Path(CLONE_DIR) / repo_name.split("/")[-1]
        if not repo_root.exists():
            print(f"Skipping {repo_name} — not cloned yet (run clone_repos.py first).")
            continue

        print(f"Building graph for {repo_name} ...")
        graph = build_graph_for_repo(repo_root)
        metrics = compute_metrics(graph)

        for file_path, m in metrics.items():
            all_rows.append({
                "file_path": file_path,
                "betweenness": round(m["betweenness"], 6),
                "pagerank": round(m["pagerank"], 6),
                "degree": m["degree"],
                "repo_name": repo_name,
            })

        print(f"  {graph.number_of_nodes()} files, {graph.number_of_edges()} import edges")

    output_path = Path(OUTPUT_DIR) / "graph_metrics.csv"
    output_path.parent.mkdir(exist_ok=True)
    export_graph_metrics_csv(all_rows, output_path)


if __name__ == "__main__":
    main()
