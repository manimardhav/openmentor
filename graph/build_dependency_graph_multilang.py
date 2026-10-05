"""
build_dependency_graph_multilang.py — language-agnostic version of
build_dependency_graph.py. Works on any language listed in
language_config.py, instead of only Python.

WHY THIS EXISTS: the original build_dependency_graph.py used Python's
built-in `ast` module, which can only ever understand Python syntax. This
version uses tree-sitter instead — a parser that supports hundreds of
languages — so adding a new language is a config change, not a rewrite.

SETUP (one-time):
    pip install tree-sitter tree-sitter-language-pack

Output format is UNCHANGED from the original script — same
data/graph_metrics.csv columns (file_path, betweenness, pagerank, degree,
repo_name) — so nothing downstream (Member 1's scoring code) needs to
change at all. This script is a drop-in upgrade, not a new pipeline.
"""

import csv
from pathlib import Path

import networkx as nx
from tree_sitter import Query, QueryCursor
from tree_sitter_language_pack import get_parser

from repo_config import FINAL_REPOS, CLONE_DIR, OUTPUT_DIR
from language_config import LANGUAGES, IGNORE_DIRS

# Cache parsers and compiled queries per language so we don't rebuild them
# for every single file (parsing setup has real overhead at this scale).
_parser_cache = {}
_query_cache = {}


def get_cached_parser(ts_name: str):
    if ts_name not in _parser_cache:
        _parser_cache[ts_name] = get_parser(ts_name)
    return _parser_cache[ts_name]


def get_cached_query(ts_name: str, query_str: str, parser):
    key = ts_name
    if key not in _query_cache:
        _query_cache[key] = Query(parser.language, query_str)
    return _query_cache[key]


def find_code_files(repo_root: Path):
    """Walks the repo, returns every file whose extension is in LANGUAGES."""
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        if path.suffix in LANGUAGES:
            yield path


def extract_imports(file_path: Path) -> list[str]:
    """
    Returns the raw import target strings found in one file (e.g. "os",
    "./utils", "<iostream>") — NOT yet resolved to an actual file. Works
    for any language configured in language_config.py.
    """
    config = LANGUAGES.get(file_path.suffix)
    if config is None:
        return []

    try:
        source = file_path.read_bytes()
        parser = get_cached_parser(config["ts_name"])
        tree = parser.parse(source)
        query = get_cached_query(config["ts_name"], config["import_query"], parser)
        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)
    except Exception:
        return []  # unparseable file (syntax error, binary, etc.) — skip it

    results = []
    for name, nodes in captures.items():
        if name != "module":
            continue
        for node in nodes:
            text = node.text.decode(errors="ignore")
            # strip quotes commonly left by string literal nodes (Go, PHP,
            # C/C++ <angle> or "quoted" includes)
            text = text.strip("'\"<>")
            results.append(text)
    return results


def module_parts_for(file_path: Path, repo_root: Path) -> tuple:
    parts = file_path.relative_to(repo_root).with_suffix("").parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return parts


def build_dotted_suffix_index(code_files: list[Path], repo_root: Path) -> dict:
    """Same right-aligned suffix matching the Python-only version used,
    generalized — works equally for Python's "pkg.mod" and Java's
    "com.example.Foo" style dotted imports."""
    index = {}
    for f in code_files:
        parts = module_parts_for(f, repo_root)
        for start in range(len(parts)):
            index.setdefault(parts[start:], f)
    return index


def resolve_dotted(import_text: str, suffix_index: dict):
    imp_parts = tuple(p for p in import_text.split(".") if p)
    for length in range(len(imp_parts), 0, -1):
        match = suffix_index.get(imp_parts[:length])
        if match:
            return match
    return None


def resolve_path(import_text: str, source_file: Path, repo_root: Path,
                  path_extensions: list[str], all_files_set: set):
    """
    Resolves a path-style import (relative paths, local headers) against
    the filesystem, relative to the IMPORTING file's own folder — the same
    way a real compiler/bundler would.
    """
    if not import_text.startswith((".", "/")):
        # Likely a third-party package (npm package, system header like
        # <vector>, a Go module path like "github.com/x/y") rather than a
        # same-repo file — nothing to resolve to, and that's correct/expected.
        return None

    base = (source_file.parent / import_text).resolve()
    candidates = [base]
    for ext in path_extensions:
        if ext.startswith("/"):
            candidates.append(Path(str(base) + ext))
        else:
            candidates.append(base.with_suffix(ext))

    for candidate in candidates:
        if candidate in all_files_set:
            return candidate
    return None


def build_graph_for_repo(repo_root: Path) -> nx.DiGraph:
    graph = nx.DiGraph()
    code_files = list(find_code_files(repo_root))
    all_files_set = set(code_files)

    dotted_index = build_dotted_suffix_index(code_files, repo_root)

    for f in code_files:
        graph.add_node(str(f.relative_to(repo_root)))

    language_coverage = {}  # for the printed summary at the end

    for f in code_files:
        config = LANGUAGES[f.suffix]
        language_coverage[config["ts_name"]] = language_coverage.get(config["ts_name"], 0) + 1
        source_id = str(f.relative_to(repo_root))

        for imp in extract_imports(f):
            if config["style"] == "dotted":
                target = resolve_dotted(imp, dotted_index)
            else:
                target = resolve_path(imp, f, repo_root, config.get("path_extensions", []), all_files_set)

            if target is not None:
                target_id = str(target.relative_to(repo_root))
                if target_id != source_id:
                    graph.add_edge(source_id, target_id)

    return graph, language_coverage


def compute_metrics(graph: nx.DiGraph) -> dict:
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


def export_graph_edges_csv(all_edges: list[dict], output_path: Path):
    """
    Raw, un-aggregated edge list — for anyone (e.g. a visual 'which files
    connect to which' feature) who needs the actual connections, not just
    the collapsed centrality numbers. One row per import relationship.
    """
    fieldnames = ["source_file", "target_file", "repo_name"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_edges)
    print(f"Wrote {len(all_edges)} rows to {output_path}")


def main():
    all_rows = []
    all_edges = []

    for repo_name in FINAL_REPOS:
        repo_root = Path(CLONE_DIR) / repo_name.split("/")[-1]
        if not repo_root.exists():
            print(f"Skipping {repo_name} — not cloned yet (run clone_repos.py first).")
            continue

        print(f"Building graph for {repo_name} (multi-language) ...")
        graph, language_coverage = build_graph_for_repo(repo_root)
        metrics = compute_metrics(graph)

        for file_path, m in metrics.items():
            all_rows.append({
                "file_path": file_path,
                "betweenness": round(m["betweenness"], 6),
                "pagerank": round(m["pagerank"], 6),
                "degree": m["degree"],
                "repo_name": repo_name,
            })

        for source, target in graph.edges():
            all_edges.append({
                "source_file": source,
                "target_file": target,
                "repo_name": repo_name,
            })

        langs_str = ", ".join(f"{lang}: {count} files" for lang, count in sorted(language_coverage.items()))
        print(f"  {graph.number_of_nodes()} files, {graph.number_of_edges()} edges — languages seen: {langs_str or 'none'}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    export_graph_metrics_csv(all_rows, output_dir / "graph_metrics.csv")
    export_graph_edges_csv(all_edges, output_dir / "graph_edges.csv")


if __name__ == "__main__":
    main()