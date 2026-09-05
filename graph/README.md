# OpenMentor — Repository & Dependency Graph (Member 3)

Starter code for Weeks 1-3 of your role. Run each file standalone with
`python3 <file>.py`.

## Files

| File | Week | What it does |
|---|---|---|
| `repo_config.py` | 1 | List your candidate and final repos here. |
| `clone_repos.py` | 1 | Tests your GitHub token, then clones the final repos locally. |
| `build_dependency_graph.py` | 2-3 | Parses each repo's Python files, builds an import-dependency graph with `networkx`, computes centrality metrics, and writes `../data/graph_metrics.csv`. |

## Setup

From inside this `graph/` folder:

```bash
pip install -r ../requirements.txt
```

Add your GitHub token to the project's `.env` file (same file Member 2 put
`GROQ_API_KEY` in):

```
GITHUB_TOKEN=your_token_here
```

## Order to run things

1. Fill in `CANDIDATE_REPOS` in `repo_config.py` (8-10 repos), narrow down
   to 3-5 in `FINAL_REPOS`.
2. `python clone_repos.py`
3. `python build_dependency_graph.py`
4. Check `../data/graph_metrics.csv` — this is the exact file Member 1 needs
   for scoring (see `../shared/schemas.md`).

## Not yet built (still to do)

- `issues.csv` (pulling issue data via GitHub API, including
  `resolver_is_first_time` per the plan's Week 4 addition) — Week 4.
- Language support beyond Python (`tree-sitter`) if your final repos aren't
  pure Python.
