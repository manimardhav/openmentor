"""
repo_config.py — Week 1: candidate + final repo list.

Fill CANDIDATE_REPOS with 8-10 repos you're considering (owner/name format,
same as the URL: github.com/<owner>/<name>).

Then move your final 3-5 choices into FINAL_REPOS once you've checked they
meet the criteria from the plan:
  - 100-500 open issues
  - one consistent primary language
  - commit activity within the last 6 months
"""

CANDIDATE_REPOS = [
    "matplotlib/matplotlib",
    "pandas-dev/pandas",
    "huggingface/datasets",
    "networkx/networkx",
    "vuejs/core",              # Frontend — TypeScript
    "helm/helm",               # DevOps — Go
    "sqlalchemy/sqlalchemy",   # Backend — Python, ORM/database
    "huggingface/accelerate",  # AI/ML — Python, training infra
    "expressjs/express",       # Backend — JavaScript/Node.js
    # scikit-learn/scikit-learn dropped — no active "good first issue" labels
    # scikit-image/scikit-image also checked — same problem, also dropped
    # pallets/flask and tiangolo/fastapi considered but REJECTED — both
    # have functionally empty GitHub issue trackers (moved to Discussions)
]

# NOTE: vuejs/core, helm/helm, and expressjs/express are NOT Python — they
# require build_dependency_graph_multilang.py (not the classic .py-only
# build_dependency_graph.py) to produce any graph data at all.
FINAL_REPOS = [
    "matplotlib/matplotlib",
    "pandas-dev/pandas",
    "huggingface/datasets",
    "networkx/networkx",
    "vuejs/core",
    "helm/helm",
    "sqlalchemy/sqlalchemy",
    "huggingface/accelerate",
    "expressjs/express",
]


# Paths / storage roots — do not need to change these.
CLONE_DIR = "clones"          # where repos get cloned locally
OUTPUT_DIR = "../data"        # where graph_metrics.csv / issues.csv get written