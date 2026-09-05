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
    "scikit-learn/scikit-learn",
    "matplotlib/matplotlib",
    "pandas-dev/pandas",
    "huggingface/datasets",
]

FINAL_REPOS = [
    "matplotlib/matplotlib",   # test this ONE first
]

# Paths / storage roots — do not need to change these.
CLONE_DIR = "clones"          # where repos get cloned locally
OUTPUT_DIR = "../data"        # where graph_metrics.csv / issues.csv get written
