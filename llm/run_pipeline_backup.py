"""
run_pipeline.py — wires Person A's scoring scaffold (features.py,
difficulty_model.py, skill_matching.py, ranking.py) onto the REAL data
(data/issues.csv, data/graph_metrics.csv) to produce data/ranked_issues.json,
matching the schema in shared/schemas.md.

This did not exist before review: scoring/ was toy-data scaffolding only
(see scoring/README.md, "Week 1 status" / smoke tests at bottom of each
file). This script is the missing glue, written to get an executable
end-to-end demo for tomorrow's review — NOT a claim that scoring is
tuned or validated. See the "Known limitations" printed at the end.

Design choices made to ship this tonight (flag these in review):
- centrality: data/issues.csv already has a computed
  `centrality_of_affected_files` column (from graph_metrics.csv joined
  upstream), so we use that directly instead of rebuilding a
  networkx.Graph and recomputing betweenness centrality ourselves.
- comment_count: issues.csv has no comment-count column, so this
  feature is 0 for every issue. Flag this as a known gap.
- help_wanted / good-first-issue signal: the real labels in this
  dataset don't use GitHub's standard "good first issue" / "help
  wanted" text — they use "first-contribution" and similar. Broadened
  the match accordingly.
- Skill taxonomy: kept scoring/skill_matching.py's existing taxonomy
  as-is. It was written for httpie/cli, but the actual pilot data is
  matplotlib/pandas/datasets/networkx — tag hit-rate will be weak.
  This is the taxonomy mismatch to raise in review, not something
  papered over here.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scoring"))

from features import (  # noqa: E402
    text_length_feature,
    comment_count_feature,
    referenced_files_feature,
)
from difficulty_model import weighted_difficulty_score  # noqa: E402
from skill_matching import tag_issue_skills, match_score, SKILL_TAXONOMY  # noqa: E402
from ranking import difficulty_fit, confidence_tag  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ISSUES_CSV = PROJECT_ROOT / "data" / "issues.csv"
OUT_PATH = PROJECT_ROOT / "data" / "ranked_issues.json"

# A stand-in contributor profile for tonight's demo run. The real profile
# will come from the Streamlit sidebar at request time (see interface/app.py) —
# this file just needs *a* ranked_issues.json to exist so the interface has
# real data to load instead of the mock file.
DEMO_CONTRIBUTOR_LEVEL = 0.3
DEMO_CONTRIBUTOR_SKILLS = {"testing", "docs"}

MAX_ISSUES = 200  # cap for a fast, reviewable demo run tonight


def has_first_contribution_signal(labels) -> float:
    labels_lower = "" if pd.isna(labels) else str(labels).lower()
    return 1.0 if any(
        kw in labels_lower for kw in ["first-contribution", "good first issue", "help wanted", "community support"]
    ) else 0.0


def parse_centrality(raw) -> float:
    """centrality_of_affected_files is either 'NONE_FOUND', a single float-like
    string, or semicolon-joined values for multiple files. Take the max."""
    if pd.isna(raw) or raw == "NONE_FOUND":
        return 0.0
    try:
        parts = [float(p) for p in str(raw).split(";") if p and p != "NONE_FOUND"]
        return max(parts) if parts else 0.0
    except ValueError:
        return 0.0


def main():
    df = pd.read_csv(ISSUES_CSV)
    df = df[df["state"] == "open"].dropna(subset=["title", "body"]).head(MAX_ISSUES)

    results = []
    for _, row in df.iterrows():
        centrality = parse_centrality(row.get("centrality_of_affected_files"))
        feature_row = pd.Series({
            "centrality": centrality,
            "text_length": text_length_feature(row["body"]),
            "comment_count": comment_count_feature(0),  # not available in this CSV
            "help_wanted_label": has_first_contribution_signal(row.get("labels", "")),
            "referenced_files": referenced_files_feature(row["body"]),
        })
        difficulty = max(0.0, min(1.0, weighted_difficulty_score(feature_row)))

        fit = difficulty_fit(difficulty, DEMO_CONTRIBUTOR_LEVEL)
        skill_score = match_score(DEMO_CONTRIBUTOR_SKILLS, row["body"], metric="cosine")
        combined = 0.5 * fit + 0.5 * skill_score
        matched_skills = sorted(tag_issue_skills(row["body"]))

        results.append({
            "issue_id": str(row["issue_id"]),
            "repo_name": row["repo_name"],
            "title": row["title"],
            "body": (row["body"] or "")[:600],  # trim for interface display
            "difficulty_score": round(difficulty, 3),
            "skill_match_score": round(skill_score, 3),
            "final_rank_score": round(combined, 3),
            "matched_skills": matched_skills,
            "confidence_tag": confidence_tag(combined, skill_score),
            "top_features": {
                "centrality": round(centrality, 3),
                "help_wanted_label": bool(feature_row["help_wanted_label"]),
                "referenced_files": round(feature_row["referenced_files"], 3),
            },
        })

    results.sort(key=lambda r: r["final_rank_score"], reverse=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} ranked issues to {OUT_PATH}")
    print(f"\nConfidence tag distribution:")
    print(pd.Series([r["confidence_tag"] for r in results]).value_counts())
    print(f"\nIssues with >=1 matched skill: {sum(1 for r in results if r['matched_skills'])}/{len(results)}")
    print(
        "\nKnown limitations (state these in review):\n"
        "- Skill taxonomy (skill_matching.SKILL_TAXONOMY) was written for httpie/cli; "
        "actual pilot data is matplotlib/pandas/datasets/networkx, so skill-tag hit rate is low.\n"
        "- comment_count feature is 0 for all issues (not present in this CSV extract).\n"
        "- difficulty_score uses the hand-tuned weighted formula (Week 2), not the trained "
        "logistic model (Week 3) — ground_truth.csv exists but DifficultyModel hasn't been fit on it yet."
    )


if __name__ == "__main__":
    main()