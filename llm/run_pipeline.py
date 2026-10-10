"""
run_pipeline.py — Member 2's glue script that produces data/ranked_issues.json
for the Streamlit interface (interface/app.py) and the LLM roadmap step
(llm/generate_roadmap.py), matching the schema in shared/schemas.md.

v4 changes (9-repo update, made together with Member 1's scoring/ changes):
- scoring/issue_loader.py now returns issue["id"] = unique_id
  ("<repo_name>#<issue_id>") and issue["repo_name"] directly. The old workaround here
  (read repo_name from the CSV into a dict keyed by raw issue_id) is gone: it was
  wrong for ~21% of issues and would now return "unknown" for every issue.
- The os.chdir(scoring/) workaround is gone: the loaders anchor their paths to their
  own file location, so this runs from any directory.
- top_features is now truthful:
    * "centrality" is None (not an imputed median) when no resolving PR was traced,
      which is true for 100% of OPEN issues. interface/explanations.py skips None.
    * "issue_age_days" is real days and "referenced_files" is a real count (they used
      to be normalized 0-1 values, which explanations.py printed as days / counts).
- confidence_tag is now two-valued ("high-confidence" | "exploratory"), per
  shared/schemas.md.

STILL TRUE (state in the paper's limitations section):
- centrality / referenced_files / centrality_known are post-resolution information,
  so they are unavailable for the open issues this script ranks.
- Skill-taxonomy hit rate is much lower on helm, sqlalchemy, vuejs and express than on
  the original repos (run scoring/validate_taxonomy.py).
"""

import json
import sys
from pathlib import Path

import pandas as pd

SCORING_DIR = Path(__file__).resolve().parent.parent / "scoring"
sys.path.insert(0, str(SCORING_DIR))

from issue_loader import load_issues                        # noqa: E402
from graph_metrics_loader import load_graph_metrics        # noqa: E402
from features import build_feature_dataframe                # noqa: E402
from difficulty_model import DifficultyModel                # noqa: E402
from skill_matching import tag_issue_skills, match_score     # noqa: E402
from ranking import difficulty_fit, confidence_tag          # noqa: E402
from train_difficulty_model import build_training_data      # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "data" / "ranked_issues.json"

# Stand-in contributor profile for a non-interactive demo run. The real profile comes
# from the Streamlit sidebar at request time. These skill strings must come from
# shared/skill_taxonomy.json (scoring/skill_matching.py loads its skill names from it).
DEMO_CONTRIBUTOR_LEVEL = 0.3
DEMO_CONTRIBUTOR_SKILLS = {"testing/CI", "documentation"}

# Up to 50 open issues per repo, so the demo covers every repo instead of being
# dominated by whichever repo comes first in issues.csv. Repos with fewer than 50 open
# issues contribute all of them.
PER_REPO_CAP = 50


def get_difficulty_predictions(all_issues: list) -> pd.DataFrame:
    """Trains DifficultyModel on every CLOSED issue (the only ones with a real
    days_to_close label), then scores every issue with it. The returned DataFrame is in
    the SAME ROW ORDER as all_issues (build_feature_dataframe emits one row per input
    issue, in order)."""
    training_df = build_training_data(all_issues)
    model = DifficultyModel().fit(training_df)

    try:
        metrics = load_graph_metrics()
    except FileNotFoundError:
        metrics = None

    features_df = build_feature_dataframe(all_issues, metrics)
    features_df["predicted_difficulty"] = model.predict_proba(features_df)
    return features_df


def main():
    all_issues = load_issues()
    scored_df = get_difficulty_predictions(all_issues)  # same order/length as all_issues

    # Group OPEN issues by repo, paired with their feature row by position.
    open_by_repo: dict = {}
    for idx, issue in enumerate(all_issues):
        if issue.get("state") != "open":
            continue
        open_by_repo.setdefault(issue["repo_name"], []).append((issue, scored_df.iloc[idx]))

    selected = []
    for repo, items in open_by_repo.items():
        selected.extend(items[:PER_REPO_CAP])

    results = []
    for issue, row in selected:
        difficulty = max(0.0, min(1.0, float(row["predicted_difficulty"])))
        fit = difficulty_fit(difficulty, DEMO_CONTRIBUTOR_LEVEL)
        skill_score = match_score(DEMO_CONTRIBUTOR_SKILLS, issue["body"], metric="cosine")
        combined = 0.5 * fit + 0.5 * skill_score
        matched_skills = sorted(tag_issue_skills(issue["body"]))

        results.append({
            # issue["id"] is already "<repo_name>#<issue_id>" (unique_id), the
            # collision-free key required by tests/test_ranked_issues_schema.py.
            "issue_id": issue["id"],
            "repo_name": issue["repo_name"],
            "title": issue["title"],
            "body": (issue["body"] or "")[:600],  # trimmed for interface display
            "difficulty_score": round(difficulty, 3),
            "skill_match_score": round(skill_score, 3),
            "final_rank_score": round(combined, 3),
            "matched_skills": matched_skills,
            "confidence_tag": confidence_tag(combined, skill_score),
            "top_features": {
                # None (not an imputed median) when unknown — see module docstring.
                "centrality": None if pd.isna(row["centrality_raw"]) else round(float(row["centrality_raw"]), 4),
                "text_length": round(float(row["text_length"]), 3),
                "issue_age_days": int(row["issue_age_days_raw"]),
                "first_contribution_label": bool(row["first_contribution_label"]),
                "referenced_files": int(row["referenced_files_count"]),
            },
        })

    results.sort(key=lambda r: r["final_rank_score"], reverse=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} ranked issues to {OUT_PATH}")
    print("\nRepo distribution:")
    print(pd.Series([r["repo_name"] for r in results]).value_counts())
    print("\nConfidence tag distribution:")
    print(pd.Series([r["confidence_tag"] for r in results]).value_counts())
    print(f"\nIssues with >=1 matched skill: {sum(1 for r in results if r['matched_skills'])}/{len(results)}")
    print(
        "\nKnown limitations (state these in the paper's methodology/limitations section):\n"
        "- issue_id is 'repo_name#number' (unique_id): raw GitHub issue numbers collide across repos.\n"
        "- centrality is unknown for every open issue (it comes from the resolving PR), so "
        "top_features['centrality'] is None here; difficulty_score for open issues is driven by "
        "text length, issue age and the beginner-label flag.\n"
        "- difficulty_score comes from DifficultyModel (logistic regression on median-split "
        "days_to_close), not the hand-tuned weighted formula.\n"
        "- DEMO_CONTRIBUTOR_SKILLS is a placeholder; the real profile comes from the Streamlit sidebar."
    )


if __name__ == "__main__":
    main()