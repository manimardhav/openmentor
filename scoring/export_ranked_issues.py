"""
export_ranked_issues.py — writes data/ranked_issues.json in the exact
shape interface/app.py expects: issue_id, repo_name, title, body,
confidence_tag, final_rank_score, difficulty_score, skill_match_score,
matched_skills.

NOTE on skill_match_score / final_rank_score: app.py's live_rescore()
recomputes these LIVE against whatever skills the viewer actually selects
in the sidebar — the values written here are only a neutral placeholder
(computed with a generic contributor profile) used before any skill is
selected, or in the "Browse by skill" tab. difficulty_score is NOT
recomputed live by app.py, so it must be the real trained-model output.

Writes to the same path as llm/config.py's REAL_RANKED_ISSUES_PATH
(data/ranked_issues.json) without importing that module directly, to keep
scoring/ independent of llm/.
"""

import json
from pathlib import Path

from issue_loader import load_issues
from train_difficulty_model import build_training_data
from difficulty_model import DifficultyModel
from features import build_feature_dataframe
from graph_metrics_loader import load_graph_metrics
from skill_matching import tag_issue_skills
from ranking import combined_rank_score, confidence_tag

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "ranked_issues.json"

# Neutral placeholder profile — real personalization happens live in app.py.
DEFAULT_CONTRIBUTOR_SKILLS = set()
DEFAULT_CONTRIBUTOR_LEVEL = 0.3


def build_ranked_issues() -> list:
    issues = load_issues()

    training_df = build_training_data(issues)
    model = DifficultyModel().fit(training_df)

    try:
        metrics = load_graph_metrics()
    except FileNotFoundError:
        metrics = None

    features_df = build_feature_dataframe(issues, metrics)
    features_df["predicted_difficulty"] = model.predict_proba(features_df)
    difficulty_by_id = dict(zip(features_df["issue_id"], features_df["predicted_difficulty"]))

    records = []
    for issue in issues:
        predicted_difficulty = difficulty_by_id[issue["id"]]
        matched_skills = sorted(tag_issue_skills(issue.get("body", "")))

        scores = combined_rank_score(
            predicted_difficulty=predicted_difficulty,
            contributor_level=DEFAULT_CONTRIBUTOR_LEVEL,
            contributor_skills=DEFAULT_CONTRIBUTOR_SKILLS,
            issue_text=issue.get("body", ""),
        )

        records.append({
            "issue_id": issue["id"],
            "repo_name": issue.get("repo_name", ""),
            "title": issue.get("title", ""),
            "body": issue.get("body", ""),
            "confidence_tag": confidence_tag(scores["combined_score"], scores["skill_match"]),
            "difficulty_score": round(predicted_difficulty, 3),
            "skill_match_score": round(scores["skill_match"], 3),
            "final_rank_score": round(scores["combined_score"], 3),
            "matched_skills": matched_skills,
        })

    return records


def save_ranked_issues_json(path: Path = OUTPUT_PATH) -> None:
    records = build_ranked_issues()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {len(records)} ranked issues to {path}")


if __name__ == "__main__":
    save_ranked_issues_json()