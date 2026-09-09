"""
full_pipeline.py — the complete recommendation pipeline, rewired for real
matplotlib data. No more networkx/placeholder graph.
"""

import pandas as pd
from issue_loader import load_issues
from graph_metrics_loader import load_graph_metrics
from features import build_feature_dataframe
from difficulty_model import DifficultyModel
from ranking import combined_rank_score, confidence_tag
from train_difficulty_model import build_training_data


def get_difficulty_predictions(issues: list) -> dict:
    training_df = build_training_data(issues)
    model = DifficultyModel().fit(training_df)

    try:
        metrics = load_graph_metrics()
    except FileNotFoundError:
        metrics = None

    features_df = build_feature_dataframe(issues, metrics)
    features_df["predicted_difficulty"] = model.predict_proba(features_df)

    return dict(zip(features_df["issue_id"], features_df["predicted_difficulty"]))


def run_full_pipeline(
    contributor_skills: set = None,
    contributor_level: float = None,
    alpha: float = 0.5,
    open_only: bool = True,
) -> pd.DataFrame:
    if contributor_skills is None or contributor_level is None:
        print("No contributor profile provided — using a demo profile.")
        print("(Run contributor_profile.py directly for the real interactive form.)")
        contributor_skills = contributor_skills or {"testing/CI", "packaging/build"}
        contributor_level = contributor_level if contributor_level is not None else 0.25

    issues = load_issues()
    difficulty_by_id = get_difficulty_predictions(issues)

    if open_only:
        issues = [i for i in issues if i.get("state") == "open"]

    rows = []
    for issue in issues:
        predicted_difficulty = difficulty_by_id[issue["id"]]
        scores = combined_rank_score(
            predicted_difficulty=predicted_difficulty,
            contributor_level=contributor_level,
            contributor_skills=contributor_skills,
            issue_text=issue.get("body", ""),
            alpha=alpha,
        )
        rows.append({
            "issue_id": issue["id"],
            "title": issue.get("title", "")[:55],
            "predicted_difficulty": round(predicted_difficulty, 3),
            **{k: round(v, 3) for k, v in scores.items()},
            "confidence_tag": confidence_tag(scores["combined_score"], scores["skill_match"]),
        })

    result = pd.DataFrame(rows).sort_values("combined_score", ascending=False).reset_index(drop=True)
    return result


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 55)

    result = run_full_pipeline()
    print("\nFinal ranked recommendations (open issues only):")
    print(result.to_string(index=False))