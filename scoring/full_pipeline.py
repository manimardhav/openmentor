"""
full_pipeline.py — the complete recommendation pipeline: loads issues and
the dependency graph, computes difficulty + skill match, and produces a
final ranked list with confidence tags for a given contributor.

Currently runs entirely on mock data (sanity_check.py's real issue titles +
build_placeholder_graph.py's hand-curated graph). Swap to real data later
by flipping use_real_data=True in issue_loader.py and
build_placeholder_graph.py — nothing in this file needs to change.
"""

import pandas as pd
from issue_loader import get_issues
from build_placeholder_graph import get_dependency_graph
from features import build_feature_dataframe
from difficulty_model import DifficultyModel, FEATURE_COLUMNS
from ranking import combined_rank_score, confidence_tag
from contributor_profile import prompt_contributor_skills, prompt_contributor_level
from train_difficulty_model import INTUITION_TO_LABEL, build_training_data


def get_difficulty_predictions(issues: list, use_real_graph: bool = False) -> dict:
    """
    Trains the difficulty model on mock ground truth and returns
    {issue_id: predicted_difficulty_probability} for the given issues.

    NOTE: this retrains on the mock 10-issue set every call, which is fine
    for development but should be replaced with a single frozen trained
    model once real ground truth exists.
    """
    training_df = build_training_data()  # currently always mock
    model = DifficultyModel().fit(training_df)

    graph = get_dependency_graph(use_real_data=use_real_graph)
    features_df = build_feature_dataframe(issues, graph)
    features_df["predicted_difficulty"] = model.predict_proba(features_df)

    return dict(zip(features_df["issue_id"], features_df["predicted_difficulty"]))


def run_full_pipeline(
    contributor_skills: set = None,
    contributor_level: float = None,
    alpha: float = 0.5,
    use_real_data: bool = False,
) -> pd.DataFrame:
    """
    The main entry point. Returns a DataFrame ranked by combined_score with
    a confidence_tag column, ready to hand off to whoever builds the
    roadmap-generation step downstream.
    """
    if contributor_skills is None or contributor_level is None:
        print("No contributor profile provided — using a demo profile.")
        print("(Run contributor_profile.py directly for the real interactive form.)")
        contributor_skills = contributor_skills or {"HTTP/networking", "testing"}
        contributor_level = contributor_level if contributor_level is not None else 0.25

    issues = get_issues(use_real_data=use_real_data)
    difficulty_by_id = get_difficulty_predictions(issues, use_real_graph=use_real_data)

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
            "title": issue.get("body", "")[:55],
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
    print("\nFinal ranked recommendations:")
    print(result.to_string(index=False))