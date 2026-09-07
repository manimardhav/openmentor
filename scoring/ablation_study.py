"""
ablation_study.py — Week 6: compare the full ranking system against
feature-group ablations and the baselines from baselines.py, using
precision/recall/hit-rate.

MOCK DATA NOTE: the tiny 10-issue mock set and guessed relevance set mean
the numbers below are NOT a real result. Re-run once real issues + real
tester ground truth exist — nothing about the code changes.
"""

import pandas as pd
from sanity_check import REAL_ISSUES
from build_placeholder_graph import build_placeholder_graph
from features import build_feature_dataframe
from difficulty_model import FEATURE_COLUMNS
from skill_matching import match_score
from baselines import gfi_label_baseline, single_feature_baseline, random_baseline
from evaluation_metrics import evaluate_ranking
from mock_relevance import build_mock_relevant_set, DEMO_CONTRIBUTOR_SKILLS

ABLATION_WEIGHTS = {
    "full":              {"centrality": 0.35, "text_length": 0.15, "comment_count": 0.15, "help_wanted_label": -0.20, "referenced_files": 0.25},
    "centrality_only":   {"centrality": 1.0,  "text_length": 0.0,  "comment_count": 0.0,  "help_wanted_label": 0.0,   "referenced_files": 0.0},
    "text_only":         {"centrality": 0.0,  "text_length": 0.4,  "comment_count": 0.3,  "help_wanted_label": 0.0,   "referenced_files": 0.3},
}


def score_with_weights(features_df: pd.DataFrame, weights: dict) -> pd.Series:
    return features_df.apply(lambda row: sum(weights[f] * row[f] for f in FEATURE_COLUMNS), axis=1)


def rank_by_difficulty_ablation(issues: list, ablation_name: str) -> list:
    """Ranks EASIEST first (ascending score) — what a recommender should surface to a beginner."""
    graph = build_placeholder_graph()
    features_df = build_feature_dataframe(issues, graph)
    features_df["score"] = score_with_weights(features_df, ABLATION_WEIGHTS[ablation_name])
    ranked = features_df.sort_values("score", ascending=True)
    return ranked["issue_id"].tolist()


def rank_by_skill_match_only(issues: list, contributor_skills: set) -> list:
    rows = [(issue["id"], match_score(contributor_skills, issue.get("body", ""))) for issue in issues]
    rows.sort(key=lambda x: -x[1])
    return [issue_id for issue_id, _ in rows]


def run_ablation_study():
    issues = REAL_ISSUES
    relevant_ids = build_mock_relevant_set()
    k_values = [1, 3, 5]

    results = {}

    for name in ["full", "centrality_only", "text_only"]:
        ranked_ids = rank_by_difficulty_ablation(issues, name)
        results[f"difficulty[{name}]"] = evaluate_ranking(ranked_ids, relevant_ids, k_values)

    ranked_ids = rank_by_skill_match_only(issues, DEMO_CONTRIBUTOR_SKILLS)
    results["skill_match_only"] = evaluate_ranking(ranked_ids, relevant_ids, k_values)

    gfi_ranked = gfi_label_baseline(issues)["issue_id"].tolist()
    results["baseline_gfi_label"] = evaluate_ranking(gfi_ranked, relevant_ids, k_values)

    naive_ranked = single_feature_baseline(issues, feature="comments")["issue_id"].tolist()
    results["baseline_comment_count"] = evaluate_ranking(naive_ranked, relevant_ids, k_values)

    random_ranked = random_baseline(issues)["issue_id"].tolist()
    results["baseline_random"] = evaluate_ranking(random_ranked, relevant_ids, k_values)

    comparison_df = pd.DataFrame(results).T
    return comparison_df


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    print(f"Mock relevant set: {build_mock_relevant_set()}")
    print("(Reminder: this is a GUESSED ground truth for testing code mechanics only.)\n")

    comparison = run_ablation_study()
    print("Ablation + baseline comparison table:")
    print(comparison.to_string())