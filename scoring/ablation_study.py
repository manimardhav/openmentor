"""
ablation_study.py — compares the full ranking system against feature-group
ablations and baselines, using REAL ground truth (resolved_by_newcomer).
"""

import pandas as pd
from issue_loader import load_issues
from graph_metrics_loader import load_graph_metrics
from features import build_feature_dataframe, FEATURE_COLUMNS
from skill_matching import match_score
from baselines import gfi_label_baseline, single_feature_baseline, random_baseline
from evaluation_metrics import evaluate_ranking
from ground_truth_loader import load_ground_truth, build_real_relevant_set

ABLATION_WEIGHTS = {
    "full":              {"centrality": 0.35, "text_length": 0.15, "issue_age_days": 0.15, "first_contribution_label": -0.20, "referenced_files": 0.25},
    "centrality_only":   {"centrality": 1.0,  "text_length": 0.0,  "issue_age_days": 0.0,  "first_contribution_label": 0.0,   "referenced_files": 0.0},
    "text_only":         {"centrality": 0.0,  "text_length": 0.4,  "issue_age_days": 0.3,  "first_contribution_label": 0.0,   "referenced_files": 0.3},
}

DEMO_CONTRIBUTOR_SKILLS = {"testing/CI", "packaging/build"}


def score_with_weights(features_df: pd.DataFrame, weights: dict) -> pd.Series:
    return features_df.apply(lambda row: sum(weights[f] * row[f] for f in FEATURE_COLUMNS), axis=1)


def rank_by_difficulty_ablation(issues: list, ablation_name: str, metrics_df=None) -> list:
    features_df = build_feature_dataframe(issues, metrics_df)
    features_df["score"] = score_with_weights(features_df, ABLATION_WEIGHTS[ablation_name])
    ranked = features_df.sort_values("score", ascending=True)
    return ranked["issue_id"].tolist()


def rank_by_skill_match_only(issues: list, contributor_skills: set) -> list:
    rows = [(issue["id"], match_score(contributor_skills, issue.get("body", ""))) for issue in issues]
    rows.sort(key=lambda x: -x[1])
    return [issue_id for issue_id, _ in rows]


def run_ablation_study():
    issues = load_issues()
    ground_truth_df = load_ground_truth()
    relevant_ids = build_real_relevant_set(ground_truth_df)
    k_values = [1, 3, 5]

    try:
        metrics = load_graph_metrics()
    except FileNotFoundError:
        metrics = None

    results = {}

    for name in ["full", "centrality_only", "text_only"]:
        ranked_ids = rank_by_difficulty_ablation(issues, name, metrics)
        results[f"difficulty[{name}]"] = evaluate_ranking(ranked_ids, relevant_ids, k_values)

    ranked_ids = rank_by_skill_match_only(issues, DEMO_CONTRIBUTOR_SKILLS)
    results["skill_match_only"] = evaluate_ranking(ranked_ids, relevant_ids, k_values)

    gfi_ranked = gfi_label_baseline(ground_truth_df)["issue_id"].tolist()
    results["baseline_gfi_label"] = evaluate_ranking(gfi_ranked, relevant_ids, k_values)

    naive_ranked = single_feature_baseline(issues)["issue_id"].tolist()
    results["baseline_issue_age"] = evaluate_ranking(naive_ranked, relevant_ids, k_values)

    random_ranked = random_baseline(issues)["issue_id"].tolist()
    results["baseline_random"] = evaluate_ranking(random_ranked, relevant_ids, k_values)

    comparison_df = pd.DataFrame(results).T
    return comparison_df


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    relevant = build_real_relevant_set()
    print(f"Real relevant set (issues actually resolved by a newcomer): {relevant}")
    print(f"({len(relevant)} issues)\n")

    comparison = run_ablation_study()
    print("Ablation + baseline comparison table:")
    print(comparison.to_string())