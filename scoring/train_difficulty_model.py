"""
train_difficulty_model.py — trains the difficulty model on REAL ground
truth. Ground truth is days_to_close (real resolution time), median-split
into easy/hard buckets. Only CLOSED issues have this value.
"""

import pandas as pd
from features import build_feature_dataframe
from difficulty_model import DifficultyModel, add_weighted_scores, FEATURE_COLUMNS
from graph_metrics_loader import load_graph_metrics
from issue_loader import load_issues


def build_training_data(issues: list = None) -> pd.DataFrame:
    issues = issues if issues is not None else load_issues()

    try:
        metrics = load_graph_metrics()
    except FileNotFoundError:
        metrics = None

    features_df = build_feature_dataframe(issues, metrics)

    lookup = {i["id"]: i for i in issues}
    features_df["days_to_close"] = features_df["issue_id"].map(lambda i: lookup[i].get("days_to_close"))
    features_df["state"] = features_df["issue_id"].map(lambda i: lookup[i].get("state"))
    features_df["title"] = features_df["issue_id"].map(lambda i: lookup[i].get("title", "")[:50])

    closed = features_df[features_df["state"] == "closed"].copy()
    if closed.empty or closed["days_to_close"].nunique() < 2:
        raise ValueError(
            "Not enough closed issues with varying days_to_close to train on."
        )

    median_days = closed["days_to_close"].median()
    closed["ground_truth_label"] = (closed["days_to_close"] > median_days).astype(int)

    print(f"Training set: {len(closed)} closed issues (median days_to_close = {median_days:.1f})")
    return closed


def train_and_evaluate(issues: list = None):
    df = build_training_data(issues)

    print(f"({int(df['ground_truth_label'].sum())} labeled 'harder', "
          f"{len(df) - int(df['ground_truth_label'].sum())} labeled 'easier')\n")

    model = DifficultyModel().fit(df)
    weights = model.feature_weights()

    print("Learned feature weights (logistic regression coefficients):")
    for feature, weight in sorted(weights.items(), key=lambda x: -abs(x[1])):
        direction = "increases" if weight > 0 else "decreases"
        print(f"  {feature:<28} {weight:+.3f}   ({direction} predicted difficulty)")

    n_splits = min(5, df["ground_truth_label"].value_counts().min())
    if n_splits >= 2:
        print(f"\nRunning stratified k-fold cross-validation ({n_splits} folds)...")
        cv_results = model.cross_validate(df, n_splits=n_splits)
        print(f"Per-fold AUC: {[round(a, 2) for a in cv_results['fold_aucs']]}")
        print(f"Mean AUC: {cv_results['mean_auc']:.3f} (+/- {cv_results['std_auc']:.3f})")
        print(f"\nClassification report:\n{cv_results['classification_report']}")
    else:
        print("\nSkipping cross-validation — not enough issues in the minority class yet.")
        cv_results = None

    df_with_formula = add_weighted_scores(df)
    df_with_formula["model_predicted_proba"] = model.predict_proba(df)
    comparison = df_with_formula[
        ["issue_id", "title", "difficulty_score_formula", "model_predicted_proba", "ground_truth_label"]
    ].sort_values("model_predicted_proba")

    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 50)
    print("\nWeighted formula vs. trained model, side by side:")
    print(comparison.to_string(index=False))

    return df, model, cv_results


if __name__ == "__main__":
    train_and_evaluate()