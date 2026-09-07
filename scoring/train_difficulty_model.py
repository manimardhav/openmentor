"""
train_difficulty_model.py — turns the weighted difficulty formula into a
trainable, inspectable logistic regression model.

GROUND-TRUTH NOTE: real historical difficulty labels aren't available yet —
Person A's pipeline and Round 1 tester feedback haven't landed. Until then,
this uses hand-labeled intuition from sanity_check.py as a stand-in ground
truth, purely to validate the model mechanics (training, cross-validation,
weight inspection) work correctly. Re-run this exact script once real
labels exist — nothing about the code changes, only the label source.
"""

import pandas as pd
from features import build_feature_dataframe
from difficulty_model import DifficultyModel, add_weighted_scores, FEATURE_COLUMNS
from build_placeholder_graph import build_placeholder_graph
from sanity_check import REAL_ISSUES

# Bucket the 5-level hand intuition into binary labels (0 = easier, 1 = harder).
INTUITION_TO_LABEL = {
    "easy": 0,
    "easy-medium": 0,
    "medium": 1,
    "medium-hard": 1,
    "hard": 1,
}


def build_training_data() -> pd.DataFrame:
    graph = build_placeholder_graph()
    features_df = build_feature_dataframe(REAL_ISSUES, graph)

    lookup = {i["id"]: i for i in REAL_ISSUES}
    features_df["ground_truth_label"] = features_df["issue_id"].map(
        lambda i: INTUITION_TO_LABEL[lookup[i]["your_intuition"]]
    )
    features_df["title"] = features_df["issue_id"].map(lambda i: lookup[i]["body"][:50])
    return features_df


def train_and_evaluate():
    df = build_training_data()

    print(f"Training set: {len(df)} issues "
          f"({df['ground_truth_label'].sum()} labeled 'harder', "
          f"{len(df) - df['ground_truth_label'].sum()} labeled 'easier')\n")

    model = DifficultyModel().fit(df)
    weights = model.feature_weights()

    print("Learned feature weights (logistic regression coefficients):")
    for feature, weight in sorted(weights.items(), key=lambda x: -abs(x[1])):
        direction = "increases" if weight > 0 else "decreases"
        print(f"  {feature:<20} {weight:+.3f}   ({direction} predicted difficulty)")

    print(f"\nRunning stratified k-fold cross-validation...")
    cv_results = model.cross_validate(df, n_splits=5)
    n_folds_used = len(cv_results["fold_aucs"])
    print(f"(Note: capped to {n_folds_used} folds — limited by minority-class "
          f"count in this small {len(df)}-issue sample. Expect this to be a "
          f"documented limitation until the real dataset is larger.)\n")
    print(f"Per-fold AUC: {[round(a, 2) for a in cv_results['fold_aucs']]}")
    print(f"Mean AUC: {cv_results['mean_auc']:.3f} (+/- {cv_results['std_auc']:.3f})")
    print(f"\nClassification report:\n{cv_results['classification_report']}")

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