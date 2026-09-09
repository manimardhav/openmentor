"""
difficulty_model.py — Week 2 (weighted formula) and Week 3 (trainable model).

Two scorers are provided so you can compare them directly in the ablation
work later: a hand-tuned weighted sum (fast, no training data needed) and a
logistic regression trained on ground-truth difficulty labels.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score

FEATURE_COLUMNS = ["centrality", "text_length", "issue_age_days", "first_contribution_label", "referenced_files"]
# ---------------------------------------------------------------------------
# Week 2: hand-tuned weighted formula
# ---------------------------------------------------------------------------

# Starting weights — treat these as a first guess to sanity-check against
# hand-labeled easy/hard/ambiguous issues (see Week 2 plan), then refine.
DEFAULT_WEIGHTS = {
    "centrality": 0.35,
    "text_length": 0.15,
    "issue_age_days": 0.15,
    "first_contribution_label": -0.20,   # presence of the label should LOWER predicted difficulty
    "referenced_files": 0.25,
}


def weighted_difficulty_score(feature_row: pd.Series, weights: dict = None) -> float:
    """Returns a raw difficulty score (unbounded but roughly 0-1 given normalized inputs)."""
    weights = weights or DEFAULT_WEIGHTS
    return sum(weights[f] * feature_row[f] for f in FEATURE_COLUMNS)


def add_weighted_scores(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    df = df.copy()
    df["difficulty_score_formula"] = df.apply(lambda r: weighted_difficulty_score(r, weights), axis=1)
    return df


# ---------------------------------------------------------------------------
# Week 3: trainable, inspectable model
# ---------------------------------------------------------------------------

class DifficultyModel:
    """
    Thin wrapper around LogisticRegression so the rest of the pipeline can
    treat it interchangeably with the weighted-formula scorer.

    ground_truth_label should be binary for a first pass (e.g. 0 = easy,
    1 = hard) — you can bucket a continuous difficulty rating into
    quartiles/median-split if that's what your historical data looks like.
    """

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, label_col: str = "ground_truth_label"):
        X = df[FEATURE_COLUMNS]
        y = df[label_col]
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before predicting.")
        return self.model.predict_proba(df[FEATURE_COLUMNS])[:, 1]

    def feature_weights(self) -> dict:
        """For the methodology write-up: exact learned coefficients."""
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before inspecting weights.")
        return dict(zip(FEATURE_COLUMNS, self.model.coef_[0]))

    def cross_validate(self, df: pd.DataFrame, label_col: str = "ground_truth_label", n_splits: int = 5):
        """
        Stratified k-fold cross-validation — use this instead of a single
        train/test split on small datasets (see Week 3 plan). Returns a
        classification report plus per-fold AUC so you can report variance,
        not just a single accuracy number.
        """
        X = df[FEATURE_COLUMNS]
        y = df[label_col]
        n_splits = min(n_splits, y.value_counts().min())  # can't have more folds than minority-class examples
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_aucs = []
        for train_idx, test_idx in skf.split(X, y):
            fold_model = LogisticRegression(max_iter=1000)
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = fold_model.predict_proba(X.iloc[test_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y.iloc[test_idx], proba))

        preds = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=skf)
        report = classification_report(y, preds)

        return {
            "fold_aucs": fold_aucs,
            "mean_auc": float(np.mean(fold_aucs)),
            "std_auc": float(np.std(fold_aucs)),
            "classification_report": report,
        }


if __name__ == "__main__":
    # Smoke test with synthetic data
    rng = np.random.default_rng(42)
    n = 60
    synthetic = pd.DataFrame({
        "centrality": rng.random(n),
        "text_length": rng.random(n),
        "issue_age_days": rng.random(n),
        "first_contribution_label": rng.integers(0, 2, n),
        "referenced_files": rng.random(n),
    })
    # Fake ground truth correlated with centrality + referenced_files for the smoke test
    synthetic["ground_truth_label"] = (
        (synthetic["centrality"] + synthetic["referenced_files"]) > 1.0
    ).astype(int)

    synthetic = add_weighted_scores(synthetic)
    print("Weighted formula scores (first 5):")
    print(synthetic[["difficulty_score_formula"]].head())

    model = DifficultyModel().fit(synthetic)
    print("\nLearned feature weights:", model.feature_weights())

    cv_results = model.cross_validate(synthetic, n_splits=5)
    print(f"\nMean AUC across folds: {cv_results['mean_auc']:.3f} (+/- {cv_results['std_auc']:.3f})")
    print(cv_results["classification_report"])
