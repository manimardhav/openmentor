"""
experience_stratified_eval.py — newcomer vs returning-contributor breakdown.

Question: for issues that a FIRST-TIME contributor actually resolved, did the difficulty
model predict lower difficulty than for issues resolved by a returning contributor?

9-repo update:
- Predictions are 5-fold cross-validated (out-of-fold), not in-sample.
- Closed issues with an UNKNOWN resolver are excluded, not counted as "returning".
- Reported twice: with all features, and with PRE-RESOLUTION features only
  (centrality, centrality_known and referenced_files come from the resolving PR and are
  absent for every open issue).
- Broken down per repo, since a pooled number can hide repos where it fails.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ablation_study import build_eval_frame, cv_predicted_difficulty, PRE_RESOLUTION_FEATURES
from features import FEATURE_COLUMNS


def stratify_by_experience(frame: pd.DataFrame = None) -> pd.DataFrame:
    frame = frame if frame is not None else build_eval_frame()
    frame = frame.copy()
    frame["pred_difficulty[all features]"] = cv_predicted_difficulty(frame, FEATURE_COLUMNS)
    frame["pred_difficulty[pre-resolution]"] = cv_predicted_difficulty(frame, PRE_RESOLUTION_FEATURES)
    frame["resolver"] = np.where(frame["relevant"] == 1, "first-time", "returning")
    return frame


def summarize(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean predicted difficulty by resolver type, per repo + pooled. gap < 0 => newcomers' issues scored easier (good)."""
    rows = []
    groups = [("ALL 9 REPOS", frame)] + [(r, frame[frame["repo_name"] == r]) for r in sorted(frame["repo_name"].unique())]
    for name, sub in groups:
        ft = sub.loc[sub["resolver"] == "first-time", col]
        rt = sub.loc[sub["resolver"] == "returning", col]
        auc = roc_auc_score(sub["relevant"], -sub[col]) if sub["relevant"].nunique() == 2 else float("nan")
        rows.append({
            "group": name, "first-time n": len(ft), "returning n": len(rt),
            "mean difficulty (first-time)": round(ft.mean(), 3),
            "mean difficulty (returning)": round(rt.mean(), 3),
            "gap (first-time - returning)": round(ft.mean() - rt.mean(), 3),
            "AUC": round(auc, 3),
        })
    return pd.DataFrame(rows).set_index("group")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    frame = stratify_by_experience()
    for col in ["pred_difficulty[all features]", "pred_difficulty[pre-resolution]"]:
        print(f"\n=== {col} ===")
        print("gap < 0 and AUC > 0.5 mean the model rated newcomer-resolved issues as easier.")
        print(summarize(frame, col).to_string())