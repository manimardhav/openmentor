"""
ablation_study.py — compares ranking methods against ground truth on the 9-repo data.

1. POPULATION: CLOSED issues only. Ground truth (resolved_by_newcomer) exists only for
   closed issues; open issues are unlabeled, not negatives.
2. LEAKAGE: centrality, centrality_known and referenced_files come from the PR that
   resolved the issue (known for ~37% of closed issues and 0% of open ones). So every
   method is reported two ways: with all features, and with PRE-RESOLUTION features
   only (text_length, issue_age_days, first_contribution_label) — the honest estimate
   for issues someone could pick up today.
3. STATISTICAL POWER: reports ROC-AUC, precision@10/25/50 and lift over the base rate
   (precision@1/3/5 swings 0..1 off a single issue). Ties are broken randomly and
   averaged over 20 seeds. Trained models use 5-fold cross-validated predictions.

Lower difficulty score = ranked first (easiest first, for newcomers).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from issue_loader import load_issues
from graph_metrics_loader import load_graph_metrics
from features import build_feature_dataframe, FEATURE_COLUMNS
from skill_matching import match_score
from ground_truth_loader import load_ground_truth
from difficulty_model import DEFAULT_WEIGHTS

PRE_RESOLUTION_FEATURES = ["text_length", "issue_age_days", "first_contribution_label"]
KS = (10, 25, 50)
DEMO_CONTRIBUTOR_SKILLS = {"testing/CI", "packaging/build"}

FORMULA_WEIGHTS = {
    "formula[all features]": DEFAULT_WEIGHTS,
    "formula[pre-resolution only]": {"text_length": 0.15, "issue_age_days": 0.15, "first_contribution_label": -0.20},
    "formula[centrality only]": {"centrality": 1.0},
    "formula[text+age only]": {"text_length": 0.5, "issue_age_days": 0.5},
}


def build_eval_frame(issues: list = None) -> pd.DataFrame:
    """One row per CLOSED issue: features + repo + real outcomes."""
    issues = issues if issues is not None else load_issues()
    try:
        metrics = load_graph_metrics()
    except FileNotFoundError:
        metrics = None

    feats = build_feature_dataframe(issues, metrics)
    meta = pd.DataFrame([{
        "issue_id": i["id"], "repo_name": i["repo_name"], "body": i["body"],
        "created_at": pd.to_datetime(i["created_at"], utc=True),
        "days_to_close": i["days_to_close"],
    } for i in issues])
    gt = (load_ground_truth()[["unique_id", "resolved_by_newcomer", "had_gfi_label"]]
          .rename(columns={"unique_id": "issue_id"}))

    frame = feats.merge(meta, on="issue_id").merge(gt, on="issue_id", how="inner")  # inner => closed only

    # Some closed issues have an UNKNOWN resolver (resolved_by_newcomer is blank).
    # Unknown is not "not a newcomer", so these are excluded rather than scored as negatives.
    unknown = frame["resolved_by_newcomer"].isna()
    if unknown.any():
        print(f"[note] excluding {int(unknown.sum())} closed issues with unknown resolver from evaluation")
    frame = frame[~unknown].copy()
    frame["relevant"] = frame["resolved_by_newcomer"].astype(int)
    frame["hard"] = (frame["days_to_close"] > frame["days_to_close"].median()).astype(int)
    return frame.reset_index(drop=True)


def cv_predicted_difficulty(frame: pd.DataFrame, feature_cols: list, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Out-of-fold probability of 'hard' (days_to_close above median) — never predicted on training rows."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(LogisticRegression(max_iter=1000), frame[feature_cols], frame["hard"],
                             cv=skf, method="predict_proba")[:, 1]


def formula_score(frame: pd.DataFrame, weights: dict) -> np.ndarray:
    return sum(w * frame[f].to_numpy() for f, w in weights.items())


def precision_at_k(ease: np.ndarray, y: np.ndarray, k: int, n_seeds: int = 20) -> float:
    """Precision of the top-k by `ease` (higher = ranked first); ties broken randomly, averaged over seeds."""
    k = min(k, len(ease))
    vals = []
    for s in range(n_seeds):
        jitter = np.random.default_rng(s).random(len(ease))
        order = np.lexsort((jitter, -ease))
        vals.append(y[order[:k]].mean())
    return float(np.mean(vals))


def evaluate_ease(ease: np.ndarray, y: np.ndarray, ks=KS) -> dict:
    base = float(y.mean())
    out = {"base_rate": round(base, 3)}
    out["auc"] = round(float(roc_auc_score(y, ease)), 3) if 0 < y.sum() < len(y) else float("nan")
    for k in ks:
        p = precision_at_k(ease, y, k)
        out[f"precision@{k}"] = round(p, 3)
        out[f"lift@{k}"] = round(p / base, 2) if base > 0 else float("nan")
    return out


def all_method_ease(frame: pd.DataFrame) -> dict:
    """ease score (higher = recommended first) for every method, over the given frame."""
    methods = {}
    for name, weights in FORMULA_WEIGHTS.items():
        methods[name] = -formula_score(frame, weights)
    methods["trained LR (CV) [all features]"] = -cv_predicted_difficulty(frame, FEATURE_COLUMNS)
    methods["trained LR (CV) [pre-resolution only]"] = -cv_predicted_difficulty(frame, PRE_RESOLUTION_FEATURES)
    methods["skill-match only [demo profile, not personalized eval]"] = np.array(
        [match_score(DEMO_CONTRIBUTOR_SKILLS, b) for b in frame["body"]])
    methods["baseline: GFI label"] = frame["had_gfi_label"].astype(float).to_numpy()
    methods["baseline: oldest first"] = -frame["created_at"].astype("int64").to_numpy().astype(float)
    return methods


def run_ablation_study(frame: pd.DataFrame = None, methods: dict = None) -> pd.DataFrame:
    frame = frame if frame is not None else build_eval_frame()
    methods = methods if methods is not None else all_method_ease(frame)
    y = frame["relevant"].to_numpy()
    rows = {name: evaluate_ease(ease, y) for name, ease in methods.items()}
    base = float(y.mean())
    rows["baseline: random (expected)"] = {
        "base_rate": round(base, 3), "auc": 0.5,
        **{f"precision@{k}": round(base, 3) for k in KS}, **{f"lift@{k}": 1.0 for k in KS},
    }
    return pd.DataFrame(rows).T


def run_ablation_per_repo(frame: pd.DataFrame = None, methods: dict = None, k: int = 25) -> pd.DataFrame:
    """Per-repo AUC and lift@k for the methods that matter most for the report."""
    frame = frame if frame is not None else build_eval_frame()
    methods = methods if methods is not None else all_method_ease(frame)
    keep = ["formula[all features]", "formula[pre-resolution only]",
            "trained LR (CV) [all features]", "trained LR (CV) [pre-resolution only]",
            "baseline: GFI label", "baseline: oldest first"]
    rows = []
    for repo in sorted(frame["repo_name"].unique()):
        mask = (frame["repo_name"] == repo).to_numpy()
        y = frame.loc[mask, "relevant"].to_numpy()
        row = {"repo": repo, "closed_issues": int(mask.sum()), "newcomer_resolved": int(y.sum())}
        for name in keep:
            res = evaluate_ease(methods[name][mask], y, ks=(k,))
            row[f"{name} | AUC"] = res["auc"]
            row[f"{name} | lift@{k}"] = res[f"lift@{k}"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("repo")


ORIGINAL_REPOS = {"matplotlib/matplotlib", "pandas-dev/pandas", "huggingface/datasets", "networkx/networkx"}


def run_by_repo_group(frame: pd.DataFrame = None) -> pd.DataFrame:
    """
    Same methodology on: the ORIGINAL 4 repos (the "before" population), the 5 NEW repos,
    and all 9. Models are re-cross-validated within each group.
    """
    frame = frame if frame is not None else build_eval_frame()
    groups = {
        "original 4 repos": frame[frame["repo_name"].isin(ORIGINAL_REPOS)].reset_index(drop=True),
        "new 5 repos": frame[~frame["repo_name"].isin(ORIGINAL_REPOS)].reset_index(drop=True),
        "all 9 repos": frame,
    }
    keep = ["formula[all features]", "trained LR (CV) [all features]",
            "formula[pre-resolution only]", "trained LR (CV) [pre-resolution only]",
            "baseline: GFI label", "baseline: oldest first"]
    rows = []
    for gname, sub in groups.items():
        methods = all_method_ease(sub)
        y = sub["relevant"].to_numpy()
        for name in keep:
            r = evaluate_ease(methods[name], y)
            rows.append({"group": gname, "closed issues": len(sub), "base rate": r["base_rate"], "method": name,
                         "AUC": r["auc"], "precision@25": r["precision@25"], "lift@25": r["lift@25"],
                         "precision@50": r["precision@50"], "lift@50": r["lift@50"]})
    return pd.DataFrame(rows).set_index(["group", "method"])


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    frame = build_eval_frame()
    print(f"Evaluation population: {len(frame)} closed issues, "
          f"{int(frame['relevant'].sum())} resolved by a newcomer ({frame['relevant'].mean():.1%} base rate)\n")

    methods = all_method_ease(frame)
    print("Overall (all 9 repos pooled) — higher AUC / lift is better; random = AUC 0.5, lift 1.0:")
    print(run_ablation_study(frame, methods).to_string())

    print("\nPer repo (k=25):")
    print(run_ablation_per_repo(frame, methods).to_string())

    print("\nBefore/after — same methodology on original 4 repos vs new 5 vs all 9:")
    print(run_by_repo_group(frame).to_string())