"""
topk_report.py — precision@k, recall@k and hit-rate@k (default k=5), the exact
metrics Member 3's prompt asked for, for a before/after comparison.

Each REPO is ranked on its own (a contributor browses one repo's list), then
the per-repo numbers are averaged (macro average). Evaluated on closed issues
with a known resolver; models use cross-validated predictions; ties are broken
randomly and averaged over 20 seeds.

Groups: the ORIGINAL 4 repos (the "before" population), the 5 NEW repos, all 9.
"Random" is the exact expectation (hypergeometric), not one lucky shuffle.

Read with care: with ~13-23% of issues relevant, a RANDOM top-5 already
contains at least one relevant issue ~50-85% of the time, so hit-rate@5 barely
separates methods. recall@5 is small by construction (5 slots vs. 26-231
relevant issues per repo). precision@5 per repo is noisy (5 issues). Prefer the
AUC / lift@50 numbers in ablation_study.py for conclusions.
"""

import numpy as np
import pandas as pd

from ablation_study import build_eval_frame, all_method_ease, ORIGINAL_REPOS

METHODS = ["trained LR (CV) [all features]", "trained LR (CV) [pre-resolution only]",
           "formula[all features]", "formula[pre-resolution only]",
           "baseline: GFI label", "baseline: oldest first"]


def topk_one_ranking(ease: np.ndarray, y: np.ndarray, k: int, n_seeds: int = 20):
    n_pos = int(y.sum())
    prec, rec, hit = [], [], []
    for s in range(n_seeds):
        jitter = np.random.default_rng(s).random(len(ease))
        top = y[np.lexsort((jitter, -ease))[:k]]
        prec.append(top.mean())
        rec.append(top.sum() / n_pos if n_pos else np.nan)
        hit.append(float(top.sum() > 0))
    return float(np.mean(prec)), float(np.mean(rec)), float(np.mean(hit))


def random_expectation(y: np.ndarray, k: int):
    n, n_pos = len(y), int(y.sum())
    p_none = np.prod([(n - n_pos - i) / (n - i) for i in range(k)])
    return n_pos / n, k / n, 1.0 - float(p_none)


def run_topk_report(frame: pd.DataFrame = None, k: int = 5) -> pd.DataFrame:
    frame = frame if frame is not None else build_eval_frame()
    groups = {
        "original 4 repos": frame[frame["repo_name"].isin(ORIGINAL_REPOS)].reset_index(drop=True),
        "new 5 repos": frame[~frame["repo_name"].isin(ORIGINAL_REPOS)].reset_index(drop=True),
        "all 9 repos": frame,
    }
    rows = []
    for gname, sub in groups.items():
        methods = all_method_ease(sub)
        repos = sorted(sub["repo_name"].unique())
        for name in METHODS + ["baseline: random (expected)"]:
            per_repo = []
            for repo in repos:
                mask = (sub["repo_name"] == repo).to_numpy()
                y = sub.loc[mask, "relevant"].to_numpy()
                if name.startswith("baseline: random"):
                    per_repo.append(random_expectation(y, k))
                else:
                    per_repo.append(topk_one_ranking(methods[name][mask], y, k))
            p, r, h = np.mean(per_repo, axis=0)
            rows.append({"group": gname, "method": name, f"precision@{k}": round(p, 3),
                         f"recall@{k}": round(r, 4), f"hit-rate@{k}": round(h, 3)})
    return pd.DataFrame(rows).set_index(["group", "method"])


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print("Top-k metrics, one ranking per repo, macro-averaged over repos.\n")
    print(run_topk_report(k=5).to_string())