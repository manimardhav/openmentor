"""
experience_stratified_eval.py — stratifies evaluation by contributor
experience level, using REAL resolver_is_first_time data. No fabrication
needed anymore.
"""

import pandas as pd
from issue_loader import load_issues
from train_difficulty_model import build_training_data
from difficulty_model import DifficultyModel


def stratify_by_experience(issues: list = None) -> pd.DataFrame:
    issues = issues if issues is not None else load_issues()

    df = build_training_data(issues)
    model = DifficultyModel().fit(df)
    df["predicted_difficulty"] = model.predict_proba(df)

    lookup = {i["id"]: i for i in issues}
    df["resolver_is_first_time"] = df["issue_id"].map(lambda i: lookup[i].get("resolver_is_first_time"))

    return df[["issue_id", "title", "resolver_is_first_time", "predicted_difficulty"]]


def summarize_by_group(stratified_df: pd.DataFrame) -> pd.DataFrame:
    return (stratified_df.groupby("resolver_is_first_time")["predicted_difficulty"]
            .agg(["mean", "count"]).reset_index())


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 50)

    stratified = stratify_by_experience()
    print("Per-issue results (closed issues only, real resolver data):")
    print(stratified.to_string(index=False))

    print("\nSummary by resolver experience:")
    print(summarize_by_group(stratified).to_string(index=False))