"""
experience_stratified_eval.py — Week 6: stratify evaluation results by
contributor experience level (first-time vs. returning).

MOCK DATA NOTE: MOCK_RESOLVER_HISTORY is entirely fabricated. Replace it
wholesale once Person C's real tester data lands — the stratify_by_experience()
function itself doesn't need to change.
"""

import pandas as pd
from full_pipeline import run_full_pipeline

MOCK_RESOLVER_HISTORY = [
    {"contributor": "alice", "experience": "first-time", "resolved_issue_id": 1603, "skills": {"HTTP/networking"}, "level": 0.1},
    {"contributor": "bob",   "experience": "first-time", "resolved_issue_id": 1574, "skills": {"CLI/argparse"}, "level": 0.0},
    {"contributor": "carol", "experience": "returning",  "resolved_issue_id": 1555, "skills": {"CLI/argparse"}, "level": 0.8},
    {"contributor": "dave",  "experience": "returning",  "resolved_issue_id": 1637, "skills": {"HTTP/networking", "testing"}, "level": 0.75},
]


def stratify_by_experience(resolver_history: list = None) -> pd.DataFrame:
    resolver_history = resolver_history or MOCK_RESOLVER_HISTORY
    rows = []

    for record in resolver_history:
        ranking = run_full_pipeline(
            contributor_skills=record["skills"],
            contributor_level=record["level"],
        )
        ranking = ranking.reset_index(drop=True)
        ranking["rank"] = ranking.index + 1

        match = ranking[ranking["issue_id"] == record["resolved_issue_id"]]
        rank_of_resolved_issue = int(match["rank"].iloc[0]) if not match.empty else None

        rows.append({
            "contributor": record["contributor"],
            "experience": record["experience"],
            "resolved_issue_id": record["resolved_issue_id"],
            "rank_system_gave_it": rank_of_resolved_issue,
        })

    return pd.DataFrame(rows)


def summarize_by_group(stratified_df: pd.DataFrame) -> pd.DataFrame:
    return stratified_df.groupby("experience")["rank_system_gave_it"].agg(["mean", "count"]).reset_index()


if __name__ == "__main__":
    print("(Reminder: MOCK_RESOLVER_HISTORY is fabricated — real analysis awaits Person C's tester data.)\n")

    stratified = stratify_by_experience()
    print("Per-contributor results:")
    print(stratified.to_string(index=False))

    print("\nSummary by experience level:")
    print(summarize_by_group(stratified).to_string(index=False))