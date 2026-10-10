"""
ranking.py — combines difficulty + skill-match into one ranking score,
with a confidence tag.

9-repo update: confidence_tag() is now TWO-valued ("high-confidence" |
"exploratory"), as required by shared/schemas.md and
tests/test_ranked_issues_schema.py. It used to also return "low-confidence",
which violated that contract (13 of 424 issues failed the schema test).
"""

import pandas as pd
from skill_matching import match_score, SKILL_TAXONOMY


def difficulty_fit(predicted_difficulty: float, contributor_level: float) -> float:
    """
    Penalize distance between issue difficulty and contributor's stated
    experience level (both expected in [0, 1]) rather than just preferring
    "easy" issues outright — an easy issue is a poor match for an
    experienced contributor too.
    """
    return 1.0 - abs(predicted_difficulty - contributor_level)


def combined_rank_score(
    predicted_difficulty: float,
    contributor_level: float,
    contributor_skills: set,
    issue_text: str,
    alpha: float = 0.5,
    taxonomy: dict = None,
) -> dict:
    """
    alpha weights difficulty-fit vs skill-match. Returns the combined score
    plus its components, so you can inspect why an issue ranked where it did.
    """
    fit = difficulty_fit(predicted_difficulty, contributor_level)
    skill = match_score(contributor_skills, issue_text, taxonomy, metric="cosine")
    combined = alpha * fit + (1 - alpha) * skill

    return {
        "difficulty_fit": fit,
        "skill_match": skill,
        "combined_score": combined,
    }


def confidence_tag(combined_score: float, skill_match: float, high_threshold: float = 0.7, low_threshold: float = 0.4) -> str:
    """
    Two-valued tag per the project's data contract: "high-confidence" | "exploratory".
    Anything that isn't high-confidence is "exploratory".
    low_threshold is unused; it stays in the signature only so existing callers don't break.
    """
    if combined_score >= high_threshold and skill_match >= 0.5:
        return "high-confidence"
    return "exploratory"


def rank_issues_for_contributor_full(
    contributor_level: float,
    contributor_skills: set,
    issues: list,
    alpha: float = 0.5,
    taxonomy: dict = None,
) -> pd.DataFrame:
    """
    issues: list of dicts with keys 'id', 'body', 'predicted_difficulty'
    ('id' should be the unique_id from issue_loader.py).
    """
    rows = []
    for issue in issues:
        scores = combined_rank_score(
            predicted_difficulty=issue["predicted_difficulty"],
            contributor_level=contributor_level,
            contributor_skills=contributor_skills,
            issue_text=issue.get("body", ""),
            alpha=alpha,
            taxonomy=taxonomy,
        )
        rows.append({
            "issue_id": issue["id"],
            **scores,
            "confidence_tag": confidence_tag(scores["combined_score"], scores["skill_match"]),
        })
    return pd.DataFrame(rows).sort_values("combined_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    contributor_skills = {"testing/CI", "packaging/build"}
    contributor_level = 0.3  # relatively new contributor

    toy_issues = [
        {"id": "demo/repo#1", "body": "Add pytest coverage for the nightly CI wheel build", "predicted_difficulty": 0.25},
        {"id": "demo/repo#2", "body": "Rewrite the core rendering backend", "predicted_difficulty": 0.9},
        {"id": "demo/repo#3", "body": "pip install fails on Windows", "predicted_difficulty": 0.2},
    ]

    print(rank_issues_for_contributor_full(contributor_level, contributor_skills, toy_issues))