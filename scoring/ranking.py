"""
ranking.py — Week 5: combine difficulty + skill-match into one ranking
function, with confidence/threshold tagging.
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
    alpha weights difficulty-fit vs skill-match (tune during Week 5 testing).
    Returns the combined score plus its components, so you can inspect why
    an issue ranked where it did (useful for the "match reasoning" you're
    preparing for Round 1 testers).
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
    Simple threshold-based confidence tag (Week 5 addition). Tune thresholds
    against Round 1 tester feedback rather than treating these as fixed.
    """
    if combined_score >= high_threshold and skill_match >= 0.5:
        return "high-confidence"
    elif combined_score >= low_threshold:
        return "exploratory"
    else:
        return "low-confidence"


def rank_issues_for_contributor_full(
    contributor_level: float,
    contributor_skills: set,
    issues: list,
    alpha: float = 0.5,
    taxonomy: dict = None,
) -> pd.DataFrame:
    """
    issues: list of dicts with keys 'id', 'body', 'predicted_difficulty'
    (predicted_difficulty should come from difficulty_model.py's output).
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
    # httpie/cli-style toy issues
    contributor_skills = {"testing", "HTTP/networking"}
    contributor_level = 0.3  # relatively new contributor

    toy_issues = [
        {"id": 1, "body": "Add pytest unit tests covering redirect handling for HTTP requests", "predicted_difficulty": 0.25},
        {"id": 2, "body": "Rewrite the core plugin loading and hook system", "predicted_difficulty": 0.9},
        {"id": 3, "body": "Fix argparse flag parsing for --auth token", "predicted_difficulty": 0.2},
    ]

    ranked = rank_issues_for_contributor_full(contributor_level, contributor_skills, toy_issues)
    print(ranked)
