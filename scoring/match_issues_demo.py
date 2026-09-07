"""
match_issues_demo.py — ties together skill_matching.py (tagging + similarity)
and contributor_profile.py (skill input) into one runnable demo, using the
real issue titles from sanity_check.py. This is the hand-check for skill
matching, same spirit as train_difficulty_model.py's hand-check for the
scoring model.
"""

import pandas as pd
from skill_matching import rank_issues_for_contributor, tag_issue_skills, SKILL_TAXONOMY
from contributor_profile import prompt_contributor_skills
from sanity_check import REAL_ISSUES


def run_demo(contributor_skills: set = None):
    if contributor_skills is None:
        # Non-interactive default profile for quick testing.
        # Run contributor_profile.py directly for the real interactive form.
        contributor_skills = {"HTTP/networking", "testing"}

    print(f"Contributor skills: {contributor_skills}\n")

    # Show what the tagger inferred for each issue first, for transparency.
    print("Tagged skills per issue:")
    for issue in REAL_ISSUES:
        tags = tag_issue_skills(issue["body"], SKILL_TAXONOMY)
        print(f"  #{issue['id']} ({issue['body'][:45]:<45}) -> {tags if tags else '(none)'}")

    ranked = rank_issues_for_contributor(contributor_skills, REAL_ISSUES, metric="cosine")

    # attach titles for readability
    lookup = {i["id"]: i["body"][:50] for i in REAL_ISSUES}
    ranked["title"] = ranked["issue_id"].map(lookup)

    print("\nRanked by skill match (highest first):")
    pd.set_option("display.width", 140)
    print(ranked[["issue_id", "title", "match_score"]].to_string(index=False))

    return ranked


if __name__ == "__main__":
    run_demo()