"""
match_issues_demo.py — ties together skill_matching.py with real
matplotlib issues instead of mock data.
"""

import pandas as pd
from skill_matching import rank_issues_for_contributor, tag_issue_skills, SKILL_TAXONOMY
from issue_loader import load_issues


def run_demo(contributor_skills: set = None, open_only: bool = True):
    if contributor_skills is None:
        contributor_skills = {"testing/CI", "packaging/build"}

    issues = load_issues()
    if open_only:
        issues = [i for i in issues if i.get("state") == "open"]

    print(f"Contributor skills: {contributor_skills}\n")

    print("Tagged skills per issue:")
    for issue in issues:
        tags = tag_issue_skills(issue["body"], SKILL_TAXONOMY)
        print(f"  #{issue['id']} ({issue['title'][:45]:<45}) -> {tags if tags else '(none)'}")

    ranked = rank_issues_for_contributor(contributor_skills, issues, metric="cosine")

    lookup = {i["id"]: i["title"][:50] for i in issues}
    ranked["title"] = ranked["issue_id"].map(lookup)

    print("\nRanked by skill match (highest first):")
    pd.set_option("display.width", 140)
    print(ranked[["issue_id", "title", "match_score"]].to_string(index=False))

    return ranked


if __name__ == "__main__":
    run_demo()