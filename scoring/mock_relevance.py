"""
mock_relevance.py — placeholder "relevant issue" ground truth, used ONLY to
exercise evaluation_metrics.py and ablation_study.py before Person C's real
tester feedback exists. Replace entirely once real tester data lands.
"""

from sanity_check import REAL_ISSUES
from skill_matching import tag_issue_skills, SKILL_TAXONOMY

DEMO_CONTRIBUTOR_SKILLS = {"HTTP/networking", "testing"}
ACCEPTABLE_DIFFICULTY = {"easy", "easy-medium", "medium", "medium-hard"}


def build_mock_relevant_set() -> set:
    relevant = set()
    for issue in REAL_ISSUES:
        tags = tag_issue_skills(issue["body"], SKILL_TAXONOMY)
        skill_overlap = bool(tags & DEMO_CONTRIBUTOR_SKILLS)
        right_difficulty = issue["your_intuition"] in ACCEPTABLE_DIFFICULTY
        if skill_overlap and right_difficulty:
            relevant.add(issue["id"])
    return relevant


if __name__ == "__main__":
    relevant = build_mock_relevant_set()
    print(f"Mock 'relevant' issues for demo contributor: {relevant}")
    for issue in REAL_ISSUES:
        marker = "RELEVANT" if issue["id"] in relevant else ""
        print(f"  #{issue['id']} ({issue['your_intuition']:<12}) {issue['body'][:50]:<50} {marker}")