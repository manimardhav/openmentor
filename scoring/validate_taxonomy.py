"""
validate_taxonomy.py — checks how often each skill tag fires across the
REAL local issue data. No more live GitHub API calls — instant and
reproducible.
"""

from collections import Counter
from issue_loader import load_issues
from skill_matching import tag_issue_skills, SKILL_TAXONOMY


def validate_taxonomy(issues: list = None, taxonomy: dict = None) -> None:
    issues = issues if issues is not None else load_issues()
    taxonomy = taxonomy or SKILL_TAXONOMY

    tag_counts = Counter()
    zero_tag_count = 0

    for issue in issues:
        matched = tag_issue_skills(issue.get("body", ""), taxonomy)
        if not matched:
            zero_tag_count += 1
        for tag in matched:
            tag_counts[tag] += 1

    total = len(issues)
    print(f"Checked {total} real issues\n")
    print(f"{'Tag':<24}{'Hits':>6}{'Hit rate':>12}")
    print("-" * 42)
    for tag in taxonomy:
        hits = tag_counts.get(tag, 0)
        rate = f"{hits/total:.0%}" if total else "n/a"
        print(f"{tag:<24}{hits:>6}{rate:>12}")

    print("-" * 42)
    if total:
        rate = f"{zero_tag_count/total:.0%}"
        print(f"{'Issues with NO tag matched':<24}{zero_tag_count:>6}{rate:>12}")

    print("\nInterpretation:")
    print("- A tag with ~0% hit rate is either mis-worded or genuinely rare.")
    print("- A high 'no tag matched' rate means the taxonomy is too narrow.")


if __name__ == "__main__":
    validate_taxonomy()