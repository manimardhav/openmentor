"""
Integration test: verifies data/ranked_issues.json (the one file every
downstream module — interface/app.py, llm/generate_roadmap.py — depends
on) actually matches the contract in shared/schemas.md.

This is deliberately an integration test, not a unit test: the whole
point is to catch schema drift between Person A's scoring output and what
the interface/LLM code expects, which is exactly the kind of bug that
broke run_pipeline.py before (see run_pipeline.py's module docstring).

Skips (does not fail) if data/ranked_issues.json hasn't been generated
yet, so this doesn't block a fresh checkout — but it always runs in CI
after `python3 llm/run_pipeline.py` has produced the file.

Run with: pytest tests/test_ranked_issues_schema.py -v
"""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RANKED_ISSUES_PATH = PROJECT_ROOT / "data" / "ranked_issues.json"

REQUIRED_KEYS = {
    "issue_id", "difficulty_score", "skill_match_score",
    "final_rank_score", "matched_skills", "confidence_tag", "top_features",
}
VALID_CONFIDENCE_TAGS = {"high-confidence", "exploratory"}


@pytest.fixture(scope="module")
def ranked_issues():
    if not RANKED_ISSUES_PATH.exists():
        pytest.skip(f"{RANKED_ISSUES_PATH} not generated yet — run llm/run_pipeline.py first")
    with open(RANKED_ISSUES_PATH) as f:
        return json.load(f)


def test_file_is_a_nonempty_list(ranked_issues):
    assert isinstance(ranked_issues, list)
    assert len(ranked_issues) > 0


def test_every_issue_has_required_keys(ranked_issues):
    for issue in ranked_issues:
        missing = REQUIRED_KEYS - issue.keys()
        assert not missing, f"Issue {issue.get('issue_id')} missing keys: {missing}"


def test_scores_are_floats_in_zero_one_range(ranked_issues):
    for issue in ranked_issues:
        for field in ("difficulty_score", "skill_match_score", "final_rank_score"):
            value = issue[field]
            assert isinstance(value, (int, float)), f"{field} on {issue['issue_id']} is not numeric"
            assert 0.0 <= value <= 1.0, f"{field} on {issue['issue_id']} = {value}, out of [0,1]"


def test_matched_skills_is_a_list_of_strings(ranked_issues):
    for issue in ranked_issues:
        assert isinstance(issue["matched_skills"], list)
        assert all(isinstance(s, str) for s in issue["matched_skills"])


def test_confidence_tag_is_one_of_the_two_allowed_values(ranked_issues):
    for issue in ranked_issues:
        assert issue["confidence_tag"] in VALID_CONFIDENCE_TAGS, (
            f"Issue {issue['issue_id']} has confidence_tag="
            f"{issue['confidence_tag']!r}, expected one of {VALID_CONFIDENCE_TAGS}"
        )


def test_issue_ids_are_unique(ranked_issues):
    ids = [issue["issue_id"] for issue in ranked_issues]
    assert len(ids) == len(set(ids)), "Duplicate issue_id values found in ranked_issues.json"


def test_sorted_descending_by_final_rank_score(ranked_issues):
    scores = [issue["final_rank_score"] for issue in ranked_issues]
    assert scores == sorted(scores, reverse=True), "ranked_issues.json is not sorted by final_rank_score"


def test_taxonomy_coverage_is_not_near_zero(ranked_issues):
    """Regression guard for the taxonomy-mismatch bug found during review:
    if the interface's skill list and the scoring taxonomy ever drift
    apart again, matched_skills would silently collapse toward empty for
    most issues. Fail loudly instead of shipping a demo where skill
    matching quietly does nothing."""
    with_matches = sum(1 for i in ranked_issues if i["matched_skills"])
    coverage = with_matches / len(ranked_issues)
    assert coverage > 0.3, (
        f"Only {coverage:.0%} of ranked issues have any matched_skills — "
        "likely a taxonomy mismatch between the interface's skill list and "
        "scoring.skill_matching.SKILL_TAXONOMY."
    )