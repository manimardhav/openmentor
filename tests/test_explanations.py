"""
Unit tests for interface/explanations.py — the templated explanation
builder and its event log, kept free of Streamlit so they run fast in CI
with no UI runtime needed.

Run with: pytest tests/test_explanations.py -v
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "interface"))

from explanations import explain_score, log_explanation_event  # noqa: E402


def make_issue(**top_features):
    return {
        "issue_id": "123",
        "difficulty_score": 0.4,
        "skill_match_score": 0.7,
        "top_features": top_features,
    }


def test_explain_score_mentions_high_centrality():
    issue = make_issue(centrality=0.8)
    sentence, referenced = explain_score(issue)
    assert "highly central" in sentence
    assert "0.80" in sentence
    assert referenced == ["centrality"]


def test_explain_score_mentions_first_contribution_label():
    issue = make_issue(first_contribution_label=True)
    sentence, referenced = explain_score(issue)
    assert "first contribution" in sentence
    assert "first_contribution_label" in referenced


def test_explain_score_omits_false_first_contribution_label():
    issue = make_issue(first_contribution_label=False)
    sentence, referenced = explain_score(issue)
    assert "first contribution" not in sentence
    # the key is still present (feature was looked at), just not surfaced in prose
    assert "first_contribution_label" not in referenced


def test_explain_score_includes_difficulty_and_skill_match_numbers():
    issue = make_issue()
    sentence, _ = explain_score(issue)
    assert "0.40" in sentence
    assert "0.70" in sentence


def test_explain_score_falls_back_when_no_features_present():
    issue = {"issue_id": "1", "difficulty_score": 0.1, "skill_match_score": 0.1, "top_features": {}}
    sentence, referenced = explain_score(issue)
    assert "overall difficulty and skill-match balance" in sentence
    assert referenced == []


def test_explain_score_never_mentions_a_feature_not_in_referenced_list():
    """Guards against the explanation drifting from the features that
    actually drove it (the exact failure mode the docx Week 3 'Added'
    item calls out)."""
    issue = make_issue(centrality=0.9, referenced_files=3)
    sentence, referenced = explain_score(issue)
    known_feature_words = {
        "centrality": "central",
        "first_contribution_label": "first contribution",
        "referenced_files": "references",
        "issue_age_days": "been open",
    }
    for feature, keyword in known_feature_words.items():
        if keyword in sentence:
            assert feature in referenced, (
                f"Sentence mentions '{keyword}' but '{feature}' isn't in the "
                "logged referenced-features list"
            )


def test_log_explanation_event_writes_one_json_line(tmp_path):
    log_path = tmp_path / "explanation_log.jsonl"
    log_explanation_event(
        issue_id="123",
        contributor_skills=["testing/CI"],
        features_referenced=["centrality"],
        event="explanation_shown",
        log_path=log_path,
    )
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["issue_id"] == "123"
    assert record["contributor_skills"] == ["testing/CI"]
    assert record["features_referenced"] == ["centrality"]
    assert record["event"] == "explanation_shown"
    assert "timestamp" in record


def test_log_explanation_event_appends_not_overwrites(tmp_path):
    log_path = tmp_path / "explanation_log.jsonl"
    log_explanation_event("1", [], [], "explanation_shown", log_path=log_path)
    log_explanation_event("2", [], [], "roadmap_generated", log_path=log_path)
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2