"""
Unit tests for llm/validators.py — the grounding/format checks that guard
every LLM output before it reaches a user. These are pure functions (no
network calls), so they run fast and deterministically in CI.

Run with: pytest tests/test_validators.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm"))

from validators import (  # noqa: E402
    validate_extraction_json,
    check_roadmap_grounding,
    ValidationError,
    _strip_markdown_fences,
)

TAXONOMY = ["testing/CI", "documentation", "rendering/backends"]


# --- validate_extraction_json -------------------------------------------

def test_valid_extraction_passes():
    raw = json.dumps({
        "likely_skills": ["testing/CI"],
        "one_line_summary": "Fixes a flaky test",
        "estimated_complexity": "low",
    })
    result = validate_extraction_json(raw, TAXONOMY, allow_proposed=False)
    assert result["likely_skills"] == ["testing/CI"]
    assert result["estimated_complexity"] == "low"


def test_extraction_strips_markdown_fences():
    raw = "```json\n" + json.dumps({
        "likely_skills": [],
        "one_line_summary": "x",
        "estimated_complexity": "medium",
    }) + "\n```"
    result = validate_extraction_json(raw, TAXONOMY, allow_proposed=False)
    assert result["estimated_complexity"] == "medium"


def test_invalid_json_raises_validation_error():
    with pytest.raises(ValidationError, match="not valid JSON"):
        validate_extraction_json("not json at all {{{", TAXONOMY, allow_proposed=False)


def test_missing_required_key_raises():
    raw = json.dumps({"likely_skills": [], "one_line_summary": "x"})  # missing estimated_complexity
    with pytest.raises(ValidationError, match="Missing required keys"):
        validate_extraction_json(raw, TAXONOMY, allow_proposed=False)


def test_out_of_taxonomy_skill_rejected_when_not_allowed():
    raw = json.dumps({
        "likely_skills": ["quantum computing"],
        "one_line_summary": "x",
        "estimated_complexity": "low",
    })
    with pytest.raises(ValidationError, match="not in the fixed taxonomy"):
        validate_extraction_json(raw, TAXONOMY, allow_proposed=False)


def test_out_of_taxonomy_skill_moved_to_proposed_when_allowed():
    raw = json.dumps({
        "likely_skills": ["testing/CI", "quantum computing"],
        "one_line_summary": "x",
        "estimated_complexity": "low",
    })
    result = validate_extraction_json(raw, TAXONOMY, allow_proposed=True)
    assert result["likely_skills"] == ["testing/CI"]
    assert result["proposed_skills"] == ["quantum computing"]


def test_invalid_complexity_value_rejected():
    raw = json.dumps({
        "likely_skills": [],
        "one_line_summary": "x",
        "estimated_complexity": "extremely high",  # not one of low/medium/high
    })
    with pytest.raises(ValidationError, match="estimated_complexity must be"):
        validate_extraction_json(raw, TAXONOMY, allow_proposed=False)


# --- check_roadmap_grounding ---------------------------------------------

def test_grounding_passes_when_no_files_invented():
    title = "Bug in auth_handler.py"
    body = "The auth_handler.py file has a bug in validate_token()."
    roadmap = "Start by reading auth_handler.py and tracing validate_token()."
    warnings = check_roadmap_grounding(roadmap, title, body)
    assert warnings == []


def test_grounding_flags_invented_filename():
    title = "Bug in auth_handler.py"
    body = "The auth_handler.py file has a bug."
    roadmap = "Also check totally_made_up_file.py which is unrelated to the issue."
    warnings = check_roadmap_grounding(roadmap, title, body)
    assert len(warnings) == 1
    assert "totally_made_up_file.py" in warnings[0]


def test_grounding_ignores_files_that_do_appear_in_source():
    title = "Bug"
    body = "See utils/helpers.py for context."
    roadmap = "Modify utils/helpers.py as described."
    warnings = check_roadmap_grounding(roadmap, title, body)
    assert warnings == []


# --- _strip_markdown_fences ------------------------------------------------

def test_strip_markdown_fences_removes_json_fence():
    text = "```json\n{\"a\": 1}\n```"
    assert _strip_markdown_fences(text) == '{"a": 1}'


def test_strip_markdown_fences_noop_on_plain_text():
    text = '{"a": 1}'
    assert _strip_markdown_fences(text) == '{"a": 1}'