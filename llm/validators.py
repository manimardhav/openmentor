"""
Validation layer: catches malformed output and hallucination.

Why this file exists:
An LLM system prompt is an instruction, not a guarantee. The model can
still return malformed JSON, invent skills outside the taxonomy, or
reference files/functions that were never mentioned in the issue text.
This file checks the model's output in code, rather than trusting the
system prompt alone. If a check fails, the caller (generate_roadmap.py)
retries with the specific error fed back to the model — this is the
"retry with correction" pattern used in production LLM systems.
"""

import json
import re


class ValidationError(Exception):
    """Raised when model output fails a validation check. The message
    is written to be re-sent to the model so it can correct itself."""
    pass


# --- Extraction output validation --------------------------------------

def validate_extraction_json(raw_text: str, taxonomy: list[str], allow_proposed: bool) -> dict:
    """
    Validates the JSON returned by the extraction prompt.
    Checks:
      1. It is valid JSON (models sometimes wrap output in markdown
         fences or add commentary despite instructions not to).
      2. It has exactly the required keys.
      3. likely_skills only contains taxonomy skills, UNLESS
         allow_proposed is True and skills are listed separately under
         'proposed_skills' rather than silently mixed into likely_skills.
      4. estimated_complexity is one of the three allowed values.
    """
    cleaned = _strip_markdown_fences(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValidationError(
            f"Output was not valid JSON: {e}. "
            "Return ONLY a JSON object, no markdown fences, no commentary."
        )

    required_keys = {"likely_skills", "one_line_summary", "estimated_complexity"}
    missing = required_keys - data.keys()
    if missing:
        raise ValidationError(
            f"Missing required keys: {missing}. "
            f"The JSON must contain exactly: {required_keys}"
        )

    invalid_skills = [s for s in data["likely_skills"] if s not in taxonomy]
    if invalid_skills and not allow_proposed:
        raise ValidationError(
            f"These skills are not in the fixed taxonomy: {invalid_skills}. "
            f"Only use skills from: {taxonomy}"
        )
    if invalid_skills and allow_proposed:
        # Move out-of-taxonomy skills into a separate field instead of
        # silently discarding them — this is the fix for the "what if
        # the repo needs a skill we didn't think of" problem.
        data["proposed_skills"] = invalid_skills
        data["likely_skills"] = [s for s in data["likely_skills"] if s in taxonomy]

    if data["estimated_complexity"] not in ("low", "medium", "high"):
        raise ValidationError(
            f"estimated_complexity must be 'low', 'medium', or 'high', "
            f"got: {data['estimated_complexity']!r}"
        )

    return data


# --- Roadmap grounding check --------------------------------------------

def check_roadmap_grounding(roadmap_text: str, issue_title: str, issue_body: str) -> list[str]:
    """
    Advisory grounding check for generated roadmaps.

    This does NOT attempt full semantic verification (that would need
    another LLM call and isn't reliable either). Instead it does a
    cheap, explainable check: extract filename-like tokens (e.g.
    'SearchResults.js', 'auth_handler.py') from the roadmap, and flag
    any that do not appear anywhere in the original issue text. If the
    model invents a specific file name that was never mentioned, that's
    a concrete, checkable sign of hallucination.

    Returns a list of warning strings (empty list = no concerns found).
    This is advisory, not a hard rejection — file names are the clearest
    hallucination signal, but their absence doesn't guarantee the whole
    roadmap is grounded, and a false positive here shouldn't block a
    genuinely good roadmap.
    """
    filename_pattern = r"\b[\w\-/]+\.(?:py|js|jsx|ts|tsx|java|go|rb|json|yaml|yml|md|css|html)\b"

    roadmap_files = set(re.findall(filename_pattern, roadmap_text))
    source_text = f"{issue_title}\n{issue_body}"
    source_files = set(re.findall(filename_pattern, source_text))

    invented_files = roadmap_files - source_files

    warnings = []
    if invented_files:
        warnings.append(
            f"Roadmap references file(s) not found in the original issue text: "
            f"{sorted(invented_files)}. Verify these are not hallucinated."
        )
    return warnings


def _strip_markdown_fences(text: str) -> str:
    """Removes ```json ... ``` or ``` ... ``` wrapping some models add
    despite being told not to, so json.loads() doesn't choke on it."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # drop closing fence
        text = "\n".join(lines)
    return text.strip()