"""
Roadmap generation — orchestration script.

This is the only file that wires the other pieces together:
  config      -> where is the data, what model, what settings
  groq_client -> how to actually call the LLM, with retries
  prompts     -> what to say to the LLM
  validators  -> how to check what the LLM said back

Flow:
  1. Load an issue (mock data today, real ranked_issues.json later —
     one line changes, nothing else does).
  2. Build the roadmap prompt.
  3. Call Groq.
  4. Run the grounding check.
  5. If the check raises concerns, send the specific problem back to the
     model and ask it to correct itself (retry-with-correction), instead
     of either silently accepting a possibly-hallucinated roadmap or
     failing outright.
  6. Print the final, checked roadmap.
"""

import json
import logging

from config import MOCK_RANKED_ISSUES_PATH, ROADMAP_TEMPERATURE
from groq_client import call_groq, LLMCallError
from prompts.roadmap import build_roadmap_prompt, build_correction_prompt
from validators import check_roadmap_grounding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_roadmap")

MAX_CORRECTION_ATTEMPTS = 2


def generate_roadmap_for_issue(issue: dict, contributor_skills: list[str]) -> dict:
    """
    Generates a grounding-checked roadmap for a single issue.

    Returns a dict with the roadmap text and any grounding warnings that
    remained even after correction attempts, so the caller (and later,
    the interface) can decide whether to show a warning to the user
    rather than silently hiding uncertainty.
    """
    messages = build_roadmap_prompt(issue, contributor_skills)

    try:
        roadmap_text = call_groq(messages, temperature=ROADMAP_TEMPERATURE)
    except LLMCallError as e:
        logger.error(f"Failed to generate roadmap for issue {issue.get('issue_id')}: {e}")
        raise

    warnings = check_roadmap_grounding(roadmap_text, issue["title"], issue["body"])

    attempt = 0
    while warnings and attempt < MAX_CORRECTION_ATTEMPTS:
        attempt += 1
        logger.warning(
            f"Grounding check failed (attempt {attempt}/{MAX_CORRECTION_ATTEMPTS}): "
            f"{warnings}. Requesting correction from the model."
        )
        correction_messages = build_correction_prompt(
            messages, roadmap_text, "; ".join(warnings)
        )
        roadmap_text = call_groq(correction_messages, temperature=ROADMAP_TEMPERATURE)
        warnings = check_roadmap_grounding(roadmap_text, issue["title"], issue["body"])

    return {
        "issue_id": issue.get("issue_id"),
        "roadmap": roadmap_text,
        "remaining_warnings": warnings,  # empty list if fully resolved
        "correction_attempts_used": attempt,
    }


def main():
    with open(MOCK_RANKED_ISSUES_PATH) as f:
        issues = json.load(f)

    contributor_skills = ["testing", "REST API"]
    issue = issues[0]

    result = generate_roadmap_for_issue(issue, contributor_skills)

    print(f"Issue: {issue['title']}\n")
    print("Generated roadmap:\n")
    print(result["roadmap"])

    if result["remaining_warnings"]:
        print("\n⚠ Unresolved grounding warnings (review before showing to a user):")
        for w in result["remaining_warnings"]:
            print(f"  - {w}")
    else:
        print(f"\n✓ Grounding check passed (correction attempts used: {result['correction_attempts_used']})")


if __name__ == "__main__":
    main()