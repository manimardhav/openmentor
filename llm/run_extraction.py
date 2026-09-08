"""
Issue-text feature extraction — orchestration script.

Same pattern as generate_roadmap.py: build prompt -> call Groq ->
validate -> retry with correction if validation fails. Here validation
is strict JSON-schema checking (validators.validate_extraction_json)
rather than the advisory grounding check used for roadmaps, because
extraction output is meant to be consumed programmatically by Member 1's
scoring pipeline — malformed JSON there is a hard failure, not a
judgment call.
"""

import logging

from config import load_skill_taxonomy, allow_proposed_skills, EXTRACTION_TEMPERATURE
from groq_client import call_groq, LLMCallError
from prompts.extraction import build_extraction_prompt
from validators import validate_extraction_json, ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_extraction")

MAX_CORRECTION_ATTEMPTS = 2


def extract_issue_features(title: str, body: str) -> dict:
    taxonomy = load_skill_taxonomy()
    allow_proposed = allow_proposed_skills()

    messages = build_extraction_prompt(title, body, taxonomy)

    attempt = 0
    last_error = None

    while attempt <= MAX_CORRECTION_ATTEMPTS:
        try:
            raw_output = call_groq(messages, temperature=EXTRACTION_TEMPERATURE)
        except LLMCallError as e:
            logger.error(f"Groq call failed during extraction: {e}")
            raise

        try:
            return validate_extraction_json(raw_output, taxonomy, allow_proposed)
        except ValidationError as e:
            last_error = e
            attempt += 1
            logger.warning(
                f"Extraction validation failed (attempt {attempt}/{MAX_CORRECTION_ATTEMPTS}): {e}"
            )
            messages = messages + [
                {"role": "assistant", "content": raw_output},
                {"role": "user", "content": f"That output was invalid: {e}. Please fix it and return only the corrected JSON."},
            ]

    raise ValidationError(
        f"Extraction failed validation after {MAX_CORRECTION_ATTEMPTS} correction attempts. "
        f"Last error: {last_error}"
    )


def main():
    title = "Fix broken pagination on the search results page"
    body = (
        "When navigating to page 2 of search results, the page resets to "
        "page 1 instead of showing the next set of results. This happens "
        "because the page query parameter is not being read correctly in "
        "SearchResults.js."
    )

    result = extract_issue_features(title, body)
    print("Extraction result:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()