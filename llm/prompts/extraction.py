"""
Issue-text feature extraction prompt template.

Design notes:
- The taxonomy is passed in as an argument (loaded from
  shared/skill_taxonomy.json via config.py), never hardcoded here. This
  file has zero knowledge of what the actual skills are — it just knows
  how to build a prompt around whatever taxonomy it's given.
- The model is explicitly told it MAY propose a skill outside the fixed
  list, tagged separately, rather than being forced to either mis-tag it
  as an existing skill or silently omit it. validators.py then keeps
  proposed skills separate from the trusted taxonomy list.
"""

EXTRACTION_SYSTEM_PROMPT_TEMPLATE = """You extract structured metadata \
from GitHub issue text. You always respond with valid JSON and nothing \
else — no explanation, no markdown formatting, no backticks, no text \
before or after the JSON object.

The team's current skill taxonomy is: {taxonomy}

Prefer skills from this taxonomy when they fit. If the issue clearly \
requires a skill that is NOT in this list, you may include it under a \
separate "proposed_skills" field instead of forcing a poor fit into \
"likely_skills"."""

EXTRACTION_USER_PROMPT_TEMPLATE = """Issue title: {title}

Issue description:
{body}

Return JSON with exactly these keys:
- "likely_skills": a list of 1-4 skills, preferring the taxonomy above
- "proposed_skills": a list of skills not in the taxonomy that this \
issue requires, if any (empty list if none)
- "one_line_summary": a single sentence summarizing the issue
- "estimated_complexity": one of "low", "medium", "high\""""


def build_extraction_prompt(title: str, body: str, taxonomy: list[str]) -> list[dict]:
    system_content = EXTRACTION_SYSTEM_PROMPT_TEMPLATE.format(taxonomy=taxonomy)
    user_content = EXTRACTION_USER_PROMPT_TEMPLATE.format(title=title, body=body)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]