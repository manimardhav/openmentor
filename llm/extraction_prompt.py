"""
Issue-text feature extraction prompt template.
"""

SKILL_TAXONOMY = [
    "REST API", "async/await", "database", "testing", "frontend-CSS",
    "authentication", "documentation", "CLI tooling", "build/CI",
]

EXTRACTION_SYSTEM_PROMPT = f"""You extract structured metadata from GitHub \
issue text. You only choose skills from this fixed list: {SKILL_TAXONOMY}. \
You always respond with valid JSON and nothing else — no explanation, no \
markdown formatting, no backticks."""

EXTRACTION_USER_PROMPT_TEMPLATE = """Issue title: {title}

Issue description:
{body}

Return JSON with exactly these keys:
- "likely_skills": a list of 1-4 skills from the fixed taxonomy
- "one_line_summary": a single sentence summarizing the issue
- "estimated_complexity": one of "low", "medium", "high\""""


def build_extraction_prompt(title, body):
    user_content = EXTRACTION_USER_PROMPT_TEMPLATE.format(title=title, body=body)
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]