"""
Roadmap generation prompt template.
"""

ROADMAP_SYSTEM_PROMPT = """You are an assistant that helps new open-source \
contributors get started on a specific GitHub issue. You write short, \
concrete, step-by-step roadmaps. You never give generic advice like "read \
the documentation" or "look at the codebase" without saying exactly what \
to look for. You only use information given to you about the issue and the \
contributor's skills — do not invent file names, function names, or issue \
details that were not provided."""

ROADMAP_USER_PROMPT_TEMPLATE = """Issue title: {title}

Issue description:
{body}

Skills this issue likely requires: {matched_skills}
Contributor's known skills: {contributor_skills}

Write a roadmap of 3-5 concrete steps this contributor could take to start \
working on this issue. Be specific to the issue description above. Keep \
each step to one sentence. Do not restate the issue description."""


def build_roadmap_prompt(issue, contributor_skills):
    user_content = ROADMAP_USER_PROMPT_TEMPLATE.format(
        title=issue["title"],
        body=issue["body"],
        matched_skills=", ".join(issue.get("matched_skills", [])),
        contributor_skills=", ".join(contributor_skills) if contributor_skills else "not specified",
    )
    return [
        {"role": "system", "content": ROADMAP_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]