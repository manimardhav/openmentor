"""
Roadmap generation prompt template.

Design notes:
- No hardcoded skill list — matched_skills comes from whatever Member 1's
  scoring pipeline (or the extraction prompt) actually found for this
  issue, so this template works regardless of which repo or skill set
  is in play.
- The system prompt explicitly forbids inventing file names, function
  names, or details not present in the issue — this is the prompt-level
  half of hallucination prevention. The code-level half is
  validators.check_roadmap_grounding(), which checks the output
  afterward. Neither alone is sufficient; together they cover both the
  "ask nicely" and "verify after" layers.
"""

ROADMAP_SYSTEM_PROMPT = """You are an assistant that helps new open-source \
contributors get started on a specific GitHub issue. You write short, \
concrete, step-by-step roadmaps.

Rules you must follow:
1. Only reference file names, function names, or technical details that \
appear explicitly in the issue title or description given to you. Never \
invent or assume file names that were not mentioned.
2. Never give generic advice such as "read the documentation" or "look \
at the codebase" without saying specifically what to look for, based on \
the issue text.
3. If the issue description does not contain enough detail to write a \
specific step, say so explicitly in that step (e.g. "Locate the relevant \
handler for X — the issue does not specify the exact file") rather than \
inventing a plausible-sounding but unverified detail.
4. Keep each step to one sentence. Do not restate the issue description."""

ROADMAP_USER_PROMPT_TEMPLATE = """Issue title: {title}

Issue description:
{body}

Skills this issue likely requires: {matched_skills}
Contributor's known skills: {contributor_skills}

Write a roadmap of 3-5 concrete steps this contributor could take to \
start working on this issue."""


def build_roadmap_prompt(issue: dict, contributor_skills: list[str]) -> list[dict]:
    """
    issue: one item from ranked_issues.json (or mock data). Must contain
           at minimum 'title', 'body', and optionally 'matched_skills'.
    contributor_skills: list of skill strings the contributor selected.
    """
    matched_skills = issue.get("matched_skills", [])
    user_content = ROADMAP_USER_PROMPT_TEMPLATE.format(
        title=issue["title"],
        body=issue["body"],
        matched_skills=", ".join(matched_skills) if matched_skills else "not specified",
        contributor_skills=", ".join(contributor_skills) if contributor_skills else "not specified",
    )
    return [
        {"role": "system", "content": ROADMAP_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_correction_prompt(original_messages: list[dict], bad_output: str, error: str) -> list[dict]:
    """
    Builds a follow-up prompt asking the model to fix its own output
    after a validation failure (grounding warning, etc). This is the
    "retry with correction" pattern — instead of just re-asking blind
    and hoping for a better roll, we tell the model exactly what was
    wrong with its previous answer.
    """
    return original_messages + [
        {"role": "assistant", "content": bad_output},
        {
            "role": "user",
            "content": (
                f"That roadmap had a problem: {error}\n"
                "Please rewrite the roadmap, fixing this issue and "
                "following the original instructions exactly."
            ),
        },
    ]