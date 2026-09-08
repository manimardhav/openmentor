"""
skill_matching.py — Week 4: skill-tagging pipeline + contributor-issue matching.

Two similarity metrics are implemented (Jaccard, cosine) so you can compare
them directly — see the guidance to default to cosine once you move beyond
binary skill vectors (e.g. weighting by keyword frequency).
"""

import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Skill taxonomy + keyword mapping
# ---------------------------------------------------------------------------

# Start small (Week 1 guidance) and expand as you validate against real
# issues. Each tag maps to a list of keywords/regex fragments that, if found
# in the issue text, suggest that skill is required.
#
# Tailored to httpie/cli (pilot repo, chosen Week 1): a Python CLI HTTP
# client. Tags reflect httpie's actual subsystems rather than a generic
# web-app taxonomy — validate/expand these against real issue text in Week 4.
SKILL_TAXONOMY = {
    "HTTP/networking": ["http request", "response", "status code", "headers?", "ssl", "tls", "redirect", "websocket"],
    "CLI/argparse": ["command line", "cli", "argument", "flag", "argparse", "option"],
    "output formatting": ["formatting", "pretty print", "colou?r output", "json output"],
    "auth": ["authentication", "auth", "login", "token", "oauth", "credentials?", "session"],
    "security": ["security", "vulnerability", "injection", "plaintext", "escape sequence", "exploit"],
    "plugins/extensibility": ["plugin", "extension", "hook", "middleware"],
    "testing": ["test", "pytest", "unit test", "assert", "mock", "fixture"],
    "config/environment": ["config", "environment variable", "\\.httpie", "settings", "session file"],
    "docs": ["readme", "documentation", "docstring", "typo", "changelog", "doc coverage"],
}


def tag_issue_skills(issue_text: str, taxonomy: dict = None) -> set:
    """
    Keyword/regex matching (Week 4 starting point). Returns the set of skill
    tags whose keywords appear in the issue text.

    Known limitation (flag this in your writeup): substring matches can
    false-positive (e.g. "test" inside "testament"). Word-boundary regex
    below reduces but doesn't eliminate this — a curated synonym dictionary
    per tag is the recommended next step if false positives show up in
    Round 1 feedback.
    """
    taxonomy = taxonomy or SKILL_TAXONOMY
    text_lower = (issue_text or "").lower()
    matched = set()
    for skill, keywords in taxonomy.items():
        for kw in keywords:
            # NOTE: keywords are treated as regex fragments, not escaped —
            # the taxonomy intentionally uses patterns like "colou?r" and
            # "headers?". Word boundaries are added around the whole
            # fragment; don't add \b inside individual keyword strings.
            if re.search(r"\b" + kw + r"\b", text_lower):
                matched.add(skill)
                break
    return matched


# ---------------------------------------------------------------------------
# 2. Vectorize skills for similarity computation
# ---------------------------------------------------------------------------

def skills_to_vector(skill_set: set, taxonomy: dict = None) -> np.ndarray:
    """Binary vector over the full taxonomy, ordered consistently."""
    taxonomy = taxonomy or SKILL_TAXONOMY
    all_skills = list(taxonomy.keys())
    return np.array([1.0 if s in skill_set else 0.0 for s in all_skills])


# ---------------------------------------------------------------------------
# 3. Similarity metrics
# ---------------------------------------------------------------------------

def jaccard_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    set_a, set_b = set(np.where(vec_a > 0)[0]), set(np.where(vec_b > 0)[0])
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a, norm_b = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# 4. Contributor <-> issue matching
# ---------------------------------------------------------------------------

def match_score(contributor_skills: set, issue_text: str, taxonomy: dict = None, metric: str = "cosine") -> float:
    """
    Returns a 0-1 match score between a contributor's stated skills and an
    issue's inferred required skills.
    """
    taxonomy = taxonomy or SKILL_TAXONOMY
    issue_skills = tag_issue_skills(issue_text, taxonomy)

    contributor_vec = skills_to_vector(contributor_skills, taxonomy)
    issue_vec = skills_to_vector(issue_skills, taxonomy)

    if metric == "jaccard":
        return jaccard_similarity(contributor_vec, issue_vec)
    return cosine_similarity(contributor_vec, issue_vec)


def rank_issues_for_contributor(contributor_skills: set, issues: list, taxonomy: dict = None, metric: str = "cosine") -> pd.DataFrame:
    """
    issues: list of dicts with keys 'id' and 'body'.
    Returns a DataFrame sorted by match_score, descending.
    """
    rows = []
    for issue in issues:
        score = match_score(contributor_skills, issue.get("body", ""), taxonomy, metric)
        rows.append({"issue_id": issue["id"], "match_score": score})
    return pd.DataFrame(rows).sort_values("match_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # httpie/cli-style toy issues
    contributor = {"testing", "HTTP/networking"}

    toy_issues = [
        {"id": 1, "body": "Add pytest unit tests covering redirect handling for HTTP requests"},
        {"id": 2, "body": "Support a new --plugin hook for custom output formatting"},
        {"id": 3, "body": "Fix argparse flag parsing when --auth token is passed with special characters"},
    ]

    ranked = rank_issues_for_contributor(contributor, toy_issues)
    print(ranked)

    print("\nSingle-pair check (Jaccard vs cosine):")
    print("Jaccard:", match_score(contributor, toy_issues[0]["body"], metric="jaccard"))
    print("Cosine:", match_score(contributor, toy_issues[0]["body"], metric="cosine"))
