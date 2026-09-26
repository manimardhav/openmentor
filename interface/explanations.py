"""
Pure, testable logic split out of interface/app.py: building a templated
explanation from an issue's top_features, and logging which features were
referenced. Kept free of any Streamlit import so it can be unit tested
without a Streamlit runtime (see tests/test_explanations.py).
"""

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPLANATION_LOG_PATH = PROJECT_ROOT / "data" / "explanation_log.jsonl"


def explain_score(issue: dict) -> tuple[str, list[str]]:
    """Builds a templated, guaranteed-grounded sentence directly from
    top_features — no LLM involved, so nothing here can be hallucinated.
    Returns (sentence, list of feature names actually referenced), so the
    caller can log exactly what drove the explanation (docx Week 3 item)."""
    tf = issue.get("top_features", {})
    parts = []
    referenced = []

    centrality = tf.get("centrality")
    if centrality is not None:
        referenced.append("centrality")
        if centrality >= 0.5:
            parts.append(f"it touches a highly central file (centrality {centrality:.2f})")
        elif centrality > 0:
            parts.append(f"it touches a moderately central file (centrality {centrality:.2f})")

    if tf.get("first_contribution_label"):
        referenced.append("first_contribution_label")
        parts.append("it's labeled as friendly to a first contribution")

    referenced_files = tf.get("referenced_files")
    if referenced_files is not None:
        referenced.append("referenced_files")
        if referenced_files >= 2:
            parts.append(f"it references {referenced_files:.0f} files, giving more context to work from")

    issue_age = tf.get("issue_age_days")
    if issue_age is not None:
        referenced.append("issue_age_days")
        parts.append(f"it's been open {issue_age:.0f} days")

    if not parts:
        parts.append("of its overall difficulty and skill-match balance")

    sentence = (
        f"Difficulty {issue.get('difficulty_score', 0):.2f} and skill match "
        f"{issue.get('skill_match_score', 0):.2f} combine into this rank because "
        + "; ".join(parts) + "."
    )
    return sentence, referenced


def log_explanation_event(
    issue_id: str,
    contributor_skills: list[str],
    features_referenced: list[str],
    event: str,
    log_path: Path = EXPLANATION_LOG_PATH,
) -> None:
    """Appends one JSON line per explanation/roadmap shown, so Round 1 vs
    Round 2 explanation quality (docx Week 5 item) can be checked against
    what actually drove each ranking, not reconstructed from memory."""
    record = {
        "timestamp": time.time(),
        "issue_id": issue_id,
        "contributor_skills": contributor_skills,
        "features_referenced": features_referenced,
        "event": event,
    }
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass  # logging must never break the demo