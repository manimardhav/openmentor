"""
OpenMentor — minimal demo interface (Week 2 deliverable).

Scope, deliberately: input a contributor's skill profile, output a
placeholder ranked list. This is NOT the full demo (that's Week 3-4 —
roadmap + explanation per issue, once Person A's real ranked_issues.json
and the LLM roadmap step are wired in here).

Data source:
- Uses data/ranked_issues.json if Person A has produced it.
- Falls back to data/mock_ranked_issues.json otherwise, so this runs
  standalone today.

Ranking shown here is placeholder: issues are just sorted by the
final_rank_score already present in the data (computed upstream by
Person A's scoring pipeline). This screen does no scoring itself —
that would duplicate Person A's job and drift out of sync with it.
"""

import json
import sys
from pathlib import Path

import streamlit as st

# Reuse the same paths/skill list the LLM module already uses, so this
# interface and generate_roadmap.py never disagree about where data lives.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm"))
from config import REAL_RANKED_ISSUES_PATH, MOCK_RANKED_ISSUES_PATH, load_skill_taxonomy  # noqa: E402


def load_issues() -> list[dict]:
    path = REAL_RANKED_ISSUES_PATH if REAL_RANKED_ISSUES_PATH.exists() else MOCK_RANKED_ISSUES_PATH
    with open(path) as f:
        issues = json.load(f)
    return issues, path.name


def main():
    st.set_page_config(page_title="OpenMentor", layout="centered")
    st.title("OpenMentor")
    st.caption("Find a good first issue matched to your skills.")

    core_skills = load_skill_taxonomy()

    with st.sidebar:
        st.header("Your skill profile")
        contributor_skills = st.multiselect(
            "Select the skills you have",
            options=core_skills,
            default=[],
        )
        st.caption(
            "Ranking below is placeholder: it sorts by the score already "
            "computed upstream, it does not re-score against your selection yet."
        )

    issues, source_name = load_issues()
    st.caption(f"Data source: `{source_name}`")

    if not contributor_skills:
        st.info("Select at least one skill in the sidebar to see recommendations.")
        return

    ranked = sorted(issues, key=lambda i: i.get("final_rank_score", 0), reverse=True)

    st.subheader(f"Ranked issues ({len(ranked)})")
    for issue in ranked:
        overlap = set(issue.get("matched_skills", [])) & set(contributor_skills)
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{issue['title']}**")
                st.caption(issue.get("repo_name", ""))
            with col2:
                tag = issue.get("confidence_tag", "")
                st.markdown(f"`{tag}`")

            st.write(issue["body"])

            st.write(
                f"Rank score: **{issue.get('final_rank_score', 0):.2f}**  ·  "
                f"Difficulty: {issue.get('difficulty_score', 0):.2f}  ·  "
                f"Skill match: {issue.get('skill_match_score', 0):.2f}"
            )

            if overlap:
                st.success(f"Overlaps your skills: {', '.join(sorted(overlap))}")
            else:
                st.warning("No direct overlap with your selected skills yet.")

            st.caption("Roadmap and explanation generation: coming in Week 3.")


if __name__ == "__main__":
    main()