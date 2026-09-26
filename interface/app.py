"""
OpenMentor — demo interface (Week 3 deliverable: roadmap + explanation
per issue, now wired in).

v2 changes from the Week 2 placeholder:
1. Skill picker now reads from scoring.skill_matching.SKILL_TAXONOMY (the
   taxonomy actually used to compute skill_match_score / matched_skills),
   not shared/skill_taxonomy.json (a different, generic list the scoring
   code never sees). Previously the sidebar and the ranking used two
   different vocabularies, so "overlaps your skills" was structurally
   almost-always empty. See run_pipeline.py's module docstring for the
   full writeup — this is the interface-side half of that fix; the
   LLM-extraction-side half (prompts/extraction.py still reads the old
   taxonomy) is a separate, team-level decision, not changed here.
2. Per-issue templated explanation, built directly from top_features
   (no LLM call, so it's guaranteed grounded — see docx Week 3 "Added"
   item: log which features are referenced in each explanation).
3. Per-issue LLM roadmap, generated on demand (button, not on every
   rerun — a Groq call per issue on every page load would be slow and
   wasteful) via llm/generate_roadmap.py, cached in session_state so
   re-render doesn't re-call the model.
4. Explanation + roadmap generation events are appended to
   data/explanation_log.jsonl so explanation quality can later be
   checked against the features that actually drove the ranking
   (the Week 3 "Added" item), and so Week 4's survey can be cross-
   referenced against what each tester actually saw.
5. "Browse by skill" tab: shows every issue tagged under a chosen skill,
   independent of the contributor's own profile — so someone can explore
   what kinds of work exist in an area before committing to it, not just
   see the (possibly short) list that happens to overlap their profile.

Data source:
- Uses data/ranked_issues.json if Person A has produced it.
- Falls back to data/mock_ranked_issues.json otherwise, so this runs
  standalone today.

Ranking shown here is still placeholder in one sense: issues are sorted
by final_rank_score already computed upstream by Person A's scoring
pipeline. This screen does no re-scoring against the live skill
selection — that would duplicate Person A's job and drift out of sync
with it. It DOES use the live skill selection for the "overlap" badge
and for what's fed into the roadmap prompt.
"""

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_DIR = PROJECT_ROOT / "llm"
SCORING_DIR = PROJECT_ROOT / "scoring"
sys.path.insert(0, str(LLM_DIR))
sys.path.insert(0, str(SCORING_DIR))

from config import REAL_RANKED_ISSUES_PATH, MOCK_RANKED_ISSUES_PATH  # noqa: E402
from skill_matching import SKILL_TAXONOMY, match_score  # noqa: E402
from ranking import difficulty_fit  # noqa: E402
from generate_roadmap import generate_roadmap_for_issue  # noqa: E402
from groq_client import LLMCallError  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explanations import explain_score, log_explanation_event  # noqa: E402


def load_issues() -> tuple[list[dict], str]:
    path = REAL_RANKED_ISSUES_PATH if REAL_RANKED_ISSUES_PATH.exists() else MOCK_RANKED_ISSUES_PATH
    with open(path) as f:
        issues = json.load(f)
    return issues, path.name


def render_issue_card(issue: dict, contributor_skills: list[str], context: str) -> None:
    """Renders one issue card: score line, overlap badge, explanation, and
    on-demand roadmap. Shared by both the 'Recommended for you' and
    'Browse by skill' views so they never drift apart.

    `context` must be a string that's unique to WHERE this card is being
    drawn from (e.g. "recommended" or "browse::3D plotting") — Streamlit
    runs the code for every tab on every rerun (it only hides the inactive
    tab visually), so if the same issue appears in two tabs in the same
    run, its "Generate roadmap" button would get the same widget key twice
    and crash with StreamlitDuplicateElementKey. The cache key used for
    st.session_state (roadmap_key, below) deliberately does NOT include
    context, so a roadmap generated from one tab is still shown, cached,
    in the other — only the on-screen widget's key needs to be unique."""
    issue_id = issue.get("issue_id", "")
    overlap = set(issue.get("matched_skills", [])) & set(contributor_skills)

    tag = issue.get("confidence_tag", "")
    tag_class = "pill-high" if tag == "high-confidence" else "pill-exploratory"

    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{issue['title']}**")
            st.caption(issue.get("repo_name", ""))
        with col2:
            st.markdown(f'<span class="pill {tag_class}">{tag}</span>', unsafe_allow_html=True)

        st.write(issue["body"])

        st.markdown(
            f'<div class="score-line">'
            f'<span class="score-num">{issue.get("final_rank_score", 0):.2f}</span> rank score &nbsp;·&nbsp; '
            f'difficulty {issue.get("difficulty_score", 0):.2f} &nbsp;·&nbsp; '
            f'skill match {issue.get("skill_match_score", 0):.2f}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if issue.get("matched_skills"):
            chips = "".join(f'<span class="chip">{s}</span>' for s in sorted(issue["matched_skills"]))
            st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)

        if contributor_skills:
            if overlap:
                overlap_chips = "".join(f'<span class="chip chip-match">{s}</span>' for s in sorted(overlap))
                st.markdown(
                    f'<div class="overlap-line">Overlaps your skills: {overlap_chips}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="overlap-line overlap-none">No direct overlap with your selected skills.</div>',
                    unsafe_allow_html=True,
                )

        explanation, features_used = explain_score(issue)
        st.caption(explanation)
        log_explanation_event(issue_id, contributor_skills, features_used, event="explanation_shown")

        roadmap_key = f"roadmap::{issue_id}::{','.join(sorted(contributor_skills))}"
        if roadmap_key in st.session_state:
            st.markdown('<div class="roadmap-label">Roadmap</div>', unsafe_allow_html=True)
            st.write(st.session_state[roadmap_key]["roadmap"])
            if st.session_state[roadmap_key]["remaining_warnings"]:
                st.caption(
                    "⚠ Unresolved grounding warnings: "
                    + "; ".join(st.session_state[roadmap_key]["remaining_warnings"])
                )
        else:
            if st.button("Generate roadmap", key=f"btn::{context}::{roadmap_key}"):
                with st.spinner("Generating roadmap..."):
                    try:
                        result = generate_roadmap_for_issue(issue, contributor_skills)
                        st.session_state[roadmap_key] = result
                        log_explanation_event(
                            issue_id, contributor_skills,
                            features_used, event="roadmap_generated",
                        )
                        st.rerun()
                    except LLMCallError as e:
                        st.error(f"Roadmap generation failed: {e}")


def live_rescore(issue: dict, contributor_skills: list[str], experience_level: float) -> dict:
    """Recomputes skill_match_score and final_rank_score LIVE against this
    specific viewer's selected skills and experience — this is the fix for
    'Recommended for you looked the same for everyone': the file on disk
    only ever stored one hardcoded demo profile's scores. difficulty_score
    itself is left alone (it's a property of the issue, not the viewer —
    recomputing it would mean reloading the trained model on every
    Streamlit rerun, which is Person A's job, not this screen's).
    Returns a shallow-copied issue dict so the original in `issues` is
    untouched (needed since the same issue is re-scored differently in
    the Browse tab, where no profile may be selected)."""
    issue = dict(issue)
    if not contributor_skills:
        return issue  # nothing to re-score against; keep the stored values
    skill_score = match_score(set(contributor_skills), issue.get("body", ""), metric="cosine")
    fit = difficulty_fit(issue.get("difficulty_score", 0), experience_level)
    issue["skill_match_score"] = round(skill_score, 3)
    issue["final_rank_score"] = round(0.5 * fit + 0.5 * skill_score, 3)
    return issue


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background-color: #F6F7F9; color: #1B2430; }

h1 {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    border-bottom: 3px solid #1F7A4D;
    padding-bottom: 0.3rem;
    display: inline-block;
}

section[data-testid="stSidebar"] { background-color: #EDEFF3; border-right: 1px solid #D6DAE1; }

div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #D6DAE1 !important;
    border-radius: 6px !important;
    background-color: #FFFFFF;
    box-shadow: none !important;
}

.pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 4px;
    white-space: nowrap;
}
.pill-high { background-color: #E4F2E9; color: #1F7A4D; border: 1px solid #1F7A4D; }
.pill-exploratory { background-color: #FBEEDF; color: #C77D22; border: 1px solid #C77D22; }

.score-line {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #4B5563;
    margin: 0.4rem 0;
}
.score-num { color: #3457D5; font-weight: 500; }

.chip-row { margin: 0.3rem 0; }
.chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    padding: 2px 7px;
    margin: 2px 4px 2px 0;
    border-radius: 4px;
    background-color: #EEF1F6;
    color: #4B5563;
    border: 1px solid #D6DAE1;
}
.chip-match { background-color: #E4F2E9; color: #1F7A4D; border-color: #1F7A4D; }

.overlap-line { font-size: 0.85rem; margin: 0.3rem 0; color: #1F7A4D; }
.overlap-none { color: #8A6D3B; }

.roadmap-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    color: #3457D5;
    margin-top: 0.5rem;
    border-left: 3px solid #3457D5;
    padding-left: 8px;
}

.stButton button {
    background-color: #3457D5;
    color: white;
    border: none;
    border-radius: 5px;
}
.stButton button:hover { background-color: #2A45AD; color: white; }
</style>
"""


def main():
    st.set_page_config(page_title="OpenMentor", layout="centered")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("OpenMentor")
    st.caption("Find a good first issue matched to your skills.")

    core_skills = sorted(SKILL_TAXONOMY.keys())

    with st.sidebar:
        st.header("Your skill profile")
        contributor_skills = st.multiselect(
            "Select the skills you have",
            options=core_skills,
            default=[],
            help="Drawn from the same taxonomy the scoring pipeline uses, "
                 "so overlaps below are real matches, not coincidences.",
        )
        experience_level = st.slider(
            "Your experience level (0 = new contributor, 1 = experienced)",
            min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        )
        st.caption(
            "Recommended-for-you filters down to issues that overlap your "
            "selected skills, then ranks them live against your skills and "
            "experience level above."
        )

    issues, source_name = load_issues()
    st.caption(f"Data source: `{source_name}`")

    ranked = sorted(issues, key=lambda i: i.get("final_rank_score", 0), reverse=True)

    tab_recommended, tab_browse = st.tabs(["Recommended for you", "Browse by skill"])

    with tab_recommended:
        if not contributor_skills:
            st.info("Select at least one skill in the sidebar to see recommendations.")
        else:
            live_ranked = sorted(
                (live_rescore(i, contributor_skills, experience_level) for i in issues),
                key=lambda i: i["final_rank_score"], reverse=True,
            )
            overlapping = [
                i for i in live_ranked
                if set(i.get("matched_skills", [])) & set(contributor_skills)
            ]
            show_all = st.checkbox(
                "Show all issues ranked by fit, including ones with no direct skill overlap",
                value=False,
                help="Off by default so picking skills actually narrows the list down, "
                     "instead of just reordering all 200.",
            )
            shown = live_ranked if show_all else overlapping

            if not overlapping and not show_all:
                st.warning(
                    f"None of the 200 issues have a skill tag matching your selection "
                    f"({', '.join(contributor_skills)}). Try a broader or different skill, "
                    f"or check 'Show all issues' above."
                )
            else:
                st.subheader(f"Ranked issues ({len(shown)})")
                for issue in shown:
                    render_issue_card(issue, contributor_skills, context="recommended")

    with tab_browse:
        st.caption(
            "Every skill tag, independent of your selected profile above — "
            "browse what kinds of issues exist in an area, e.g. to scope out "
            "a skill before adding it to your profile. Sorted by the "
            "issue's own (non-personalized) difficulty/skill-match balance."
        )
        skill_counts = {}
        for skill in core_skills:
            skill_counts[skill] = sum(1 for i in issues if skill in i.get("matched_skills", []))
        untagged_count = sum(1 for i in issues if not i.get("matched_skills"))

        skill_options = [f"{s} ({skill_counts[s]})" for s in core_skills if skill_counts[s] > 0]
        chosen = st.selectbox("Choose a skill to browse", options=skill_options)
        chosen_skill = chosen.rsplit(" (", 1)[0] if chosen else None

        if chosen_skill:
            matching = [i for i in ranked if chosen_skill in i.get("matched_skills", [])]
            st.subheader(f"{chosen_skill} ({len(matching)} issues)")
            for issue in matching:
                render_issue_card(issue, contributor_skills, context=f"browse::{chosen_skill}")

        if untagged_count:
            st.caption(f"{untagged_count} issues matched no skill tag and aren't shown in this view.")


if __name__ == "__main__":
    main()