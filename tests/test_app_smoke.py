"""
Smoke tests for interface/app.py using Streamlit's own AppTest framework —
simulates real widget interaction (picking skills, clicking buttons)
without needing a browser. This is the test class that would have caught
both bugs found during manual testing:
1. Selecting skills not actually narrowing the recommended list down.
2. StreamlitDuplicateElementKey crash when the same issue's "Generate
   roadmap" button was drawn in two tabs in the same script run.

Requires: pip install groq (only needed so llm/generate_roadmap.py's
import chain resolves — no real network call is made by these tests).

Run with: pytest tests/test_app_smoke.py -v
"""

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parent.parent / "interface" / "app.py"

SAMPLE_SKILLS = ["3D plotting", "testing/CI", "documentation", "rendering/backends", "plotting/charts"]


@pytest.fixture
def app():
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    return at


def test_app_loads_without_exception(app):
    assert not app.exception


def test_selecting_skills_does_not_crash(app):
    app.sidebar.multiselect[0].set_value(SAMPLE_SKILLS).run()
    assert not app.exception


def test_selecting_skills_actually_narrows_the_list(app):
    """Regression test for the 'picking 5 skills still shows all 200'
    report — Recommended-for-you must show FEWER issues than the total
    once skills are selected, not just reorder the same 200."""
    total_issues = len(app.sidebar.multiselect[0].options)  # sanity: taxonomy is non-empty
    assert total_issues > 0

    app.sidebar.multiselect[0].set_value(SAMPLE_SKILLS).run()
    subheaders = [h.value for h in app.subheader]
    recommended_header = next(h for h in subheaders if h.startswith("Ranked issues"))
    shown_count = int(recommended_header.split("(")[1].split(")")[0])
    assert shown_count < 200, (
        f"Expected picking {len(SAMPLE_SKILLS)} skills to filter the 200-issue "
        f"set down, but {shown_count} were shown — filtering isn't working."
    )


def test_clicking_generate_roadmap_does_not_crash_with_duplicate_key(app):
    """Regression test for StreamlitDuplicateElementKey: clicking a roadmap
    button (which can legitimately appear in both tabs' render passes for
    the same issue) must not crash the whole app."""
    app.sidebar.multiselect[0].set_value(SAMPLE_SKILLS).run()
    assert len(app.button) > 0, "No 'Generate roadmap' buttons rendered — check test data"
    app.button[0].click().run()
    assert not app.exception


def test_no_skills_selected_shows_prompt_not_a_crash(app):
    app.sidebar.multiselect[0].set_value([]).run()
    assert not app.exception