"""
Central configuration for the LLM module.

Why this file exists:
Hardcoding model names, file paths, and settings across multiple files
means changing one thing (e.g. switching models, moving a data file)
requires hunting through every script. This file is the single source
of truth — everything else imports from here.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
# BASE_DIR = the llm/ folder itself. We compute paths relative to this
# file's location, not the current working directory, so scripts work
# correctly no matter which folder you run them from.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
SHARED_DIR = PROJECT_ROOT / "shared"

MOCK_RANKED_ISSUES_PATH = DATA_DIR / "mock_ranked_issues.json"
REAL_RANKED_ISSUES_PATH = DATA_DIR / "ranked_issues.json"
SKILL_TAXONOMY_PATH = SHARED_DIR / "skill_taxonomy.json"

# --- LLM settings --------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Create a .env file in the project root "
        "with: GROQ_API_KEY=your_key_here"
    )

MODEL_NAME = "llama-3.3-70b-versatile"

# Lower temperature = more consistent, less creative. Roadmaps and
# extraction both need consistency, not creativity, so both stay low.
ROADMAP_TEMPERATURE = 0.4
EXTRACTION_TEMPERATURE = 0.2  # even lower: this task must be near-deterministic

# Retry/robustness settings (see groq_client.py)
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2  # doubles each retry: 2s, 4s, 8s

# --- Skill taxonomy --------------------------------------------------------
def load_skill_taxonomy() -> list[str]:
    """
    Loads the shared skill taxonomy from shared/skill_taxonomy.json.
    Raises a clear error if the file is missing or malformed, rather than
    silently returning an empty list (which would make every extraction
    call fail in a confusing way).
    """
    if not SKILL_TAXONOMY_PATH.exists():
        raise FileNotFoundError(
            f"Skill taxonomy not found at {SKILL_TAXONOMY_PATH}. "
            "This file must exist and be shared across the whole team."
        )
    with open(SKILL_TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    return taxonomy["core_skills"]


def allow_proposed_skills() -> bool:
    with open(SKILL_TAXONOMY_PATH) as f:
        taxonomy = json.load(f)
    return taxonomy.get("allow_llm_proposed_skills", False)