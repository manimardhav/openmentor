"""
contributor_profile.py — a simple form where a contributor selects their
known skills from the taxonomy, plus save/load so a profile persists across
runs (useful once this needs to plug into the actual interface/ UI later —
that team can call get_saved_profile() the same way this script does).
"""

import json
import os
from skill_matching import SKILL_TAXONOMY

PROFILES_PATH = "contributor_profiles.json"


def prompt_contributor_skills(taxonomy: dict = None, input_source: str = None) -> set:
    """
    Presents the taxonomy as a numbered list and returns the set of skills
    the contributor selects.

    input_source: pass a comma-separated string directly (e.g. "1,3,5") to
    skip the interactive input() call — used for testing/scripting. Leave
    None for normal interactive use in a terminal.
    """
    taxonomy = taxonomy or SKILL_TAXONOMY
    skills_list = list(taxonomy.keys())

    print("\nSelect the skills you already know:")
    for i, skill in enumerate(skills_list, start=1):
        print(f"  {i}. {skill}")

    if input_source is None:
        raw = input("\nEnter the numbers of your skills, comma-separated (e.g. 1,3,5): ")
    else:
        raw = input_source

    selected_indices = []
    for token in raw.split(","):
        token = token.strip()
        if token.isdigit():
            selected_indices.append(int(token))

    selected_skills = {
        skills_list[i - 1] for i in selected_indices
        if 1 <= i <= len(skills_list)
    }

    if not selected_skills:
        print("No valid skills selected — profile will be empty.")

    return selected_skills


def prompt_contributor_level(input_source: str = None) -> float:
    """
    Asks the contributor to self-rate experience on a 1(beginner)-5(expert)
    scale, mapped to [0, 1] for use in ranking.py's difficulty_fit().
    """
    print("\nHow would you rate your experience with this codebase/language?")
    print("  1. Complete beginner")
    print("  2. Some experience")
    print("  3. Comfortable")
    print("  4. Experienced")
    print("  5. Expert")

    if input_source is None:
        raw = input("Enter a number 1-5: ").strip()
    else:
        raw = input_source

    try:
        level = int(raw)
        level = max(1, min(5, level))
    except ValueError:
        level = 1  # default to beginner if input is invalid

    return (level - 1) / 4  # maps 1-5 to 0.0-1.0


def save_contributor_profile(contributor_name: str, skills: set, level: float = None, path: str = PROFILES_PATH) -> None:
    """Persists a contributor's skill profile (and optional experience level) to a local JSON file."""
    profiles = {}
    if os.path.exists(path):
        with open(path) as f:
            profiles = json.load(f)

    profiles[contributor_name] = {"skills": sorted(skills), "level": level}

    with open(path, "w") as f:
        json.dump(profiles, f, indent=2)


def load_contributor_profile(contributor_name: str, path: str = PROFILES_PATH) -> tuple:
    """Loads a previously saved contributor's (skills, level), or (empty set, None) if not found."""
    if not os.path.exists(path):
        return set(), None
    with open(path) as f:
        profiles = json.load(f)
    entry = profiles.get(contributor_name, {"skills": [], "level": None})
    return set(entry.get("skills", [])), entry.get("level")


def get_or_create_profile(contributor_name: str, path: str = PROFILES_PATH) -> tuple:
    """
    Convenience wrapper: loads an existing profile if one exists, otherwise
    runs the interactive form and saves the result for next time.
    Returns (skills: set, level: float).
    """
    existing_skills, existing_level = load_contributor_profile(contributor_name, path)
    if existing_skills:
        print(f"Loaded existing profile for '{contributor_name}': skills={existing_skills}, level={existing_level}")
        return existing_skills, existing_level

    skills = prompt_contributor_skills()
    level = prompt_contributor_level()
    save_contributor_profile(contributor_name, skills, level, path)
    print(f"Saved new profile for '{contributor_name}': skills={skills}, level={level}")
    return skills, level


if __name__ == "__main__":
    demo_skills = prompt_contributor_skills(input_source="1,6")
    demo_level = prompt_contributor_level(input_source="2")
    print(f"\nSelected skills: {demo_skills}, level: {demo_level}")

    save_contributor_profile("demo_user", demo_skills, demo_level)
    reloaded_skills, reloaded_level = load_contributor_profile("demo_user")
    print(f"Reloaded from disk: skills={reloaded_skills}, level={reloaded_level}")
    assert reloaded_skills == demo_skills and reloaded_level == demo_level, "Save/load round-trip failed!"
    print("Save/load round-trip verified correctly.")