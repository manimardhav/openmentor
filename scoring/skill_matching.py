"""
skill_matching.py — skill tagging + similarity matching.

Skill NAMES are loaded from shared/skill_taxonomy.json, the single team-wide source
of truth that the LLM extraction prompt also uses, so the two can no longer drift
apart. The KEYWORD patterns that detect each skill in issue text stay in this file.

Known limitation: the keywords were written for the original Python/data-science
repos. Run validate_taxonomy.py for per-repo hit rates; coverage is much lower on the
helm, sqlalchemy, vuejs and express repos.
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

SKILL_TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "shared" / "skill_taxonomy.json"

# Keyword patterns per skill. Treated as regex fragments (not escaped) — e.g. "qt5?"
# and "colou?r" are intentional patterns, not typos.
SKILL_KEYWORDS = {
    "rendering/backends": ["backend", "cairo", "agg", "qt5?", "wxpython", "webagg", "renderer", "retina", "macos"],
    "text/font rendering": ["font", "freetype", "ft2font", "text shaping", "arabic", "inkscape", "svg", "glyph"],
    "3D plotting": ["3d", "poly3dcollection", "mplot3d", "surface plot"],
    "plotting/charts": ["boxplot", "scatter", "\\bbar\\(", "tripcolor", "histogram", "subplot", "\\bplot\\("],
    "widgets/interactive": ["textbox", "widget", "interactive", "jupyter", "kernel", "event handler"],
    "animation": ["animation", "\\bgif\\b", "pillow", "frames", "funcanimation"],
    "packaging/build": ["\\bpip\\b", "wheel", "install", "conda", "dependency", "setup\\.py", "build system"],
    "dataframe/series ops": ["dataframe", "\\bseries\\b", "groupby", "\\brolling\\b", "dtype", "nan", "\\bna\\b", "pivot", "merge"],
    "data IO/serialization": ["read_html", "read_csv", "to_csv", "\\bparquet\\b", "\\bpickle\\b", "fsspec", "pyarrow", "\\barrow\\b", "to_timedelta"],
    "dataset loading/sharding": ["load_dataset", "iterabledataset", "\\bshard\\b", "\\bsplit\\b", "concatenate_datasets", "from_generator", "from_list", "push_to_hub", "hf hub"],
    "graph algorithms": ["shortest path", "\\bdigraph\\b", "multidigraph", "\\btriad\\b", "spanner", "louvain", "\\btraverse", "sparsifier", "rooted_product"],
    "graph IO/formats": ["graphml", "\\bgexf\\b", "\\bpydot\\b", "write_graphml", "read_graphml"],
    "performance/regression": ["performance", "regression", "degradation", "\\bslow\\b", "\\bspeed\\b"],
    "Cython/C extensions": ["cython", "\\.pyx\\b", "\\bpybind\\b", "c extension", "compiled extension"],
    "testing/CI": ["\\btest\\b", "pytest", "\\bci\\b", "nightly", "circleci", "github actions"],
    "documentation": ["\\bdoc\\b", "documentation", "docstring", "sphinx", "hyperlink", "changelog"],
    "legend/colormap": ["legend", "colorbar", "colormap", "facecolor", "gridline", "colou?r"],
}


def load_skill_taxonomy(path: Path = SKILL_TAXONOMY_PATH) -> list:
    """Loads the shared, team-wide skill list. Raises clearly if missing, rather than silently returning []."""
    if not path.exists():
        raise FileNotFoundError(
            f"Skill taxonomy not found at {path}. This file must exist and be shared across the whole team."
        )
    with open(path) as f:
        taxonomy = json.load(f)
    return taxonomy["core_skills"]


def _build_skill_taxonomy() -> dict:
    names = load_skill_taxonomy()
    missing_keywords = [n for n in names if n not in SKILL_KEYWORDS]
    if missing_keywords:
        # Don't crash — fall back to a bare word-match on the skill's own name so a
        # newly-added shared skill is at least matchable, but add real keywords.
        for name in missing_keywords:
            SKILL_KEYWORDS[name] = [re.escape(w) for w in name.replace("/", " ").split()]
        print(f"WARNING: no curated keywords for {missing_keywords} — "
              f"using a bare name-word fallback. Add real keywords to SKILL_KEYWORDS.")
    return {name: SKILL_KEYWORDS[name] for name in names}


SKILL_TAXONOMY = _build_skill_taxonomy()


def tag_issue_skills(issue_text: str, taxonomy: dict = None) -> set:
    taxonomy = taxonomy or SKILL_TAXONOMY
    text_lower = (issue_text or "").lower()
    matched = set()
    for skill, keywords in taxonomy.items():
        for kw in keywords:
            if re.search(r"\b" + kw + r"\b", text_lower):
                matched.add(skill)
                break
    return matched


def skills_to_vector(skill_set: set, taxonomy: dict = None) -> np.ndarray:
    taxonomy = taxonomy or SKILL_TAXONOMY
    all_skills = list(taxonomy.keys())
    return np.array([1.0 if s in skill_set else 0.0 for s in all_skills])


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


def match_score(contributor_skills: set, issue_text: str, taxonomy: dict = None, metric: str = "cosine") -> float:
    taxonomy = taxonomy or SKILL_TAXONOMY
    issue_skills = tag_issue_skills(issue_text, taxonomy)
    contributor_vec = skills_to_vector(contributor_skills, taxonomy)
    issue_vec = skills_to_vector(issue_skills, taxonomy)
    if metric == "jaccard":
        return jaccard_similarity(contributor_vec, issue_vec)
    return cosine_similarity(contributor_vec, issue_vec)


def rank_issues_for_contributor(contributor_skills: set, issues: list, taxonomy: dict = None, metric: str = "cosine") -> pd.DataFrame:
    rows = []
    for issue in issues:
        score = match_score(contributor_skills, issue.get("body", ""), taxonomy, metric)
        rows.append({"issue_id": issue["id"], "match_score": score})
    return pd.DataFrame(rows).sort_values("match_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print(f"Loaded {len(SKILL_TAXONOMY)} skills from shared/skill_taxonomy.json: {list(SKILL_TAXONOMY.keys())}\n")

    contributor_skills = {"testing/CI", "packaging/build"}
    toy_issues = [
        {"id": 1, "body": "pip install matplotlib fails on Windows for Python 3.15"},
        {"id": 2, "body": "Add pytest coverage for the nightly CI wheel build"},
        {"id": 3, "body": "TextBox widget raises AttributeError on ResizeEvent"},
    ]
    ranked = rank_issues_for_contributor(contributor_skills, toy_issues)
    print(ranked)