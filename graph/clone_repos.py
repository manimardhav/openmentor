"""
clone_repos.py — Week 1: test GitHub API access, then clone the chosen repos.

Setup before running:
1. Create a GitHub personal access token:
   github.com -> Settings -> Developer settings -> Personal access tokens
   -> Generate new token (classic) -> tick "repo" scope -> Generate.
2. In the project root, create a file named .env (same folder as
   requirements.txt) containing:
       GITHUB_TOKEN=your_token_here
       GROQ_API_KEY=your_groq_key_here   (Member 2 already has this line)
3. Run:  pip install -r ../requirements.txt   (from inside graph/)
4. Run:  python clone_repos.py
"""

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from github import Github

from repo_config import CANDIDATE_REPOS, FINAL_REPOS, CLONE_DIR

load_dotenv()


def test_github_connection():
    """Confirms the token works and prints basic metadata for each final repo."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not found — add it to your .env file first.")

    gh = Github(token)
    print(f"Authenticated as: {gh.get_user().login}\n")

    for repo_name in FINAL_REPOS:
        repo = gh.get_repo(repo_name)
        print(f"{repo_name}")
        print(f"  language:      {repo.language}")
        print(f"  open issues:   {repo.open_issues_count}")
        print(f"  last pushed:   {repo.pushed_at}")
        print()


def clone_final_repos():
    """Clones every repo in FINAL_REPOS into CLONE_DIR (skips if already cloned)."""
    Path(CLONE_DIR).mkdir(exist_ok=True)

    for repo_name in FINAL_REPOS:
        target = Path(CLONE_DIR) / repo_name.split("/")[-1]
        if target.exists():
            print(f"Already cloned: {target}")
            continue

        url = f"https://github.com/{repo_name}.git"
        print(f"Cloning {repo_name} ...")
        subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)

    print("\nDone. Confirm the code is there:")
    for repo_name in FINAL_REPOS:
        target = Path(CLONE_DIR) / repo_name.split("/")[-1]
        n_files = sum(1 for _ in target.rglob("*") if _.is_file())
        print(f"  {target}  ({n_files} files)")


if __name__ == "__main__":
    if not FINAL_REPOS:
        print("FINAL_REPOS is empty — fill it in repo_config.py first "
              "(shortlist CANDIDATE_REPOS, narrow down to 3-5, then move them "
              "into FINAL_REPOS).")
    else:
        test_github_connection()
        clone_final_repos()
