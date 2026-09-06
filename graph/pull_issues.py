"""
pull_issues.py — Week 4: pull issue data for each repo in FINAL_REPOS and
export data/issues.csv, matching the schema in shared/schemas.md:

    issue_id, repo_name, title, labels, state, created_at, closed_at,
    closed_by, days_to_close, resolver_is_first_time

Run this AFTER clone_repos.py has already confirmed your GitHub token works.

Important honesty note (write this into your report too): "resolver_is_first_time"
here means "first time we SAW this username close an issue, within the sample
we collected" — not their true full account history on GitHub (that would need
many extra API calls per user and isn't worth the API-rate-limit cost for a
final-year project). State this as a known limitation/approximation.

Similarly, "closed_by" is GitHub's own record of who closed the issue. Most of
the time (especially for auto-closed issues) this is the person whose pull
request fixed it, but occasionally a maintainer closes something someone else
fixed. This is a reasonable approximation, not a perfect ground truth.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from github import Github
import os

from repo_config import FINAL_REPOS, OUTPUT_DIR

load_dotenv()

# Safety cap so a huge repo (e.g. pandas has 3000+ issues) doesn't eat your
# entire hourly API rate limit or take forever on a laptop. Raise this later
# once you've confirmed everything works end to end.
MAX_ISSUES_PER_REPO = 1000


def get_github_client() -> Github:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not found — check your .env file.")
    return Github(token)


def days_between(start, end) -> float:
    if start is None or end is None:
        return None
    return round((end - start).total_seconds() / 86400, 2)


def pull_issues_for_repo(gh: Github, repo_name: str) -> list[dict]:
    repo = gh.get_repo(repo_name)
    rows = []
    seen_closers = set()  # tracks usernames we've already seen close an issue

    print(f"Pulling issues for {repo_name} ...")
    issues = repo.get_issues(state="all", sort="created", direction="desc")

    count = 0
    for issue in issues:
        if issue.pull_request is not None:
            continue  # GitHub's API lists PRs as "issues" too — skip those

        count += 1
        if count > MAX_ISSUES_PER_REPO:
            print(f"  Reached MAX_ISSUES_PER_REPO ({MAX_ISSUES_PER_REPO}), stopping early.")
            break

        closer = issue.closed_by.login if issue.closed_by else None
        is_first_time = None
        if closer:
            is_first_time = closer not in seen_closers
            seen_closers.add(closer)

        rows.append({
            "issue_id": issue.number,
            "repo_name": repo_name,
            "title": issue.title,
            "labels": ";".join(label.name for label in issue.labels),
            "state": issue.state,
            "created_at": issue.created_at.isoformat() if issue.created_at else None,
            "closed_at": issue.closed_at.isoformat() if issue.closed_at else None,
            "closed_by": closer,
            "days_to_close": days_between(issue.created_at, issue.closed_at),
            "resolver_is_first_time": is_first_time,
        })

        if count % 100 == 0:
            print(f"  ...{count} issues processed so far")

    print(f"  Done: {count} issues pulled for {repo_name}")
    return rows


def export_issues_csv(all_rows: list[dict], output_path: Path):
    fieldnames = ["issue_id", "repo_name", "title", "labels", "state",
                  "created_at", "closed_at", "closed_by", "days_to_close",
                  "resolver_is_first_time"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {output_path}")


def main():
    gh = get_github_client()
    all_rows = []

    for repo_name in FINAL_REPOS:
        rows = pull_issues_for_repo(gh, repo_name)
        all_rows.extend(rows)

    output_path = Path(OUTPUT_DIR) / "issues.csv"
    output_path.parent.mkdir(exist_ok=True)
    export_issues_csv(all_rows, output_path)


if __name__ == "__main__":
    main()