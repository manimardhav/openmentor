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
    # timeout=60: wait longer before giving up on a slow/flaky connection
    # retry=3: automatically retry a failed request up to 3 times before erroring
    return Github(token, timeout=60, retry=3)


def days_between(start, end) -> float:
    if start is None or end is None:
        return None
    return round((end - start).total_seconds() / 86400, 2)


import re

# Patterns for common leaked-secret formats that sometimes appear pasted
# (accidentally) inside GitHub issue text. We redact these before saving,
# since committing someone else's real token would be a genuine security
# problem, not just a GitHub push-protection annoyance.
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),           # Hugging Face tokens
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),          # GitHub personal access tokens
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),  # GitHub fine-grained tokens
    re.compile(r"sk-[A-Za-z0-9]{20,}"),           # OpenAI-style keys
    re.compile(r"AKIA[A-Z0-9]{16}"),              # AWS access key IDs
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack tokens
]


def redact_secrets(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED_SECRET]", text)
    return text


def truncate_body(text, cap=2000):
    if not text:
        return ""
    text = text.replace("\r\n", " ").replace("\n", " ").strip()
    text = redact_secrets(text)
    return text[:cap]


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

        resolver = issue.closed_by.login if issue.closed_by else None
        is_first_time = None
        if resolver:
            is_first_time = resolver not in seen_closers
            seen_closers.add(resolver)

        rows.append({
            "issue_id": issue.number,
            "repo_name": repo_name,
            "title": issue.title,
            "body": truncate_body(issue.body),
            "labels": ";".join(label.name for label in issue.labels),
            # filled in later by enrich_issue_links.py, only for closed issues:
            "linked_pr": "",
            "resolver": resolver,
            "close_date": issue.closed_at.isoformat() if issue.closed_at else None,
            "resolver_is_first_time": is_first_time,
            "affected_files": "",
            "centrality_of_affected_files": "",
            # extra columns kept for our own Week 5 scripts (not required by
            # shared/schemas.md, but harmless — Member 1 can ignore them):
            "state": issue.state,
            "created_at": issue.created_at.isoformat() if issue.created_at else None,
            "days_to_close": days_between(issue.created_at, issue.closed_at),
        })

        if count % 100 == 0:
            print(f"  ...{count} issues processed so far")

    print(f"  Done: {count} issues pulled for {repo_name}")
    return rows


FIELDNAMES = ["issue_id", "repo_name", "title", "body", "labels",
              "linked_pr", "resolver", "close_date", "resolver_is_first_time",
              "affected_files", "centrality_of_affected_files",
              "state", "created_at", "days_to_close"]


def export_issues_csv(all_rows: list[dict], output_path: Path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {output_path}")


def load_already_done_repos(output_path: Path) -> tuple[list[dict], set]:
    """
    RESUME SUPPORT: if issues.csv already exists from a previous (possibly
    crashed) run, load it and figure out which repos are already fully done,
    so we don't waste time/API-calls re-pulling them.

    SAFETY CHECK: only trust the existing file if it actually has the
    CURRENT expected columns. An older file (from before body/linked_pr/etc.
    were added) has the same repo names in it, but the wrong schema — so we
    detect that and treat the whole file as stale, forcing a fresh pull,
    rather than silently skipping repos that need to be redone.
    """
    if not output_path.exists():
        return [], set()

    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)

    if set(FIELDNAMES) - set(header):
        missing = set(FIELDNAMES) - set(header)
        print(f"Existing issues.csv has an OUTDATED schema (missing columns: "
              f"{', '.join(missing)}). Ignoring it and pulling all repos fresh.")
        return [], set()

    done_repos = set(r["repo_name"] for r in rows)
    print(f"Found existing issues.csv (current schema) with data for: {', '.join(done_repos) or '(none)'}")
    return rows, done_repos


def main():
    gh = get_github_client()
    output_path = Path(OUTPUT_DIR) / "issues.csv"
    output_path.parent.mkdir(exist_ok=True)

    all_rows, done_repos = load_already_done_repos(output_path)

    for repo_name in FINAL_REPOS:
        if repo_name in done_repos:
            print(f"Skipping {repo_name} — already pulled in a previous run.")
            continue

        rows = pull_issues_for_repo(gh, repo_name)
        all_rows.extend(rows)

        # SAVE IMMEDIATELY after each repo finishes — so if the NEXT repo
        # crashes (network blip, rate limit, etc.), everything up to here
        # is already safely on disk and won't need to be re-pulled.
        export_issues_csv(all_rows, output_path)


if __name__ == "__main__":
    main()