"""
enrich_issue_links.py — fills in the 3 fields your team's schema needs that
require extra GitHub API calls, so we don't slow down the main issue pull:

    linked_pr                     — which PR actually closed this issue
    affected_files                — which files that PR changed
    centrality_of_affected_files  — the highest centrality score among those
                                     files (from graph_metrics.csv), matching
                                     the MAX-based approach scoring/features.py
                                     already uses for this exact purpose

Only applies to CLOSED issues — an open issue has no resolving PR yet, so
these 3 fields stay blank for open rows (correct, not missing data).

Run this AFTER pull_issues.py has already produced data/issues.csv.

SAFE TO INTERRUPT: this script saves its progress back to issues.csv every
25 issues (and on Ctrl+C), and skips any closed issue it already checked if
you run it again — so you can stop anytime with Ctrl+C and re-run later to
pick up where you left off, without losing earlier progress or re-doing
work.

How "linked_pr" is found: GitHub's issue timeline includes "cross-referenced"
events whenever a pull request mentions this issue (e.g. "Fixes #123"). We
take the LAST such reference to a since-closed PR before the issue itself
closed — a reasonable heuristic, not a guarantee (documented as a known
approximation, same spirit as the resolver_is_first_time approximation).
"""

import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from github import Github

from repo_config import OUTPUT_DIR

load_dotenv()

ISSUES_FILE = Path(OUTPUT_DIR) / "issues.csv"
GRAPH_METRICS_FILE = Path(OUTPUT_DIR) / "graph_metrics.csv"

MAX_FILES_PER_PR = 30      # safety cap for unusually huge PRs
SAVE_EVERY = 25            # write progress back to disk this often
NOT_FOUND = "NONE_FOUND"   # marks "we checked, found nothing" vs "not checked yet"


def get_github_client() -> Github:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not found — check your .env file.")
    return Github(token)


def load_centrality_index() -> dict:
    """{repo_name: {normalized_file_path: betweenness}} — forward slashes,
    so it matches GitHub's path format regardless of what OS graph_metrics.csv
    was generated on."""
    index = {}
    with open(GRAPH_METRICS_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            repo = row["repo_name"]
            path = row["file_path"].replace("\\", "/")
            index.setdefault(repo, {})[path] = float(row["betweenness"])
    return index


def find_linked_pr(repo, issue_number: int):
    try:
        issue = repo.get_issue(issue_number)
        candidate = None
        for event in issue.get_timeline():
            if event.event == "cross-referenced" and event.source and event.source.issue:
                src = event.source.issue
                if src.pull_request is not None and src.state == "closed":
                    candidate = src.number  # keep the LAST match found
        return candidate
    except Exception as e:
        print(f"    (couldn't check timeline for issue #{issue_number}: {e})")
        return None


def get_affected_files(repo, pr_number: int):
    try:
        files = []
        for f in repo.get_pull(pr_number).get_files():
            files.append(f.filename)
            if len(files) >= MAX_FILES_PER_PR:
                break
        return files
    except Exception as e:
        print(f"    (couldn't fetch files for PR #{pr_number}: {e})")
        return []


def save_rows(rows: list[dict], fieldnames: list[str]):
    with open(ISSUES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    gh = get_github_client()
    centrality_index = load_centrality_index()

    with open(ISSUES_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

    repo_cache = {}
    processed = 0
    enriched = 0
    already_done = 0
    since_last_save = 0

    try:
        for row in rows:
            if row["state"] != "closed":
                continue  # open issues: leave the 3 fields blank, correctly

            # RESUME LOGIC: skip anything we already checked in a previous run
            if row["linked_pr"] not in ("", None):
                already_done += 1
                continue

            processed += 1
            repo_name = row["repo_name"]
            if repo_name not in repo_cache:
                repo_cache[repo_name] = gh.get_repo(repo_name)
            repo = repo_cache[repo_name]

            pr_number = find_linked_pr(repo, int(row["issue_id"]))
            if pr_number is None:
                row["linked_pr"] = NOT_FOUND
                row["affected_files"] = NOT_FOUND
                row["centrality_of_affected_files"] = NOT_FOUND
            else:
                affected_files = get_affected_files(repo, pr_number)
                if not affected_files:
                    # ACTUAL BUG FIX: previously this kept the (possibly
                    # invalid/cross-repo) pr_number here even though the file
                    # fetch failed — misleading. Mark all 3 fields consistently
                    # as not-found instead.
                    row["linked_pr"] = NOT_FOUND
                    row["affected_files"] = NOT_FOUND
                    row["centrality_of_affected_files"] = NOT_FOUND
                else:
                    repo_centrality = centrality_index.get(repo_name, {})
                    scores = [repo_centrality.get(f, 0.0) for f in affected_files]
                    row["linked_pr"] = pr_number
                    row["affected_files"] = ";".join(affected_files)
                    row["centrality_of_affected_files"] = round(max(scores) if scores else 0.0, 6)
                    enriched += 1

            since_last_save += 1
            if since_last_save >= SAVE_EVERY:
                save_rows(rows, fieldnames)
                since_last_save = 0
                print(f"  ...checked {processed} issues this run "
                      f"({already_done} already done earlier, {enriched} matched to a PR) — saved")

    except KeyboardInterrupt:
        print("\nInterrupted — saving progress before exiting...")
        save_rows(rows, fieldnames)
        print(f"Saved. {processed} issues checked this run before stopping. "
              f"Re-run this same command later to continue from here.")
        return

    save_rows(rows, fieldnames)
    print(f"\nDone. Checked {processed} closed issues this run "
          f"({already_done} were already done in an earlier run), "
          f"successfully linked {enriched} to a PR + affected files.")
    print(f"Updated {ISSUES_FILE}")


if __name__ == "__main__":
    main()