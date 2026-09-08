"""
build_ground_truth.py — Week 5: build the "answer key" your team's system will
be tested against later, plus two baselines to compare it with.

Reads: data/issues.csv (already collected in Week 4 — no new API calls needed)

Writes three files into data/:
  1. ground_truth.csv     — every historically CLOSED issue, with whether it
                             was resolved by a first-time contributor and how
                             long it took. This is the "correct answers" file
                             used in Week 6 to score your system's accuracy.
  2. baseline_gfi.csv      — every currently OPEN issue that has a "good first
                             issue" label, exactly as GitHub presents it, with
                             no extra logic applied. Baseline #1.
  3. baseline_naive.csv    — every currently OPEN issue, ranked oldest-first.
                             No AI, no graph, just issue age. Baseline #2.

Honesty note (carry this into your report): "resolved_by_newcomer" here reuses
the "resolver_is_first_time" flag from Week 4, which — as already documented —
reflects first-time-within-our-sample, not each user's full GitHub history.
"""

import csv
from pathlib import Path

from dotenv import load_dotenv
from github import Github
import os

from repo_config import OUTPUT_DIR, FINAL_REPOS

load_dotenv()

INPUT_FILE = Path(OUTPUT_DIR) / "issues.csv"

# Safety cap when scanning a repo's open issues live (pandas/matplotlib can
# have thousands open; we stop scanning once we've checked this many).
MAX_OPEN_ISSUES_SCANNED = 3000


def load_issues() -> list[dict]:
    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_good_first_issue(labels_field: str) -> bool:
    return "good first issue" in labels_field.lower()


def build_ground_truth(issues: list[dict]) -> list[dict]:
    rows = []
    for issue in issues:
        if issue["state"] != "closed":
            continue
        rows.append({
            "issue_id": issue["issue_id"],
            "repo_name": issue["repo_name"],
            "title": issue["title"],
            "had_gfi_label": is_good_first_issue(issue["labels"]),
            "resolved_by_newcomer": issue["resolver_is_first_time"],
            "days_to_close": issue["days_to_close"],
        })
    return rows


def build_baseline_gfi_live(gh: Github) -> list[dict]:
    """
    Your Week 4 sample only kept the 1000 MOST RECENT issues per repo, and
    good-first-issue-labeled issues tend to get claimed and closed fast — so
    the stored sample can easily contain zero currently-open GFI issues.
    This fetches directly from GitHub instead, so the baseline reflects what
    a contributor would actually see right now.
    """
    rows = []
    for repo_name in FINAL_REPOS:
        repo = gh.get_repo(repo_name)
        print(f"Scanning open issues for {repo_name} (live) ...")
        scanned = 0
        found = 0
        for issue in repo.get_issues(state="open", sort="created", direction="desc"):
            if issue.pull_request is not None:
                continue
            scanned += 1
            if scanned > MAX_OPEN_ISSUES_SCANNED:
                break
            label_names = ";".join(l.name for l in issue.labels)
            if is_good_first_issue(label_names):
                found += 1
                rows.append({
                    "issue_id": issue.number,
                    "repo_name": repo_name,
                    "title": issue.title,
                    "created_at": issue.created_at.isoformat() if issue.created_at else None,
                })
        print(f"  scanned {scanned} open issues, found {found} with a good-first-issue label")
    return rows


def build_baseline_naive(issues: list[dict]) -> list[dict]:
    open_issues = [i for i in issues if i["state"] == "open"]
    # oldest first = naive "just work through the backlog in order" ranking
    open_issues.sort(key=lambda i: i["created_at"])
    rows = []
    for rank, issue in enumerate(open_issues, start=1):
        rows.append({
            "rank": rank,
            "issue_id": issue["issue_id"],
            "repo_name": issue["repo_name"],
            "title": issue["title"],
            "created_at": issue["created_at"],
        })
    return rows


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def get_github_client() -> Github:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not found — check your .env file.")
    return Github(token)


def main():
    issues = load_issues()
    print(f"Loaded {len(issues)} issues from {INPUT_FILE}")

    ground_truth = build_ground_truth(issues)
    write_csv(
        ground_truth,
        Path(OUTPUT_DIR) / "ground_truth.csv",
        ["issue_id", "repo_name", "title", "had_gfi_label",
         "resolved_by_newcomer", "days_to_close"],
    )

    gh = get_github_client()
    baseline_gfi = build_baseline_gfi_live(gh)
    write_csv(
        baseline_gfi,
        Path(OUTPUT_DIR) / "baseline_gfi.csv",
        ["issue_id", "repo_name", "title", "created_at"],
    )

    baseline_naive = build_baseline_naive(issues)
    write_csv(
        baseline_naive,
        Path(OUTPUT_DIR) / "baseline_naive.csv",
        ["rank", "issue_id", "repo_name", "title", "created_at"],
    )

    # Quick printed summary so you can eyeball the balance without opening files
    n_closed = len(ground_truth)
    n_newcomer = sum(1 for r in ground_truth if r["resolved_by_newcomer"] == "True")
    print(f"\nGround truth: {n_closed} closed issues, "
          f"{n_newcomer} ({100*n_newcomer/max(n_closed,1):.1f}%) resolved by a first-time contributor")
    print(f"Baseline (GFI-labeled, open): {len(baseline_gfi)} issues")
    print(f"Baseline (naive, oldest-first, open): {len(baseline_naive)} issues")


if __name__ == "__main__":
    main()