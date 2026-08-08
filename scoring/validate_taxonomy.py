"""
validate_taxonomy.py — Week 1 sanity check: pull real issues from the pilot
repo and see how often each skill tag in SKILL_TAXONOMY actually fires.

Usage:
    python3 validate_taxonomy.py                  # anonymous, 60 req/hr limit
    GITHUB_TOKEN=ghp_xxx python3 validate_taxonomy.py   # 5000 req/hr limit

If you hit "API rate limit exceeded" with no token, either wait an hour,
run this from your own machine/network instead of a shared IP, or set
GITHUB_TOKEN (a personal access token with no special scopes needed for
public repos) as an environment variable.
"""

import os
import json
import urllib.request
import urllib.error
from collections import Counter

from skill_matching import tag_issue_skills, SKILL_TAXONOMY

REPO = "httpie/cli"
PAGES_TO_FETCH = 3       # ~90 issues at 30/page — enough for a first sanity check
PER_PAGE = 30


def fetch_issues(repo: str, pages: int = PAGES_TO_FETCH, per_page: int = PER_PAGE) -> list:
    """
    Pulls issues (both open and closed) via the public GitHub REST API.
    Filters out pull requests, since the /issues endpoint returns both.
    """
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "openmentor-taxonomy-check"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    all_issues = []
    for page in range(1, pages + 1):
        url = f"https://api.github.com/repos/{repo}/issues?state=all&per_page={per_page}&page={page}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            print(f"HTTP {e.code} fetching page {page}: {body[:300]}")
            break

        if isinstance(data, dict) and "message" in data:
            # Rate limit or other API error — data is an error object, not a list
            print(f"API error: {data['message']}")
            break

        page_issues = [i for i in data if "pull_request" not in i]
        if not page_issues:
            break
        all_issues.extend(page_issues)

    return all_issues


def validate_taxonomy(issues: list, taxonomy: dict = None) -> None:
    taxonomy = taxonomy or SKILL_TAXONOMY
    tag_counts = Counter()
    zero_tag_count = 0

    for issue in issues:
        text = f"{issue.get('title', '')} {issue.get('body') or ''}"
        matched = tag_issue_skills(text, taxonomy)
        if not matched:
            zero_tag_count += 1
        for tag in matched:
            tag_counts[tag] += 1

    total = len(issues)
    print(f"\nFetched {total} issues from {REPO}\n")
    print(f"{'Tag':<22}{'Hits':>6}{'Hit rate':>12}")
    print("-" * 40)
    for tag in taxonomy:
        hits = tag_counts.get(tag, 0)
        rate = f"{hits/total:.0%}" if total else "n/a"
        print(f"{tag:<22}{hits:>6}{rate:>12}")

    print("-" * 40)
    print(f"{'Issues with NO tag matched':<22}{zero_tag_count:>6}{zero_tag_count/total:.0%}" if total else "")

    print("\nInterpretation:")
    print("- A tag with ~0% hit rate is either mis-worded (check keyword list)")
    print("  or genuinely rare in this repo (consider dropping/merging it).")
    print("- A high 'no tag matched' rate means the taxonomy is too narrow —")
    print("  sample a few of those issues manually and add a tag for them.")


if __name__ == "__main__":
    issues = fetch_issues(REPO)
    if issues:
        validate_taxonomy(issues)
    else:
        print("No issues fetched — see error above. Run locally or set GITHUB_TOKEN.")
