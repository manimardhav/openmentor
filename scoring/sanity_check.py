"""
week2_sanity_check.py — Week 2 final step: "test the script on a small
sample of issues... and sanity-check the outputs by hand."

Data source note: titles and labels are REAL, pulled from httpie/cli's
GitHub issues via search. `files_touched` is a HEURISTIC GUESS based on
issue content — not real GitHub file-diff data. Comment counts are set to
0 (unknown) rather than guessed. Swap in real values once you have API/
Person A access.
"""

import pandas as pd
from features import build_feature_dataframe
from difficulty_model import add_weighted_scores
from build_placeholder_graph import build_placeholder_graph

REAL_ISSUES = [
    {
        "id": 1571, "body": "unable to CONTRIBUTING",
        "comments": 0, "labels": [], "files_touched": [],
        "your_intuition": "easy",
    },
    {
        "id": 1604, "body": "please provide support for scoop download",
        "comments": 0, "labels": ["bug"], "files_touched": [],
        "your_intuition": "easy-medium",
    },
    {
        "id": 1568, "body": "Ability to define a custom default user-agent in config.json",
        "comments": 0, "labels": ["enhancement"], "files_touched": ["httpie/config.py"],
        "your_intuition": "medium",
    },
    {
        "id": 1569, "body": "Add .netrc support when using --auth-type bearer",
        "comments": 0, "labels": ["enhancement"], "files_touched": ["httpie/sessions.py", "httpie/client.py"],
        "your_intuition": "medium",
    },
    {
        "id": 1603, "body": "Intermediate Response headers not printed for Digest Auth",
        "comments": 0, "labels": ["bug"], "files_touched": ["httpie/client.py", "httpie/sessions.py"],
        "your_intuition": "medium-hard",
    },
    {
        "id": 1637,
        "body": ("I'm getting weird behavior when specifying headers on the command line. "
                  "I've searched for similar issues. I'm using the latest version of HTTPie. "
                  "httpie v3.2.4 Win11 Pro x64 24H2 Python 3.10.0 Note that I'm new to httpie. "
                  "Running the example from https://httpie.io/docs/cli/http-headers "
                  "(adding --offline so it outputs the request headers), the only header "
                  "that shows up from those given on the command line is the Referer header."),
        "comments": 0, "labels": ["bug"], "files_touched": ["httpie/client.py", "httpie/cli/argparser.py"],
        "your_intuition": "medium-hard",
    },
    {
        "id": 1564, "body": "conda-forge lists __win as a MatchSpec dependency for httpie=2.2.0 preventing installation on linux-64",
        "comments": 0, "labels": ["bug"], "files_touched": [],
        "your_intuition": "hard",
    },
    {
        "id": 1562, "body": "it is not possible install for arm linux",
        "comments": 0, "labels": ["bug"], "files_touched": [],
        "your_intuition": "hard",
    },
    {
        "id": 1555,
        "body": "Add cli option to generate scriptable or compilable code, or a native executable, that performs the command wherein the option was included",
        "comments": 0, "labels": ["enhancement"], "files_touched": ["httpie/core.py", "httpie/cli/definition.py"],
        "your_intuition": "hard",
    },
    {
        "id": 1574, "body": "CLI help doesn't show how to launch in a browser!",
        "comments": 0, "labels": [], "files_touched": ["httpie/cli/definition.py"],
        "your_intuition": "easy",
    },
]


def run_sanity_check():
    graph = build_placeholder_graph()
    df = build_feature_dataframe(REAL_ISSUES, graph)
    df = add_weighted_scores(df)

    lookup = {i["id"]: i for i in REAL_ISSUES}
    df["title"] = df["issue_id"].map(lambda i: lookup[i]["body"][:55])
    df["your_intuition"] = df["issue_id"].map(lambda i: lookup[i]["your_intuition"])

    df = df.sort_values("difficulty_score_formula").reset_index(drop=True)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 55)
    print(df[["issue_id", "title", "difficulty_score_formula", "your_intuition"]].to_string(index=False))
    return df


if __name__ == "__main__":
    run_sanity_check()