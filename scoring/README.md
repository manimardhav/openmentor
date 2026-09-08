# OpenMentor — Difficulty Scoring & Skill Matching Scaffold

Starter code for Weeks 2–5 of your role. Each file is runnable standalone
(`python3 <file>.py`) with a toy smoke test at the bottom — replace the toy
data with real GitHub issues + your repo's dependency graph as they become
available.

## Files

| File | Week | What it does |
|---|---|---|
| `features.py` | 1–2 | Extracts centrality, text-length, comment-count, label, and referenced-file features from a raw issue. |
| `difficulty_model.py` | 2–3 | `weighted_difficulty_score()` = hand-tuned formula; `DifficultyModel` = trainable logistic regression with stratified k-fold cross-validation and inspectable feature weights. |
| `skill_matching.py` | 4 | Keyword-based skill tagging against `SKILL_TAXONOMY`, plus Jaccard and cosine similarity for contributor-issue matching. |
| `ranking.py` | 5 | Combines difficulty-fit + skill-match into one score, with confidence tagging (`high-confidence` / `exploratory` / `low-confidence`). |

## Week 1 status

- **Pilot repo:** `httpie/cli` — active Python CLI HTTP client, confirmed open
  issues as recently as April 2026. Good fit: single-language, moderate size,
  real ongoing issue traffic.
- **Skill taxonomy:** validated against 12 real httpie/cli issue titles
  (pulled live). Found and fixed a real bug in `tag_issue_skills()` — it was
  regex-escaping keywords, silently breaking pattern-style entries like
  `"colou?r output"`. Also added a `security` tag (2/12 real issues were
  security-related: escape-sequence injection, plaintext credential storage
  — not covered by the original 7 tags) and fixed singular/plural gaps
  (`"header"` vs `"headers"`). No-match rate on the sample dropped from 6/12
  to 4/12 after fixes — titles alone will always under-match vs. title+body,
  so re-validate with `validate_taxonomy.py` once you have full issue bodies.
- **Environment:** verified working (`requirements.txt` added).
- **Literature tie-in:** Santos et al. (2022) is your closest precedent for
  the skill-tagging approach; RepoHyper (2024) for the centrality/graph
  approach. Your gap: nobody in the survey combines both into one ranked
  recommender — that's the novelty angle for Week 8.
- **`validate_taxonomy.py`** — pulls real issues via GitHub API and reports
  per-tag hit rate. The sandbox's shared IP is rate-limited by GitHub
  (60 req/hr, unauthenticated) — run this locally, or set `GITHUB_TOKEN`
  for a 5000 req/hr limit, to validate against a larger sample (~90 issues).

## Suggested next steps

1. **Replace `SKILL_TAXONOMY`** in `skill_matching.py` with tags pulled from
   your actual pilot repo's labels/topics (Week 1).
2. **Wire `features.py` → real GitHub data** using `PyGithub` or the REST API
   to pull issues, comments, and labels for your pilot repo.
3. **Build the dependency graph** (likely from Person A's ingestion module)
   as a `networkx.Graph` and pass it into `compute_centrality_scores()`.
4. **Collect ground-truth difficulty labels** (however your team defines
   them — e.g. time-to-resolution buckets, or manual tester ratings) to
   train `DifficultyModel` for real (Week 3).
5. **Week 6 additions** (not yet scaffolded — happy to build if useful):
   - Baseline scorers: GFI-label-only, single-feature (e.g. issue age).
   - Ablation harness: re-run `DifficultyModel` / `rank_issues_for_contributor_full`
     with each feature group zeroed out, log precision/recall/hit-rate.
   - Stratified evaluation by contributor experience level.

## Known limitations to flag in your methodology write-up

- Skill tagging is pure keyword/regex — expect false positives; a curated
  synonym dictionary per skill tag is the natural v2.
- Betweenness centrality is computed exactly; for large repos, switch to the
  sampled version (`k=100`) for speed.
- Cross-validation fold count is capped by the minority class size — flag
  your dataset size explicitly if it's under ~100 issues.
