# Data Contract

## Member 3 → data/graph_metrics.csv
file_path, betweenness, pagerank, degree, repo_name

## Member 3 → data/issues.csv
issue_id, repo_name, title, body, labels, linked_pr, resolver, close_date,
resolver_is_first_time, affected_files, centrality_of_affected_files

## Member 1 → data/ranked_issues.json
List of objects:
{
  "issue_id": str,
  "difficulty_score": float,
  "skill_match_score": float,
  "final_rank_score": float,
  "matched_skills": [str],
  "confidence_tag": "high-confidence" | "exploratory",
  "top_features": { ... }
}