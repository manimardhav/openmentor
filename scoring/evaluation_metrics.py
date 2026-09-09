"""
evaluation_metrics.py — standard ranked-recommendation metrics: precision@k,
recall@k, hit-rate@k. Used by both the ablation study and the baseline
comparison table.
"""


def precision_at_k(ranked_ids: list, relevant_ids: set, k: int) -> float:
    """Of the top k recommended issues, what fraction are actually relevant?"""
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for issue_id in top_k if issue_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(ranked_ids: list, relevant_ids: set, k: int) -> float:
    """Of all the actually-relevant issues, what fraction appear in the top k?"""
    if not relevant_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def hit_rate_at_k(ranked_ids: list, relevant_ids: set, k: int) -> float:
    """Binary: does at least one relevant issue appear in the top k?"""
    top_k = set(ranked_ids[:k])
    return 1.0 if (top_k & relevant_ids) else 0.0


def evaluate_ranking(ranked_ids: list, relevant_ids: set, k_values: list = None) -> dict:
    """Convenience: compute all three metrics across several k values at once."""
    k_values = k_values or [1, 3, 5]
    results = {}
    for k in k_values:
        results[f"precision@{k}"] = round(precision_at_k(ranked_ids, relevant_ids, k), 3)
        results[f"recall@{k}"] = round(recall_at_k(ranked_ids, relevant_ids, k), 3)
        results[f"hit_rate@{k}"] = round(hit_rate_at_k(ranked_ids, relevant_ids, k), 3)
    return results


if __name__ == "__main__":
    ranked = [101, 102, 103, 104, 105]
    relevant = {102, 105, 999}

    print("Ranked:", ranked)
    print("Relevant:", relevant)
    print(evaluate_ranking(ranked, relevant, k_values=[1, 3, 5]))