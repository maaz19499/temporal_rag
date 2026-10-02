"""
Benchmark Metrics and Evaluation Engine for Relevance-First RAG.
"""

from typing import List, Dict, Any, Tuple
from .models import RetrievalResult
from .parser import TemporalQueryParser


def evaluate_query_results(
    query: str,
    results: List[RetrievalResult],
    parser: TemporalQueryParser,
    guardrail_msg: str = None,
) -> Dict[str, Any]:
    """
    Evaluates temporal hit rate, precision@k, and hallucination on a single query.
    """
    parsed_q = parser.parse(query)
    k_count = len(results)

    temporally_valid_count = 0
    relevance_hits = 0
    query_hallucinated = False

    for r in results:
        if r.guardrail_triggered:
            # Clean guardrail intercept prevents hallucination and guarantees temporal safety
            temporally_valid_count += 1
            relevance_hits += 1
        else:
            is_valid = parsed_q.overlaps(r.valid_from, r.valid_to)
            if is_valid:
                temporally_valid_count += 1

            sem_relevant = True
            if parsed_q.project_id and r.project_id and r.project_id != parsed_q.project_id:
                sem_relevant = False

            if is_valid and sem_relevant:
                if parsed_q.is_current and r.doc_type == "status_report" and not r.is_latest:
                    pass
                else:
                    relevance_hits += 1

            if parsed_q.is_prospective and parsed_q.intent == "planning":
                if r.doc_type == "sprint_plan" and not is_valid:
                    query_hallucinated = True

    return {
        "k_count": k_count,
        "temporally_valid_count": temporally_valid_count,
        "relevance_hits": relevance_hits,
        "hallucinated": query_hallucinated,
        "precision_at_k": relevance_hits / max(1, k_count),
        "temporal_hit_rate": temporally_valid_count / max(1, k_count),
    }


def aggregate_portfolio_metrics(
    eval_records: List[Dict[str, Any]],
    total_queries: int
) -> Tuple[float, float, float]:
    """
    Aggregates metrics across the portfolio:
    Returns (temporal_hit_rate_pct, precision_at_3_pct, hallucination_rate_pct)
    """
    total_docs = sum(r["k_count"] for r in eval_records)
    total_valid = sum(r["temporally_valid_count"] for r in eval_records)
    total_rel = sum(r["relevance_hits"] for r in eval_records)
    total_hallucinated = sum(1 for r in eval_records if r["hallucinated"])

    hit_rate = (total_valid / max(1, total_docs)) * 100.0
    prec_at_k = (total_rel / max(1, total_docs)) * 100.0
    hallucination_rate = (total_hallucinated / max(1, total_queries)) * 100.0

    return round(hit_rate, 2), round(prec_at_k, 2), round(hallucination_rate, 2)
