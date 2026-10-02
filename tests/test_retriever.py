"""
Integration tests for retrievers and guardrail mechanisms.
"""

from datetime import date
import pytest
from temporal_rag.config import DEFAULT_CONFIG
from temporal_rag.parser import TemporalQueryParser
from temporal_rag.indexer import PMODataIngestion, EmbeddingIndexer
from temporal_rag.engine import VanillaRetriever, RelevanceFirstRetriever


@pytest.fixture(scope="module")
def setup_system():
    ingestor = PMODataIngestion(data_dir="data")
    docs = ingestor.load_corpus(include_tasks=True)
    indexer = EmbeddingIndexer()
    parser = TemporalQueryParser(DEFAULT_CONFIG, ref_date=date(2026, 10, 2))
    vanilla = VanillaRetriever(docs, indexer)
    relevance_first = RelevanceFirstRetriever(docs, indexer, config=DEFAULT_CONFIG, parser=parser)
    return docs, parser, vanilla, relevance_first


def test_guardrail_triggers_for_future_plan(setup_system):
    docs, parser, vanilla, relevance_first = setup_system
    query = "What is the plan for next week for Atlas?"

    # Relevance-First must intercept and trigger guardrail
    results, guardrail_msg = relevance_first.retrieve(query, top_k=3)
    assert guardrail_msg is not None
    assert "No plan found for requested period" in guardrail_msg
    assert "PRJ-101-S12" in guardrail_msg
    assert results[0].guardrail_triggered is True


def test_vanilla_retrieves_expired_docs(setup_system):
    docs, parser, vanilla, relevance_first = setup_system
    query = "Current status of Orion project?"
    results = vanilla.retrieve(query, top_k=3, parser=parser)
    # At least one result in top-3 should be an expired older report
    expired_count = sum(1 for r in results if not r.is_temporally_valid)
    assert expired_count > 0


def test_relevance_first_retrieves_active_status(setup_system):
    docs, parser, vanilla, relevance_first = setup_system
    query = "Current status of Orion project?"
    results, guardrail = relevance_first.retrieve(query, top_k=3)
    assert guardrail is None
    # Top result must be the active status report PRJ-102-STATUS-W12
    assert results[0].doc_id == "PRJ-102-STATUS-W12"
    assert results[0].is_temporally_valid is True
