"""
Unit tests for PMODataIngestion.
"""

import pytest
from temporal_rag.indexer import PMODataIngestion, normalize_date_str


def test_normalize_date_str():
    assert normalize_date_str("13-07-2026") == "2026-07-13"
    assert normalize_date_str("2026-07-13") == "2026-07-13"
    assert normalize_date_str("") == ""
    assert normalize_date_str(None) == ""


def test_corpus_loading():
    ingestor = PMODataIngestion(data_dir="data")
    docs = ingestor.load_corpus(include_tasks=True)
    assert len(docs) >= 95

    # Check supersession
    status_reports = [d for d in docs if d.doc_type == "status_report" and d.project_id == "PRJ-102"]
    assert len(status_reports) > 0
    latest_reports = [d for d in status_reports if d.is_latest]
    assert len(latest_reports) == 1
    assert latest_reports[0].doc_id == "PRJ-102-STATUS-W12"
