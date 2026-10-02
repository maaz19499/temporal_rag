"""
ingestion.py
------------
Root entrypoint delegating to src.temporal_rag.indexer.
Maintains backward compatibility with assignment evaluation harness.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from temporal_rag.models import Document
from temporal_rag.indexer import PMODataIngestion, EmbeddingIndexer, normalize_date_str

__all__ = ["Document", "PMODataIngestion", "EmbeddingIndexer", "normalize_date_str"]

if __name__ == "__main__":
    ingestor = PMODataIngestion()
    docs = ingestor.load_corpus(include_tasks=True)
    print(f"Total documents ingested: {len(docs)}")
    types = {}
    for d in docs:
        types[d.doc_type] = types.get(d.doc_type, 0) + 1
    print("Document type breakdown:", types)

    indexer = EmbeddingIndexer()
    print(f"Embedding backend initialized: {indexer.backend}")
