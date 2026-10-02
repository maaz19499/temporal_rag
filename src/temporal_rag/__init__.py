"""
Temporal RAG: Production-Grade Relevance-First RAG Framework for Enterprise PMO.
"""

from .config import PMOConfig, DEFAULT_CONFIG
from .models import Document, ParsedQuery, RetrievalResult
from .parser import TemporalQueryParser
from .indexer import PMODataIngestion, EmbeddingIndexer
from .engine import VanillaRetriever, RelevanceFirstRetriever
from .exceptions import (
    PMORagException,
    TemporalParsingError,
    CorpusIngestionError,
    EmbeddingError,
    GuardrailTriggeredException,
)
from .metrics import evaluate_query_results, aggregate_portfolio_metrics
from .logging_config import setup_logger

__version__ = "1.0.0"
__all__ = [
    "PMOConfig",
    "DEFAULT_CONFIG",
    "Document",
    "ParsedQuery",
    "RetrievalResult",
    "TemporalQueryParser",
    "PMODataIngestion",
    "EmbeddingIndexer",
    "VanillaRetriever",
    "RelevanceFirstRetriever",
    "PMORagException",
    "TemporalParsingError",
    "CorpusIngestionError",
    "EmbeddingError",
    "GuardrailTriggeredException",
    "evaluate_query_results",
    "aggregate_portfolio_metrics",
    "setup_logger",
]
