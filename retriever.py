"""
retriever.py
------------
Root entrypoint delegating to src.temporal_rag.engine.
Maintains backward compatibility with assignment evaluation harness.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from temporal_rag.models import RetrievalResult
from temporal_rag.engine import BaseRetriever, VanillaRetriever, RelevanceFirstRetriever

__all__ = ["RetrievalResult", "BaseRetriever", "VanillaRetriever", "RelevanceFirstRetriever"]
