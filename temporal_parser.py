"""
temporal_parser.py
------------------
Root entrypoint delegating to src.temporal_rag.parser.
Maintains backward compatibility with assignment evaluation harness.
"""

import sys
import os

# Ensure src/ is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from temporal_rag.models import ParsedQuery
from temporal_rag.parser import TemporalQueryParser
from temporal_rag.config import DEFAULT_CONFIG

PROJECT_MAPPINGS = DEFAULT_CONFIG.project_mappings
INTENT_DOC_TYPE_BOOST = DEFAULT_CONFIG.intent_doc_type_mapping

__all__ = ["TemporalQueryParser", "ParsedQuery", "PROJECT_MAPPINGS", "INTENT_DOC_TYPE_BOOST"]

if __name__ == "__main__":
    parser = TemporalQueryParser()
    sample_queries = [
        "What is the plan for next week for Atlas?",
        "Current status of Orion project?",
        "Who is blocked right now in Phoenix?",
        "What are the risks for Orion?",
        "Who is available next week?",
    ]
    for q in sample_queries:
        parsed = parser.parse(q)
        print(f"Query: {q}")
        print(f"  Project: {parsed.project_name} ({parsed.project_id})")
        print(f"  Intent: {parsed.intent} (Boost: {parsed.target_doc_type})")
        print(f"  Target Range: {parsed.target_start} to {parsed.target_end}")
        print(f"  Flags: Prospective={parsed.is_prospective}, Current={parsed.is_current}")
        print()
