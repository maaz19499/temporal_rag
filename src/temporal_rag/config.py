"""
Centralized Configuration for Temporal PMO RAG Agent.
Supports environment variable overrides and typed dataclass parameters.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Tuple
import os


@dataclass(frozen=True)
class PMOConfig:
    # Anchor date for temporal benchmark (defaults to Oct 2, 2026 per dataset specification)
    default_ref_date: date = field(
        default_factory=lambda: date.fromisoformat(
            os.getenv("PMO_REF_DATE", "2026-10-02")
        )
    )

    # Retrieval weights (0.6 temporal + 0.4 semantic per assignment recommendation)
    w_time: float = float(os.getenv("PMO_W_TIME", "0.6"))
    w_sem: float = float(os.getenv("PMO_W_SEM", "0.4"))
    w_type: float = float(os.getenv("PMO_W_TYPE", "0.5"))
    w_proj: float = float(os.getenv("PMO_W_PROJ", "0.4"))

    # Vector model
    embedding_model_name: str = os.getenv("PMO_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Paths
    data_dir: str = os.getenv("PMO_DATA_DIR", "data")
    corpus_filename: str = "rag_corpus.jsonl"
    tasks_filename: str = "tasks.csv"

    # Project Mapping: key -> (project_id, project_name)
    project_mappings: Dict[str, Tuple[str, str]] = field(
        default_factory=lambda: {
            "atlas": ("PRJ-101", "Atlas - Mobile App Revamp"),
            "orion": ("PRJ-102", "Orion - Data Migration"),
            "phoenix": ("PRJ-103", "Phoenix - API Gateway"),
            "prj-101": ("PRJ-101", "Atlas - Mobile App Revamp"),
            "prj-102": ("PRJ-102", "Orion - Data Migration"),
            "prj-103": ("PRJ-103", "Phoenix - API Gateway"),
        }
    )

    # Intent to doc_type boost mapping
    intent_doc_type_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "planning": "sprint_plan",
            "status": "status_report",
            "blocker": "task",
            "risk": "raid",
            "availability": "team_status",
        }
    )


# Global default configuration instance
DEFAULT_CONFIG = PMOConfig()
