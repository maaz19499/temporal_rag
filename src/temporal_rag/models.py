"""
Core Domain Models and Data Structures for Temporal PMO RAG.
"""

from dataclasses import dataclass, asdict, field
from datetime import date
from typing import Optional, Dict, Any


@dataclass
class Document:
    doc_id: str
    project_id: str
    project_name: str
    doc_type: str  # sprint_plan, status_report, task, team_status, raid
    text: str
    created_at: str
    valid_from: str
    valid_to: str
    is_latest: bool = True
    superseded_by: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedQuery:
    raw_query: str
    cleaned_query: str
    target_start: Optional[date]
    target_end: Optional[date]
    intent: str  # planning, status, blocker, risk, availability, general
    target_doc_type: Optional[str]
    project_id: Optional[str]
    project_name: Optional[str]
    is_prospective: bool  # Asking about the future (e.g. next week)
    is_current: bool      # Asking about right now / active state
    is_retrospective: bool # Asking about the past
    sprint_number: Optional[int] = None

    def overlaps(self, valid_from_str: Optional[str], valid_to_str: Optional[str]) -> bool:
        """
        Evaluates whether document validity interval overlaps query target interval.
        Two intervals [A, B] and [C, D] overlap iff A <= D and C <= B.
        """
        if not self.target_start or not self.target_end:
            return True

        try:
            doc_start = date.fromisoformat(valid_from_str) if valid_from_str else date.min
            doc_end = date.fromisoformat(valid_to_str) if valid_to_str else date.max
        except (ValueError, TypeError):
            return False

        return doc_start <= self.target_end and self.target_start <= doc_end

    def temporal_overlap_score(self, valid_from_str: Optional[str], valid_to_str: Optional[str]) -> float:
        """
        Computes interval overlap fraction in [0.0, 1.0] with soft distance decay.
        """
        if not self.target_start or not self.target_end:
            return 1.0

        try:
            doc_start = date.fromisoformat(valid_from_str) if valid_from_str else date.min
            doc_end = date.fromisoformat(valid_to_str) if valid_to_str else date.max
        except (ValueError, TypeError):
            return 0.0

        # Exact interval overlap
        if doc_start <= self.target_end and self.target_start <= doc_end:
            overlap_start = max(doc_start, self.target_start)
            overlap_end = min(doc_end, self.target_end)
            overlap_days = max(0, (overlap_end - overlap_start).days + 1)
            query_days = max(1, (self.target_end - self.target_start).days + 1)
            return min(1.0, overlap_days / query_days)

        # Distance penalty if separated
        if doc_end < self.target_start:
            gap_days = (self.target_start - doc_end).days
        else:
            gap_days = (doc_start - self.target_end).days

        # Soft exponential decay based on days of separation
        return max(0.0, float(0.5 ** (gap_days / 7.0)))


@dataclass
class RetrievalResult:
    doc_id: str
    project_id: str
    project_name: str
    doc_type: str
    text: str
    valid_from: str
    valid_to: str
    created_at: str
    semantic_score: float
    temporal_score: float
    type_boost: float
    project_boost: float
    final_score: float
    is_temporally_valid: bool
    is_latest: bool
    guardrail_triggered: bool = False
    guardrail_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "doc_type": self.doc_type,
            "text": self.text,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "semantic_score": round(self.semantic_score, 4),
            "temporal_score": round(self.temporal_score, 4),
            "final_score": round(self.final_score, 4),
            "is_temporally_valid": self.is_temporally_valid,
            "guardrail_triggered": self.guardrail_triggered,
            "guardrail_message": self.guardrail_message,
        }
