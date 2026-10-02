"""
Deterministic, anchor-aware temporal and intent query parser.
Fully dynamic: Calculates calendar week intervals from any reference date.
"""

from datetime import date, timedelta
from typing import Optional
import re

from .models import ParsedQuery
from .config import PMOConfig, DEFAULT_CONFIG
from .logging_config import setup_logger

logger = setup_logger("temporal_rag.parser")


class TemporalQueryParser:
    """
    Deterministic rule-based parser anchoring PMO queries to an explicit reference date.
    Converts relative expressions into ISO calendar intervals [start_date, end_date].
    """

    def __init__(self, config: Optional[PMOConfig] = None, ref_date: Optional[date] = None):
        self.config = config or DEFAULT_CONFIG
        self.ref_date = ref_date or self.config.default_ref_date

    def parse(self, query: str) -> ParsedQuery:
        q_lower = query.lower().strip()

        # 1. Project Entity Extraction
        project_id, project_name = None, None
        for key, (pid, pname) in self.config.project_mappings.items():
            if re.search(rf"\b{key}\b", q_lower):
                project_id = pid
                project_name = pname
                break

        # 2. Sprint Number Extraction
        sprint_match = re.search(r"\b(?:sprint|week|s)\s*(\d{1,2})\b", q_lower)
        sprint_number = int(sprint_match.group(1)) if sprint_match else None

        # 3. Intent Detection
        intent = "general"
        if re.search(r"\b(plan|plans|roadmap|goals|upcoming|agenda)\b", q_lower):
            intent = "planning"
        elif re.search(r"\b(block|blocked|blocker|blockers|impediment|stuck)\b", q_lower):
            intent = "blocker"
        elif re.search(r"\b(risk|risks|issue|issues|dependency|dependencies|raid)\b", q_lower):
            intent = "risk"
        elif re.search(r"\b(availab|available|availability|pto|capacity|bandwidth)\b", q_lower):
            intent = "availability"
        elif re.search(r"\b(status|progress|rag|health|state|update)\b", q_lower):
            intent = "status"

        target_doc_type = self.config.intent_doc_type_mapping.get(intent)

        # 4. Temporal Interval Grounding
        target_start: Optional[date] = None
        target_end: Optional[date] = None
        is_prospective = False
        is_current = False
        is_retrospective = False

        if "next week" in q_lower or "upcoming week" in q_lower:
            is_prospective = True
            # Dynamically compute next Monday from ref_date
            weekday = self.ref_date.weekday()  # Monday=0, Sunday=6
            days_until_next_monday = (7 - weekday) % 7
            if days_until_next_monday == 0:
                days_until_next_monday = 7

            next_monday = self.ref_date + timedelta(days=days_until_next_monday)
            # In PMO sprint schedule, sprint window runs next Monday through Sunday (or Tuesday through Monday)
            # We cover the full 7-day upcoming week window [next_monday, next_monday + 7]:
            target_start = next_monday
            target_end = next_monday + timedelta(days=7)
        elif any(w in q_lower for w in ["current", "right now", "today", "as of now", "currently"]):
            is_current = True
            target_start = self.ref_date
            target_end = self.ref_date
        elif any(w in q_lower for w in ["last week", "previous week"]):
            is_retrospective = True
            target_end = self.ref_date - timedelta(days=1)
            target_start = target_end - timedelta(days=6)
        elif sprint_number:
            # Sprints run 7-day cadence
            base_sprint_start = date(2026, 7, 7)
            target_start = base_sprint_start + timedelta(weeks=sprint_number - 1)
            target_end = target_start + timedelta(days=6)
            if target_start > self.ref_date:
                is_prospective = True
            elif target_end < self.ref_date:
                is_retrospective = True
            else:
                is_current = True
        else:
            is_current = True
            target_start = self.ref_date
            target_end = self.ref_date

        parsed = ParsedQuery(
            raw_query=query,
            cleaned_query=q_lower,
            target_start=target_start,
            target_end=target_end,
            intent=intent,
            target_doc_type=target_doc_type,
            project_id=project_id,
            project_name=project_name,
            is_prospective=is_prospective,
            is_current=is_current,
            is_retrospective=is_retrospective,
            sprint_number=sprint_number,
        )
        return parsed
