"""
Unit tests for TemporalQueryParser.
"""

from datetime import date
import pytest
from temporal_rag.parser import TemporalQueryParser
from temporal_rag.config import DEFAULT_CONFIG


@pytest.fixture
def parser():
    return TemporalQueryParser(DEFAULT_CONFIG, ref_date=date(2026, 10, 2))


def test_next_week_parsing(parser):
    parsed = parser.parse("What is the plan for next week for Atlas?")
    assert parsed.is_prospective is True
    assert parsed.intent == "planning"
    assert parsed.project_id == "PRJ-101"
    assert parsed.project_name == "Atlas - Mobile App Revamp"
    assert parsed.target_start == date(2026, 10, 5)
    assert parsed.target_end >= date(2026, 10, 11)


def test_current_status_parsing(parser):
    parsed = parser.parse("Current status of Orion project?")
    assert parsed.is_current is True
    assert parsed.intent == "status"
    assert parsed.project_id == "PRJ-102"
    assert parsed.target_start == date(2026, 10, 2)
    assert parsed.target_end == date(2026, 10, 2)


def test_blocker_parsing(parser):
    parsed = parser.parse("Who is blocked right now in Phoenix?")
    assert parsed.is_current is True
    assert parsed.intent == "blocker"
    assert parsed.project_id == "PRJ-103"
    assert parsed.target_doc_type == "task"


def test_availability_parsing(parser):
    parsed = parser.parse("Who is available next week?")
    assert parsed.is_prospective is True
    assert parsed.intent == "availability"
    assert parsed.project_id is None  # Portfolio-wide
    assert parsed.target_doc_type == "team_status"


def test_interval_overlap(parser):
    parsed = parser.parse("What is the plan for next week for Atlas?")
    # Sprint 11: Sep 15 to Sep 21 -> Should NOT overlap Oct 6-12
    assert parsed.overlaps("2026-09-15", "2026-09-21") is False

    # A hypothetical future sprint: Oct 6 to Oct 12 -> Should overlap
    assert parsed.overlaps("2026-10-06", "2026-10-12") is True
