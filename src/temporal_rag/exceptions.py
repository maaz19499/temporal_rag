"""
Domain-specific exceptions for the Temporal PMO RAG Agent.
"""


class PMORagException(Exception):
    """Base exception for PMO RAG agent."""
    pass


class TemporalParsingError(PMORagException):
    """Raised when temporal query parsing fails or receives invalid date inputs."""
    pass


class CorpusIngestionError(PMORagException):
    """Raised when document corpus loading or tagging fails."""
    pass


class EmbeddingError(PMORagException):
    """Raised when vector embedding generation fails."""
    pass


class GuardrailTriggeredException(PMORagException):
    """Raised/signaled when a deterministic guardrail intercepts execution."""
    def __init__(self, message: str, latest_available_info: str = ""):
        super().__init__(message)
        self.latest_available_info = latest_available_info
