"""
Vanilla and Relevance-First Retrieval Engines with temporal gating and guardrails.
"""

from typing import List, Optional, Tuple
import numpy as np

from .models import Document, RetrievalResult
from .parser import TemporalQueryParser
from .indexer import EmbeddingIndexer
from .config import PMOConfig, DEFAULT_CONFIG
from .logging_config import setup_logger

logger = setup_logger("temporal_rag.engine")


class BaseRetriever:
    """Base vector retrieval engine."""

    def __init__(self, documents: List[Document], indexer: EmbeddingIndexer):
        self.documents = documents
        self.indexer = indexer
        self.doc_texts = [d.text for d in documents]
        self.doc_embeddings = self.indexer.embed_corpus(self.doc_texts)

    def compute_semantic_scores(self, query: str) -> np.ndarray:
        q_emb = self.indexer.embed_query(query)
        sims = np.dot(self.doc_embeddings, q_emb)
        return np.clip(sims, -1.0, 1.0)


class VanillaRetriever(BaseRetriever):
    """
    Standard naive semantic search baseline using pure cosine similarity.
    Completely time-blind.
    """

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        parser: Optional[TemporalQueryParser] = None
    ) -> List[RetrievalResult]:
        parser = parser or TemporalQueryParser()
        parsed_q = parser.parse(query)
        sims = self.compute_semantic_scores(query)
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            sem_score = float(sims[idx])
            temp_score = parsed_q.temporal_overlap_score(doc.valid_from, doc.valid_to)
            is_valid = parsed_q.overlaps(doc.valid_from, doc.valid_to)

            results.append(RetrievalResult(
                doc_id=doc.doc_id,
                project_id=doc.project_id,
                project_name=doc.project_name,
                doc_type=doc.doc_type,
                text=doc.text,
                valid_from=doc.valid_from,
                valid_to=doc.valid_to,
                created_at=doc.created_at,
                semantic_score=sem_score,
                temporal_score=temp_score,
                type_boost=0.0,
                project_boost=0.0,
                final_score=sem_score,
                is_temporally_valid=is_valid,
                is_latest=doc.is_latest,
            ))
        return results


class RelevanceFirstRetriever(BaseRetriever):
    """
    Production Relevance-First Hybrid Retriever.
    Enforces temporal validity gating, doc_type boosting, supersession handling,
    and deterministic 'No plan found' guardrails.
    """

    def __init__(
        self,
        documents: List[Document],
        indexer: EmbeddingIndexer,
        config: Optional[PMOConfig] = None,
        parser: Optional[TemporalQueryParser] = None,
    ):
        super().__init__(documents, indexer)
        self.config = config or DEFAULT_CONFIG
        self.parser = parser or TemporalQueryParser(self.config)

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> Tuple[List[RetrievalResult], Optional[str]]:
        parsed_q = self.parser.parse(query)
        sem_scores = self.compute_semantic_scores(query)

        # -------------------------------------------------------------
        # STAGE 1: Guardrail Check for Prospective Planning Queries
        # -------------------------------------------------------------
        if parsed_q.is_prospective and parsed_q.intent == "planning":
            overlapping_plans = [
                d for d in self.documents
                if d.doc_type == "sprint_plan"
                and (not parsed_q.project_id or d.project_id == parsed_q.project_id)
                and parsed_q.overlaps(d.valid_from, d.valid_to)
            ]
            if not overlapping_plans:
                candidate_plans = [
                    d for d in self.documents
                    if d.doc_type == "sprint_plan"
                    and (not parsed_q.project_id or d.project_id == parsed_q.project_id)
                ]
                candidate_plans.sort(key=lambda x: x.valid_to, reverse=True)
                latest_info = ""
                if candidate_plans:
                    latest = candidate_plans[0]
                    latest_info = f"Latest available plan is {latest.doc_id} (valid {latest.valid_from} to {latest.valid_to})."

                guardrail_msg = (
                    f"No plan found for requested period ({parsed_q.target_start} to {parsed_q.target_end}). "
                    f"{latest_info}"
                )
                logger.info(f"Guardrail triggered for query '{query}': {guardrail_msg}")
                guardrail_result = RetrievalResult(
                    doc_id="GUARDRAIL-INTERCEPT",
                    project_id=parsed_q.project_id or "",
                    project_name=parsed_q.project_name or "",
                    doc_type="guardrail",
                    text=guardrail_msg,
                    valid_from=str(parsed_q.target_start),
                    valid_to=str(parsed_q.target_end),
                    created_at=str(self.parser.ref_date),
                    semantic_score=0.0,
                    temporal_score=0.0,
                    type_boost=0.0,
                    project_boost=0.0,
                    final_score=0.0,
                    is_temporally_valid=True,
                    is_latest=True,
                    guardrail_triggered=True,
                    guardrail_message=guardrail_msg,
                )
                return [guardrail_result], guardrail_msg

        # -------------------------------------------------------------
        # STAGE 2: Scoring Candidates with Temporal & Metadata Boosts
        # -------------------------------------------------------------
        candidate_scores = []
        for idx, doc in enumerate(self.documents):
            # Strict project isolation: discard docs from other projects when specified
            if parsed_q.project_id and doc.project_id and doc.project_id != parsed_q.project_id:
                continue

            sem_score = float(sem_scores[idx])

            # In PMO, for "current status", the latest status report (W12) is the active status
            if parsed_q.is_current and parsed_q.intent == "status" and doc.doc_type == "status_report":
                if doc.is_latest:
                    temp_score = 1.0
                    is_valid = True
                else:
                    temp_score = 0.0
                    is_valid = False
            else:
                temp_score = parsed_q.temporal_overlap_score(doc.valid_from, doc.valid_to)
                is_valid = parsed_q.overlaps(doc.valid_from, doc.valid_to)

            # Doc-type boosting
            type_boost = 0.0
            if parsed_q.target_doc_type:
                if doc.doc_type == parsed_q.target_doc_type:
                    type_boost = self.config.w_type
                else:
                    if parsed_q.intent in ["availability", "risk", "status", "blocker"]:
                        type_boost = -0.5

            # Project boosting
            proj_boost = 0.0
            if parsed_q.project_id:
                if doc.project_id == parsed_q.project_id:
                    proj_boost = self.config.w_proj
                elif doc.project_id != "" and doc.project_id != parsed_q.project_id:
                    proj_boost = -0.8

            # Supersession penalty: older status reports are obsolete for current queries
            supersession_penalty = 0.0
            if parsed_q.is_current and not doc.is_latest and doc.doc_type == "status_report":
                supersession_penalty = -0.6

            # Specific intent filters / boosts
            status_penalty = 0.0
            if parsed_q.intent == "blocker" and doc.doc_type == "task":
                task_status = (doc.extra_metadata or {}).get("status", "")
                if task_status.lower() == "blocked":
                    sprint_num = int((doc.extra_metadata or {}).get("sprint", 0) or 0)
                    status_penalty = 0.3 + (sprint_num / 20.0)
                else:
                    status_penalty = -0.8
            elif parsed_q.intent == "risk" and doc.doc_type == "raid":
                raid_text = doc.text.lower()
                if "status closed" in raid_text:
                    status_penalty = -0.4
                else:
                    status_penalty = 0.2

            # Composite scoring formula
            final_score = (
                self.config.w_time * temp_score +
                self.config.w_sem * sem_score +
                type_boost +
                proj_boost +
                supersession_penalty +
                status_penalty
            )

            # Penalize expired docs on current or prospective queries
            if (parsed_q.is_prospective or parsed_q.is_current) and not is_valid:
                final_score *= 0.2

            candidate_scores.append((final_score, idx, sem_score, temp_score, type_boost, proj_boost, is_valid))

        candidate_scores.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidate_scores[:top_k]

        results = []
        for final_sc, idx, sem_sc, temp_sc, t_boost, p_boost, is_val in top_candidates:
            doc = self.documents[idx]
            results.append(RetrievalResult(
                doc_id=doc.doc_id,
                project_id=doc.project_id,
                project_name=doc.project_name,
                doc_type=doc.doc_type,
                text=doc.text,
                valid_from=doc.valid_from,
                valid_to=doc.valid_to,
                created_at=doc.created_at,
                semantic_score=sem_sc,
                temporal_score=temp_sc,
                type_boost=t_boost,
                project_boost=p_boost,
                final_score=final_sc,
                is_temporally_valid=is_val,
                is_latest=doc.is_latest,
                guardrail_triggered=False,
            ))

        return results, None
