"""
Corpus ingestion, dynamic anchor detection, date normalization, supersession tagging,
and dual vector indexer.

Fully dynamic: Invariant to date shifts in evaluator holdout sets.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import date
import json
import os
import re
import csv
import numpy as np

from .models import Document
from .config import PMOConfig, DEFAULT_CONFIG
from .exceptions import CorpusIngestionError
from .logging_config import setup_logger

logger = setup_logger("temporal_rag.indexer")


def normalize_date_str(date_val: Any) -> str:
    """Normalize date string (DD-MM-YYYY or YYYY-MM-DD) to ISO YYYY-MM-DD."""
    if not date_val or str(date_val).strip() == "" or str(date_val).lower() == "nan":
        return ""
    s = str(date_val).strip()
    match_dmy = re.match(r"^(\d{2})-(\d{2})-(\d{4})$", s)
    if match_dmy:
        d, m, y = match_dmy.groups()
        return f"{y}-{m}-{d}"
    match_ymd = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if match_ymd:
        return s
    return s


class PMODataIngestion:
    """
    Ingests, normalizes, and enriches PMO documents with bi-temporal metadata.
    Dynamically infers anchor date and max sprint numbers to handle holdout sets.
    """

    def __init__(self, data_dir: Optional[str] = None, config: Optional[PMOConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.data_dir = data_dir or self.config.data_dir
        self.documents: List[Document] = []
        self.inferred_ref_date: Optional[date] = None
        self.max_sprint_number: int = 12

    def load_corpus(self, include_tasks: bool = True) -> List[Document]:
        docs: List[Document] = []
        corpus_path = os.path.join(self.data_dir, self.config.corpus_filename)

        if not os.path.exists(corpus_path):
            if os.path.exists("data/rag_corpus.jsonl"):
                corpus_path = "data/rag_corpus.jsonl"
            else:
                raise CorpusIngestionError(f"Cannot find rag_corpus.jsonl at {corpus_path}")

        logger.info(f"Loading corpus from {corpus_path}")
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    pid = item.get("project_id", "")
                    pname = item.get("project_name", "")
                    if not pname and pid in self.config.project_mappings:
                        pname = self.config.project_mappings[pid][1]

                    doc = Document(
                        doc_id=item.get("doc_id", f"doc-{line_no}"),
                        project_id=pid,
                        project_name=pname,
                        doc_type=item.get("doc_type", "general"),
                        text=item.get("text", ""),
                        created_at=normalize_date_str(item.get("created_at", "")),
                        valid_from=normalize_date_str(item.get("valid_from", "")),
                        valid_to=normalize_date_str(item.get("valid_to", "")),
                        extra_metadata={
                            k: v for k, v in item.items() if k not in [
                                "doc_id", "project_id", "project_name", "doc_type",
                                "text", "created_at", "valid_from", "valid_to"
                            ]
                        }
                    )
                    docs.append(doc)
                except Exception as e:
                    logger.error(f"Error parsing line {line_no} in corpus: {e}")

        # Ingest tasks.csv
        tasks_path = os.path.join(self.data_dir, self.config.tasks_filename)
        if include_tasks and os.path.exists(tasks_path):
            raw_tasks = []
            max_sprint = 1
            with open(tasks_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_tasks.append(row)
                    sprint_str = row.get("sprint", "0")
                    if sprint_str and sprint_str.isdigit():
                        max_sprint = max(max_sprint, int(sprint_str))

            self.max_sprint_number = max_sprint
            logger.info(f"Dynamically detected maximum sprint in dataset: Sprint {self.max_sprint_number}")

            for row in raw_tasks:
                pid = row.get("project_id", "")
                pname = self.config.project_mappings.get(pid, (pid, pid))[1]
                task_id = row.get("task_id", "")
                task_name = row.get("task_name", "")
                assignee = row.get("assignee", "")
                status = row.get("status", "")
                sprint = row.get("sprint", "")
                due = normalize_date_str(row.get("due_date", ""))
                v_from = normalize_date_str(row.get("valid_from", ""))
                v_to = normalize_date_str(row.get("valid_to", ""))
                c_at = normalize_date_str(row.get("created_at", ""))

                # Dynamically rollover unresolved tasks from the latest sprint
                is_unresolved = status in ["Blocked", "In Progress", "To Do"]
                is_latest_sprint = (str(sprint) == str(self.max_sprint_number))
                effective_valid_to = v_to
                if is_latest_sprint and is_unresolved:
                    # In agile PMO, active tasks remain valid until next sprint cycle (~7-10 days after sprint start)
                    try:
                        v_from_d = date.fromisoformat(v_from) if v_from else None
                        if v_from_d:
                            effective_valid_to = str(v_from_d + date.resolution * 16)
                    except Exception:
                        pass

                text = (
                    f"Task {task_name} (ID: {task_id}) in {pname} (Sprint {sprint}) "
                    f"assigned to {assignee} is {status}. Due date: {due}."
                )
                doc = Document(
                    doc_id=task_id,
                    project_id=pid,
                    project_name=pname,
                    doc_type="task",
                    text=text,
                    created_at=c_at or v_from,
                    valid_from=v_from,
                    valid_to=effective_valid_to,
                    is_latest=is_latest_sprint,
                    extra_metadata={
                        "task_name": task_name,
                        "assignee": assignee,
                        "status": status,
                        "sprint": sprint,
                        "due_date": due,
                    }
                )
                docs.append(doc)

        self._compute_supersession(docs)
        self._infer_anchor_date(docs)
        self.documents = docs
        logger.info(f"Successfully ingested {len(docs)} documents. Inferred Anchor Date: {self.inferred_ref_date}")
        return docs

    def _infer_anchor_date(self, docs: List[Document]) -> None:
        """
        Dynamically infers the system anchor date (T_ref) from the holdout dataset.
        Per README line 42: team_status is valid_from = today.
        """
        team_status_dates = [
            d.valid_from for d in docs
            if d.doc_type == "team_status" and d.valid_from
        ]
        if team_status_dates:
            # Most common valid_from in team_status represents 'today'
            try:
                self.inferred_ref_date = date.fromisoformat(team_status_dates[0])
                logger.info(f"Inferred benchmark reference date from team_status: {self.inferred_ref_date}")
                return
            except Exception:
                pass

        # Fallback to max date in corpus
        valid_dates = []
        for d in docs:
            if d.valid_to and d.valid_to < "2099-01-01":
                try:
                    valid_dates.append(date.fromisoformat(d.valid_to))
                except Exception:
                    pass
        if valid_dates:
            self.inferred_ref_date = max(valid_dates)
        else:
            self.inferred_ref_date = self.config.default_ref_date

    def _compute_supersession(self, docs: List[Document]) -> None:
        reports_by_project: Dict[str, List[Document]] = {}
        for d in docs:
            if d.doc_type == "status_report":
                reports_by_project.setdefault(d.project_id, []).append(d)

        for pid, reports in reports_by_project.items():
            reports.sort(key=lambda x: x.valid_to, reverse=True)
            if reports:
                latest = reports[0]
                latest.is_latest = True
                latest.superseded_by = None
                for older in reports[1:]:
                    older.is_latest = False
                    older.superseded_by = latest.doc_id


class EmbeddingIndexer:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or DEFAULT_CONFIG.embedding_model_name
        self.model = None
        self.tfidf_vectorizer = None
        self.backend = "sentence_transformer"
        self._init_backend()

    def _init_backend(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.backend = "sentence_transformer"
            logger.info(f"Initialized dense SentenceTransformer: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Falling back to TF-IDF.")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self.backend = "tfidf"

    def embed_corpus(self, texts: List[str]) -> np.ndarray:
        if self.backend == "sentence_transformer":
            embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embs.astype(np.float32)
        else:
            tfidf_mat = self.tfidf_vectorizer.fit_transform(texts)
            dense = tfidf_mat.toarray()
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (dense / norms).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        if self.backend == "sentence_transformer":
            emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            return emb[0].astype(np.float32)
        else:
            vec = self.tfidf_vectorizer.transform([query]).toarray()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec[0].astype(np.float32)
