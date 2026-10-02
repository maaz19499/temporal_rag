# Relevance-First RAG Exercise

### Role: AI Engineer / RAG Specialist
### Time: 2-3 days (take-home)
### Stack: Python + vector DB (Hyrbrid) / LLM

---

## 1. Context

You are building a **PMO (Project Management Office) Agent** for a portfolio of projects.

The agent ingests project data: sprint plans, weekly status reports (RAG status), tasks, RAID logs, and team availability. Stakeholders will ask questions like:

- "What is the plan for next week for Atlas?"
- "Current status of Orion?"
- "Who is blocked right now?"
- "Who is available next week?"

A standard RAG implementation fails here. Example:
> Q: "What is the plan for next week?"
> Naive RAG returns: Sprint plan from 2 weeks ago (because embedding similarity for "plan" is 0.89)
> User reaction: "This is useless."

**The core challenge is not accuracy, it's relevance.** A semantically accurate but temporally irrelevant answer is worse than no answer.

Your task is to design a RAG strategy where **relevance > accuracy**.

## 2. Dataset Provided

We provide two datasets. Use both.

### A. Synthetic PMO Dataset (primary - in this repo)
`./data/` contains 95 time-tagged documents across 3 projects:

| File | Content | Key Fields |
|------|---------|------------|
| `rag_corpus.jsonl` | Unified corpus ready to ingest (95 docs) | `text`, `doc_id`, `project_id`, `doc_type`, `created_at`, `valid_from`, `valid_to` |
| `sprint_plans.csv` | 36 sprint plans (12 weeks x 3 projects) | `valid_from` = sprint start, `valid_to` = sprint end |
| `status_reports.csv` | 36 weekly status reports with RAG | `rag_status`, `progress_pct`, valid for 2 days after sprint |
| `tasks.csv` | 139 tasks with status, assignee, due_date | `status` in [To Do, In Progress, Done, Blocked] |
| `team_status.csv` | Current team availability | `valid_from=today`, `valid_to=today+7d` |
| `raid_log.csv` | Risks, Issues, Dependencies | `type`, `status` |

> **Intentional trap:** `PRJ-101-S11` (Sprint 11 plan, valid Sep 15-21) has high semantic similarity to "plan for next week (Oct 6-12)" but zero temporal overlap. A correct system must NOT return it.

Schema example:
```json
{
  "doc_id": "PRJ-101-S11",
  "project_id": "PRJ-101",
  "text": "Sprint 11 Plan for Atlas...",
  "doc_type": "sprint_plan",
  "created_at": "2026-09-13",
  "valid_from": "2026-09-15",
  "valid_to": "2026-09-21"
}
```

### B. EnterpriseRAG-Bench (validation - external)
Use as secondary validation: https://github.com/onyx-dot-app/EnterpriseRAG-Bench

This is 500k enterprise docs (Slack, Gmail, Linear, Jira, Confluence). Focus on these categories from `questions.jsonl`:

- **Project Related (40 Qs):** Aggregate knowledge from single project
- **Constrained (30 Qs):** Multiple relevant docs but qualifiers disqualify all but one - perfect for temporal relevance
- **Conflicting Info (20 Qs):** Docs contradict each other (old vs new plan)
- **extra_questions.jsonl (100 Qs):** Metadata-dependent questions

You do NOT need to index full 500k. Filter to Linear + Jira + Confluence for PMO.

## 3. Exercise Tasks

### Task 1: Diagnose Failure (Mandatory)
Demonstrate the failure of vanilla semantic RAG on this query set:

```python
QUERIES = [
  "What is the plan for next week for Atlas?",
  "Current status of Orion project?",
  "Who is blocked right now in Phoenix?",
  "What are the risks for Orion?",
  "Who is available next week?"
]
```

Show what a pure `cosine_similarity(query, doc)` returns vs what SHOULD be returned. Quantify temporal mismatch rate.

### Task 2: Design Relevance-First RAG Strategy (Core)
Implement a retrieval pipeline that prioritizes relevance.

You must address:

1.  **Temporal Query Understanding:** Parse "next week", "current", "right now" into absolute date ranges `[query_start, query_end]`. Don't rely on LLM at generation time.

2.  **Time-Aware Retrieval:** Implement at least one of:
    - Hard time filters (exclude docs where `valid_to < query_start`)
    - Time-decay scoring: `final_score = w_sem * semantic + w_time * temporal_score`
    - Version-aware supersession (if doc is superseded, mark inactive)
    - Two-stage: Filter by time -> Rank by semantics

3.  **Relevance over Accuracy Guardrail:** If no doc's `valid_from` overlaps query time range, return "No plan found for requested period. Latest available is X" - never hallucinate old data as future.

Reference approach: TimelyRAG - Semantic retrieval stage (top-N) + Temporal re-ranking stage combining `S(Q,d)` with `T(Q_T, D_T)`.

### Task 3: Evaluation (Mandatory)
Define and measure relevance, not just accuracy.

Report these metrics on the 5 queries above:

- **Temporal Hit Rate:** % of retrieved docs where `overlap(doc.valid_range, query_range) > 0`
- **Relevance Precision@3:** Of top 3 docs, how many are both semantically relevant AND temporally relevant?
- **Hallucination Rate:** % where system returned old plan for future query

Bonus: Evaluate on 20 random questions from EnterpriseRAG-Bench Project Related + Constrained.

### Task 4: PMO Agent Demo (Optional but Strong Plus)
Build a simple chat interface (Streamlit / Gradio / CLI) where user can ask:

> "team", "timeline", "current status", "future state"

And show:
- Retrieved docs with `valid_from`, `valid_to`, and final score breakdown (semantic vs temporal)
- Final answer with citations

## 4. Deliverables

1.  **GitHub Repo** with:
    - `ingestion.py` - how you tag and index docs with temporal metadata
    - `retriever.py` - your temporal-aware retrieval logic
    - `evaluation.py` - metrics for Task 3
    - `README.md` (your design doc) explaining your relevance-first strategy

2.  **Short Loom / Writeup (1 page max):** Why did you choose `w_time` vs `w_sem`? How do you handle "current status" vs "retrospective" differently?

3.  **Do NOT include:** API keys, large vector DB dumps. Use local Chroma / FAISS.

## 5. Evaluation Rubric (How We Will Score You)

| Criterion | Weight | What we look for |
|-----------|--------|------------------|
| **Problem Diagnosis** | 20% | Did you clearly demonstrate why vanilla RAG fails on temporal queries with data? |
| **Temporal Grounding** | 30% | Query time parsing + doc time tagging. Is time a first-class citizen or just metadata? |
| **Relevance Strategy** | 30% | Final score formula, hard filters, guardrail against irrelevant return. Bonus for handling conflicting info (old vs new) |
| **Evaluation** | 20% | Did you measure temporal hit rate, not just BLEU / semantic similarity? |

We value **simple, explainable** over complex. A well-tuned `0.4*semantic + 0.6*temporal_overlap` with hard filters beats a 3-stage LLM reranker you can't explain.

## 6. Hints

- Read: TimelyRAG paper - Semantic-Temporal Hybrid Retrieval for Time-Critical QA
- Read: EnterpriseRAG-Bench methodology - they intentionally inject "near-duplicates with updated or conflicting facts" to test this.
- For PMO: `doc_type` matters. For "current status" -> boost `status_report`, for "plan" -> boost `sprint_plan`.
- The best answer for "plan for next week" when no future plan exists is "Not found", not last week's plan.

## 7. Submission

- Share GitHub repo link
- Include `rag_corpus.jsonl` or instructions to download it
- We will run your `evaluation.py` on our holdout set (same schema, different dates)

Good luck!

---
*This exercise tests product thinking + RAG engineering. We care how you reason about relevance, not just vector search.*
