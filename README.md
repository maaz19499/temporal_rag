# Relevance-First PMO RAG Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/Architecture-Relevance--First_Hybrid_RAG-orange.svg)]()
[![Evaluation](https://img.shields.io/badge/Temporal_Hit_Rate-92.3%25-brightgreen.svg)]()

> **Core Mandate:** *A semantically accurate document that is temporally invalid is worse than no answer.*

This repository contains the complete implementation and evaluation for the **PMO (Project Management Office) Relevance-First RAG** take-home exercise. The system diagnoses the failure of standard time-blind semantic search and implements a hybrid temporal-semantic retrieval architecture with deterministic guardrails.

---

## 1. Problem Diagnosis: Why Vanilla RAG Fails (Task 1)

In project management repositories, documents across weekly sprint cycles share nearly identical vocabulary (*"Sprint Plan"*, *"Goals"*, *"Atlas"*, *"Auth module"*, *"Capacity"*). 

When querying:
> **"What is the plan for next week for Atlas?"** (Target window: Oct 6–12, 2026)

- **Vanilla Semantic Search (`cosine_similarity(query, doc)`):**
  Matches expired status reports and older sprint plans (e.g., Sprint 10, Sprint 5, Sprint 11) because their cosine similarity is high (~0.60–0.85).
- **Consequence:**
  Stakeholders receive expired commitments from weeks ago treated as future plans.
- **The Intentional Trap (`PRJ-101-S11`):**
  Sprint 11 plan (valid Sep 15–21) has high semantic similarity to "plan for next week" but zero temporal overlap. Our system flags and eliminates this trap.

### Failure Breakdown Across 5 Benchmark Queries

| Query | Vanilla RAG Top Result | Vanilla Status | Relevance-First Result | Relevance-First Status |
| :--- | :--- | :---: | :--- | :---: |
| **1. Atlas plan for next week** | `PRJ-101-STATUS-W10` | 🔴 **EXPIRED** (Sep 14) | Guardrail Intercept ("No plan found... latest is S12") | 🟢 **GUARDED** |
| **2. Orion current status** | `PRJ-102-STATUS-W10` | 🔴 **EXPIRED** (Sep 14) | `PRJ-102-STATUS-W12` (Latest active status report) | 🟢 **VALID** |
| **3. Phoenix who is blocked** | `PRJ-103-S1-T3` | 🔴 **EXPIRED** (July 7) | `PRJ-103-S12-T2` (Lisa M., Blocked in Sprint 12) | 🟢 **VALID** |
| **4. Orion risks** | `PRJ-102-STATUS-W10` | 🔴 **EXPIRED** (Sep 14) | `PRJ-102-RAID-4` (Active open risk, valid to Dec 31) | 🟢 **VALID** |
| **5. Availability next week** | `PRJ-103-STATUS-W3` | 🔴 **EXPIRED** (July 27) | `Omar J.`, `Fatima S.`, `David P.` (`team_status`) | 🟢 **VALID** |

---

## 2. Architecture & Design (Task 2)

```
                            User Query
                                │
                                ▼
                   ┌───────────────────────────┐
                   │  Temporal Query Parser    │
                   │                           │
                   │  - Relative -> ISO range  │
                   │  - Intent detection       │
                   │  - Project entity filter  │
                   └─────────────┬─────────────┘
                                 │
                     [Target Start, Target End]
                                 │
                                 ▼
                   ┌───────────────────────────┐
                   │ Stage 1: Relevance Gate   │
                   │ (Prospective Guardrail)   │
                   └─────────────┬─────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
        [Overlapping Plans Exist]       [No Future Plan Found]
                │                                 │
                ▼                                 ▼
    ┌───────────────────────────┐        ┌─────────────────────────┐
    │ Stage 2: Hybrid Reranker  │        │ Deterministic Guardrail │
    │                           │        │                         │
    │  Final Score =            │        │ "No plan found for      │
    │  0.6 * Temporal_Overlap   │        │  requested period.      │
    │  + 0.4 * Semantic_Sim     │        │  Latest available is X" │
    │  + Doc_Type_Boost         │        └─────────────────────────┘
    │  + Project_Filter         │
    │  + Supersession_Penalty   │
    └───────────┬───────────────┘
                │
                ▼
       Top-K Valid Docs
      (with Citations)
```

### Key Components

1. **Temporal Query Parsing (`temporal_parser.py`):**
   - Deterministically anchors expressions (`"next week"`, `"current"`, `"right now"`) to the system reference date $T_{\text{ref}}$ (`2026-10-02`).
   - Normalizes intervals into ISO-8601 calendar ranges $[T_{\text{start}}, T_{\text{end}}]$.
   - Extracts PMO intent (`planning`, `status`, `blocker`, `risk`, `availability`).
2. **Bi-Temporal Document Ingestion (`ingestion.py`):**
   - Indexes documents with bi-temporal intervals: `valid_from`, `valid_to`, `created_at`.
   - Computes **supersession chains**: for active status queries, older status reports (W1–W11) are superseded by W12.
   - Carries over unresolved tasks (`Blocked`, `In Progress`) from the latest sprint as active.
3. **Hybrid Scoring Formula (`retriever.py`):**
   $$\text{FinalScore}(d) = 0.6 \cdot T(Q_t, D_t) + 0.4 \cdot S(Q, d) + \text{Boost}_{\text{type}} + \text{Boost}_{\text{proj}} + \text{Penalty}_{\text{superseded}}$$
4. **Deterministic Guardrail Fallback:**
   - Intercepts queries asking for non-existent future plans before LLM synthesis, preventing hallucinated answers.

---

## 3. Quantitative Evaluation (Task 3)

Run `python evaluation.py` to reproduce the benchmark results across the 5 standard queries:

| Metric | Vanilla RAG (Cosine Sim) | Relevance-First RAG | Improvement |
| :--- | :---: | :---: | :---: |
| **Temporal Hit Rate (%)** | **13.33%** | **92.31%** | **+79.0%** |
| **Relevance Precision@3 (%)** | **13.33%** | **92.31%** | **+79.0%** |
| **Hallucination Rate (%)** | **0.0%** | **0.0%** | **0.0% (Eliminated)** |

### Metric Definitions
- **Temporal Hit Rate:** Percentage of retrieved documents where `overlap(doc.valid_range, query_range) > 0`.
- **Relevance Precision@3:** Fraction of top-3 retrieved documents that are simultaneously semantically relevant and temporally valid.
- **Hallucination Rate:** Percentage of queries where an expired/superseded plan was returned for a future horizon query.

---

## 4. Repository Structure

```
temporal_rag/
├── README.md               # Design doc and submission walkthrough (this file)
├── WRITEUP_1PAGE.md        # 1-page technical writeup (Loom / defense script)
├── EXERCISE_PROMPT.md      # Original assignment instructions
├── docs/
│   └── approach_tradeoffs_analysis.md # In-depth architectural trade-offs & defense guide
├── rag_corpus.jsonl        # Unified time-tagged PMO corpus (included for direct evaluation)
├── temporal_parser.py      # Deterministic temporal & intent query parser
├── ingestion.py            # Bi-temporal tagging, supersession, & embedding indexer
├── retriever.py            # Vanilla & Relevance-First hybrid retrievers
├── evaluation.py           # Quantitative benchmarking harness (Task 1 & Task 3)
├── demo_app.py             # Streamlit interactive UI (Task 4)
├── cli_demo.py             # Terminal interactive CLI demo
├── data/
│   ├── rag_corpus.jsonl    # Unified time-tagged documents
│   ├── sprint_plan.csv     # 36 sprint plans
│   ├── status_reports.csv  # 36 status reports
│   ├── tasks.csv           # 139 granular tasks with status & assignees
│   ├── team_status.csv     # Team availability records
│   └── raid_log.csv        # Risks, Issues, Dependencies
└── results/
    ├── benchmark_summary.md # Formatted evaluation summary table
    └── benchmark_metrics.json # Full JSON metric traces
```

---

## 5. Quickstart & How to Run

### Dataset & Submission Details (`rag_corpus.jsonl`)
- **Corpus Included:** `rag_corpus.jsonl` is provided directly in the repository at both `./rag_corpus.jsonl` (repo root) and `./data/rag_corpus.jsonl`.
- **Holdout Set Evaluation:** Evaluators can run `python evaluation.py` directly on this corpus, or replace `rag_corpus.jsonl` / point `PMO_DATA_DIR` to a holdout set with different dates. The engine dynamically infers the timeline anchor date ($T_{\text{ref}}$) and maximum sprint number without requiring any hardcoded date modifications.

### Setup Environment
```bash
# 1. Clone repository
git clone <repo_url>
cd temporal_rag

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run Benchmarks (Task 1 & Task 3)
```bash
python evaluation.py
```

### Run Interactive Streamlit Demo (Task 4)
```bash
streamlit run demo_app.py
```

### Run Interactive CLI Demo
```bash
python cli_demo.py
```

---

## 6. Architectural Trade-offs & Interview Defense

For detailed analysis of design decisions, component comparisons, and anticipated interviewer questions, see [docs/approach_tradeoffs_analysis.md](docs/approach_tradeoffs_analysis.md) and [WRITEUP_1PAGE.md](WRITEUP_1PAGE.md).
