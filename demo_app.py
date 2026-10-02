"""
demo_app.py
-----------
Streamlit interactive demo for the Relevance-First PMO RAG Agent.
Demonstrates:
- Side-by-side comparison of Vanilla RAG vs Relevance-First RAG.
- Temporal score breakdown (Semantic vs. Temporal vs. Boosts).
- Deterministic Guardrail Intercepts for non-existent future plans.
- Citations and validity interval inspection.
"""

import streamlit as st
import datetime
import pandas as pd
from temporal_parser import TemporalQueryParser
from ingestion import PMODataIngestion, EmbeddingIndexer
from retriever import VanillaRetriever, RelevanceFirstRetriever


st.set_page_config(
    page_title="PMO Relevance-First RAG Agent",
    page_icon="⏱️",
    layout="wide",
)


@st.cache_resource
def load_system():
    ingestor = PMODataIngestion()
    docs = ingestor.load_corpus(include_tasks=True)
    indexer = EmbeddingIndexer()
    inferred_date = ingestor.inferred_ref_date or datetime.date(2026, 10, 2)
    parser = TemporalQueryParser(
        ref_date=inferred_date,
        sprint_dates=getattr(ingestor, "sprint_dates", {}),
        project_mappings=getattr(ingestor, "discovered_projects", {}),
    )
    vanilla_retriever = VanillaRetriever(docs, indexer)
    relevance_retriever = RelevanceFirstRetriever(docs, indexer, parser=parser)
    return docs, parser, vanilla_retriever, relevance_retriever


docs, parser, vanilla_retriever, relevance_retriever = load_system()

st.title("⏱️ PMO Agent: Relevance-First RAG")
st.markdown(
    "**Core Mandate:** *A semantically accurate document that is temporally invalid is worse than no answer.*"
)

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    ref_date = st.date_input("System Reference Date ($T_{ref}$)", datetime.date(2026, 10, 2))
    parser.ref_date = ref_date
    top_k = st.slider("Top-K Retrieved Documents", min_value=1, max_value=5, value=3)

    st.markdown("---")
    st.subheader("Task 4 Quick Intents")
    quick_col1, quick_col2 = st.columns(2)
    with quick_col1:
        if st.button("👥 team", use_container_width=True):
            st.session_state["query_input"] = "Who is available next week?"
        if st.button("📊 current status", use_container_width=True):
            st.session_state["query_input"] = "Current status of Orion project?"
    with quick_col2:
        if st.button("📅 timeline", use_container_width=True):
            st.session_state["query_input"] = "What was planned in Sprint 11 for Atlas?"
        if st.button("🔮 future state", use_container_width=True):
            st.session_state["query_input"] = "What is the plan for next week for Atlas?"

    st.markdown("---")
    st.subheader("Preset Benchmark Queries")
    presets = [
        "What is the plan for next week for Atlas?",
        "Current status of Orion project?",
        "Who is blocked right now in Phoenix?",
        "What are the risks for Orion?",
        "Who is available next week?",
    ]
    selected_preset = st.selectbox("Choose a benchmark query:", ["(Custom)"] + presets)
    if selected_preset != "(Custom)":
        st.session_state["query_input"] = selected_preset

# Query Input
default_q = st.session_state.get("query_input", "What is the plan for next week for Atlas?")
query = st.text_input("Enter your query:", value=default_q)

if st.button("Run Retrieval & Diagnosis", type="primary") or query:
    parsed_q = parser.parse(query)

    # Display Query Understanding Card
    st.subheader("🔍 1. Temporal Query Understanding")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Project", parsed_q.project_name or "All Projects")
    c2.metric("Intent", parsed_q.intent.capitalize())
    c3.metric("Target Horizon", f"{parsed_q.target_start} to {parsed_q.target_end}")
    c4.metric("Temporal Mode", "Prospective" if parsed_q.is_prospective else ("Current" if parsed_q.is_current else "Historical"))

    st.markdown("---")

    # Run Both Retrievers
    v_results = vanilla_retriever.retrieve(query, top_k=top_k, parser=parser)
    r_results, guardrail_msg = relevance_retriever.retrieve(query, top_k=top_k)

    # Side-by-side display
    col_left, col_right = st.columns(2)

    with col_left:
        st.error("### ❌ Vanilla Semantic RAG (Cosine Sim)")
        st.caption("Matches based solely on text embedding similarity. Time-blind.")

        for i, res in enumerate(v_results):
            valid_badge = "🟢 Valid Overlap" if res.is_temporally_valid else "🔴 EXPIRED / MISMATCH"
            with st.expander(f"#{i+1}: {res.doc_id} ({res.doc_type}) — {valid_badge}", expanded=True):
                st.write(f"**Validity Window:** `{res.valid_from}` ➔ `{res.valid_to}`")
                st.write(f"**Cosine Similarity:** `{res.semantic_score:.4f}`")
                st.info(res.text)

    with col_right:
        st.success("### ✅ Relevance-First RAG (Hybrid + Guardrails)")
        st.caption("Temporal Gating + Intent Routing + $0.4 S_{sem} + 0.6 T_{time}$")

        if guardrail_msg:
            st.warning(f"🛡️ **GUARDRAIL INTERCEPT:**\n\n{guardrail_msg}")
            st.info("The system intercepted an empty future candidate set and safely informed the user rather than hallucinating expired historical plans.")
        else:
            for i, res in enumerate(r_results):
                valid_badge = "🟢 Valid Overlap" if res.is_temporally_valid else "🔴 Out of Range"
                with st.expander(f"#{i+1}: {res.doc_id} ({res.doc_type}) — {valid_badge}", expanded=True):
                    st.write(f"**Validity Window:** `{res.valid_from}` ➔ `{res.valid_to}`")
                    st.write(
                        f"**Score Breakdown:** Semantic: `{res.semantic_score:.4f}` | "
                        f"Temporal: `{res.temporal_score:.4f}` | **Final:** `{res.final_score:.4f}`"
                    )
                    st.info(res.text)

    # Final Answer Generation with Citations
    st.markdown("---")
    st.subheader("📝 Synthesized Executive Answer (with Citations)")
    if guardrail_msg:
        st.markdown(f"**Agent Response:**\n\n> 🛡️ **Guardrail Intercept:** {guardrail_msg}")
    elif not any(res.is_temporally_valid for res in r_results):
        latest_date = r_results[0].valid_to if r_results else "N/A"
        st.markdown(
            f"**Agent Response:**\n\n"
            f"> 🛡️ **Temporal Guardrail Intercept:** No valid {parsed_q.intent} records found for requested horizon "
            f"**({parsed_q.target_start} to {parsed_q.target_end})**.\n"
            f"> The latest recorded data in the corpus expired on `{latest_date}`. "
            f"Per Relevance-First mandate, expired records are withheld to prevent executive misinformation."
        )
    else:
        st.markdown("**Agent Response:**")
        summary_lines = []
        for i, res in enumerate(r_results):
            if res.is_temporally_valid:
                summary_lines.append(f"- {res.text} *[Source: {res.doc_id}, valid {res.valid_from} to {res.valid_to}]*")
        st.markdown("\n".join(summary_lines))
