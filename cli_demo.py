"""
cli_demo.py
-----------
Command-line interactive interface for the Relevance-First PMO RAG Agent.
Allows testing queries in real time with detailed score breakdown and citations.
"""

import sys
import datetime
from tabulate import tabulate
from temporal_parser import TemporalQueryParser
from ingestion import PMODataIngestion, EmbeddingIndexer
from retriever import VanillaRetriever, RelevanceFirstRetriever


def main():
    print("=" * 70)
    print("  PMO AGENT: RELEVANCE-FIRST RAG (CLI DEMO)")
    print("  Anchor Date (T_ref): 2026-10-02")
    print("=" * 70)

    ingestor = PMODataIngestion()
    docs = ingestor.load_corpus(include_tasks=True)
    indexer = EmbeddingIndexer()
    parser = TemporalQueryParser(ref_date=datetime.date(2026, 10, 2))
    vanilla_retriever = VanillaRetriever(docs, indexer)
    relevance_retriever = RelevanceFirstRetriever(docs, indexer, parser=parser)

    presets = [
        "What is the plan for next week for Atlas?",
        "Current status of Orion project?",
        "Who is blocked right now in Phoenix?",
        "What are the risks for Orion?",
        "Who is available next week?",
    ]

    while True:
        print("\nPreset Queries:")
        for idx, p in enumerate(presets, 1):
            print(f"  [{idx}] {p}")
        print("  [C] Custom Query")
        print("  [Q] Quit")

        choice = input("\nSelect an option: ").strip().lower()
        if choice == "q":
            break
        elif choice == "c":
            query = input("Enter custom query: ").strip()
        elif choice.isdigit() and 1 <= int(choice) <= len(presets):
            query = presets[int(choice) - 1]
        else:
            print("Invalid selection. Try again.")
            continue

        print(f"\n>> QUERY: \"{query}\"")
        parsed = parser.parse(query)
        print(f"   Project: {parsed.project_name or 'All'} | Intent: {parsed.intent} | Target: {parsed.target_start} to {parsed.target_end}")

        v_res = vanilla_retriever.retrieve(query, top_k=3, parser=parser)
        r_res, guardrail = relevance_retriever.retrieve(query, top_k=3)

        print("\n--- [VANILLA RAG (Cosine Sim)] ---")
        v_rows = []
        for r in v_res:
            v_rows.append([r.doc_id, r.doc_type, f"{r.valid_from} -> {r.valid_to}", f"{r.semantic_score:.4f}", "VALID" if r.is_temporally_valid else "EXPIRED"])
        print(tabulate(v_rows, headers=["Doc ID", "Type", "Window", "Cosine Sim", "Temporal Match"], tablefmt="simple"))

        print("\n--- [RELEVANCE-FIRST RAG (Hybrid + Guardrails)] ---")
        if guardrail:
            print(f"  [GUARDRAIL TRIGGERED]: {guardrail}")
        else:
            r_rows = []
            for r in r_res:
                r_rows.append([r.doc_id, r.doc_type, f"{r.valid_from} -> {r.valid_to}", f"{r.semantic_score:.4f}", f"{r.temporal_score:.4f}", f"{r.final_score:.4f}", "VALID" if r.is_temporally_valid else "OUT-OF-BOUNDS"])
            print(tabulate(r_rows, headers=["Doc ID", "Type", "Window", "Semantic", "Temporal", "Final", "Valid"], tablefmt="simple"))

            print("\n>> Synthesized Answer:")
            for r in r_res:
                print(f"   * {r.text} [{r.doc_id}]")


if __name__ == "__main__":
    main()
