# PMO Temporal RAG Benchmark Results

| Metric                    | Vanilla RAG   | Relevance-First RAG   | Improvement   |
|---------------------------|---------------|-----------------------|---------------|
| Temporal Hit Rate (%)     | 13.33%        | 92.31%                | +79.0%        |
| Relevance Precision@3 (%) | 13.33%        | 92.31%                | +79.0%        |
| Hallucination Rate (%)    | 0.0%          | 0.0%                  | -0.0%         |

### Diagnosis Highlights
- **Vanilla RAG failure on future plan queries**: Returns expired historical sprint plans due to time-blind cosine similarity.
- **Relevance-First Guardrail intercept**: Detects empty future plan sets and alerts the user rather than hallucinating expired plans.
- **Temporal Hit Rate improved from 13.3% to 92.3%** across the benchmark query suite.
