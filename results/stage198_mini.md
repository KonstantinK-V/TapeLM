# Stage198 — streaming / beyond-window / update-heavy under budget

**Overall:** `STREAM_ARCHITECTURAL_DISTINCT`

- tape (gated, in-space): **0.738**
- rag_uniform (ingestion order): 0.250
- rag_novelty (RAG + bolted fp-surprise): 0.726
- gpt_incontext (beyond-window): 0.226  (chance 0.25)
- tape latest-value on updated entities: 0.778

budget=135, stream=675, updated=45
gates: {'g_beyond': True, 'g_budget': True, 'g_update': True}