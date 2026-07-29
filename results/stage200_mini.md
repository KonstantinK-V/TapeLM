# Stage200 — fact composition: operate vs read-by-index

**Overall:** `COMPOSE_CHAINS_BUT_RAG_PARITY`

- curve_string k1/k2/k3: 1.00 / 1.00 / 1.00
- curve_vector (no decode) k1/k2/k3: 1.00 / 1.00 / 1.00
- rag_index k1/k2/k3: 1.00 / 1.00 / 1.00
- binding one-shot 2-hop: 1.000 (chance 0.25)
- gpt_incontext 2-hop: 0.200

slots=6180 chains=60 K=3
gates: {'g_external': True, 'g_chain': True, 'g_vs_rag': False, 'g_vectornative': True}