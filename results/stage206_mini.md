# Stage206 — W5 latent hops under compute budget

**Overall:** `LATENT_HOPS_CHEAPER_BUT_RAG_VECTOR_TIES`

**Clean (compute):**

| k | curve_latent acc/ms/enc | rag_vector acc/ms/enc | rag_text acc/ms/enc |
|---|-------------------------|-----------------------|---------------------|
| 1 | 1.000 / 1.13 / 1.0 | 1.000 / 5.68 / 1.0 | 1.000 / 11.00 / 2.0 |
| 2 | 1.000 / 1.25 / 1.0 | 1.000 / 5.79 / 1.0 | 1.000 / 16.33 / 3.0 |
| 4 | 1.000 / 1.52 / 1.0 | 1.000 / 5.99 / 1.0 | 1.000 / 26.40 / 5.0 |
| 6 | 1.000 / 2.35 / 1.0 | 1.000 / 6.24 / 1.0 | 1.000 / 37.57 / 7.0 |

**Noisy (p=0.15) — compounding across hops, with and without lexicon re-anchoring (snap):**

| k | curve | curve+snap | rag_vector | rag_vector+snap |
|---|-------|-----------|------------|-----------------|
| 1 | 0.650 | **0.700** | 0.450 | 0.450 |
| 2 | 0.517 | **0.617** | 0.358 | 0.350 |
| 4 | 0.333 | **0.392** | 0.283 | 0.242 |
| 6 | 0.300 | **0.350** | 0.292 | 0.300 |

- vanilla GPT in-context (k=2, beyond window): 0.250 (chance 0.25)
- gates: deep=True cheap=True rag_vector_ties_clean=True noise=False noise_with_snap=False snap_helps_curve=False