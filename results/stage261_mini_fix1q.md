# Stage 261 natural-question retrieval (fix1q)

**NL_QUERY_NWAY_ONLY** slots=4353 eval=177 chance=0.0002

- top1: fp-only **0.006** -> fp+sem **0.011** (shuffled 0.000)
- by overlap: low **0.000** vs high **0.023** (median 0.062)
- 20-way (chance 0.05): fp-only **0.102** (frozen Wq) / trained Wq **0.102** -> fp+sem **0.124** (shuffled 0.068)
- mrr 0.018, median rank 1779, blend a 0.349
- matched GPT-2: not run
