# Stage 261 natural-question retrieval (fix1m)

**NL_QUERY_NWAY_FP_ONLY** slots=4353 eval=177 chance=0.0002

- top1: fp-only **0.006** -> fp+sem **0.006** (shuffled 0.000)
- by overlap: low **0.000** vs high **0.011** (median 0.062)
- 20-way (chance 0.05): fp-only **0.226** (frozen Wq) / trained Wq **0.113** -> fp+sem **0.130** (shuffled 0.068)
- mrr 0.017, median rank 1947, blend a 0.350
- matched GPT-2: not run
