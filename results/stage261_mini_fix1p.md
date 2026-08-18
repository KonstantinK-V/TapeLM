# Stage 261 natural-question retrieval (fix1p)

**NL_QUERY_NWAY_FP_ONLY** slots=4353 eval=177 chance=0.0002

- top1: fp-only **0.000** -> fp+sem **0.006** (shuffled 0.000)
- by overlap: low **0.000** vs high **0.011** (median 0.062)
- 20-way (chance 0.05): fp-only **0.226** (frozen Wq) / trained Wq **0.102** -> fp+sem **0.113** (shuffled 0.051)
- mrr 0.016, median rank 1684, blend a 0.350
- matched GPT-2: not run
