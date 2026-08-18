# Stage 261 natural-question retrieval (tape_rerank_val)

**NL_QUERY_NO_AT_SCALE** slots=4353 eval=177 chance=0.0002

- top1: fp-only **0.006** -> fp+sem **0.000** (shuffled 0.000)
- by overlap: low **0.000** vs high **0.000** (median 0.062)
- 20-way (chance 0.05): fp-only **0.096** (init Wq 0.096) -> fp+sem **0.096** (shuffled 0.056)
- mrr 0.005, median rank 2044, blend a 0.000
- matched GPT-2: not run
