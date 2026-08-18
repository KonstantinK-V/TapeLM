# Stage 261 natural-question retrieval (fix1)

**NL_QUERY_NWAY_ONLY** slots=4353 eval=177 chance=0.0002

- top1: fp-only **0.034** -> fp+sem **0.006** (shuffled 0.000)
- by overlap: low **0.000** vs high **0.011** (median 0.062)
- 20-way (chance 0.05): fp-only **0.209** -> fp+sem **0.107** (shuffled 0.040)
- mrr 0.020, median rank 1860, blend a 0.691
- matched GPT-2: not run
