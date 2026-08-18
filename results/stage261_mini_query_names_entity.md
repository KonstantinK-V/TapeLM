# Stage 261 natural-question retrieval (baseline)

**NL_QUERY_NO_AT_SCALE** slots=4353 eval=177 chance=0.0002

- top1: fp-only **0.000** -> fp+sem **0.000** (shuffled 0.000)
- by overlap: low **0.000** vs high **0.000** (median 0.062)
- 20-way (chance 0.05): fp-only **0.062** (init Wq 0.090) -> fp+sem **0.068** (shuffled 0.023)
- mrr 0.002, median rank 1959, blend a 0.979
- matched GPT-2: not run
