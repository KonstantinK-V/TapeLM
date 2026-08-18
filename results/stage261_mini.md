# Stage 261 natural-question retrieval (baseline)

**NL_QUERY_NO** slots=4353 eval=177 chance=0.0002

- top1: fp-only **0.085** -> fp+sem **0.000** (shuffled 0.000)
- by overlap: low **0.000** vs high **0.000** (median 0.062)
- 20-way (chance 0.05): fp-only **0.209** (init Wq 0.220) -> fp+sem **0.090** (shuffled 0.034)
- mrr 0.006, median rank 1824, blend a 0.933
- matched GPT-2: not run
