# Stage 261 natural-question retrieval (baseline)

**NL_QUERY_NO_AT_SCALE** slots=4353 eval=177 chance=0.0002

- top1: fp-only **0.023** -> fp+sem **0.000** (shuffled 0.000)
- by overlap: low **0.000** vs high **0.000** (median 0.062)
- 20-way (chance 0.05): fp-only **0.226** (init Wq 0.282) -> fp+sem **0.119** (shuffled 0.040)
- mrr 0.007, median rank 1536, blend a 0.704
- matched GPT-2: not run
