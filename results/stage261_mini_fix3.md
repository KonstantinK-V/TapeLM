# Stage 261 natural-question retrieval (fix3)

**NL_QUERY_NO** slots=4353 eval=177 chance=0.0002

- top1: fp-only **0.017** -> fp+sem **0.006** (shuffled 0.000)
- by overlap: low **0.000** vs high **0.011** (median 0.062)
- 20-way (chance 0.05): fp-only **0.175** -> fp+sem **0.090** (shuffled 0.056)
- mrr 0.008, median rank 2163, blend a 0.664
- matched GPT-2: not run
