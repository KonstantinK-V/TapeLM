# Stage 261 natural-question retrieval

**NL_QUERY_NO_AT_SCALE** slots=4353 eval=177 chance=0.0002

**Headline:** GPT and tape **0** open-domain top1 — scale verdict OK.

**Finding:** 20-way shows **fp signal**; **blend breaks it** (see [`stage261_close.md`](stage261_close.md)).

- top1: fp-only **0.034** -> fp+sem **0.000** (shuffled 0.000)
- 20-way (chance 0.05): fp-only **0.220** (~4.4×) -> fp+sem **0.090** (shuffled **0.062**)
- blend α **0.720** — sem channel given ~72% weight; 20-way **halved** vs fp-only
- by overlap: low **0.000** vs high **0.000** (median 0.062)
- matched GPT-2: top1 0.000
