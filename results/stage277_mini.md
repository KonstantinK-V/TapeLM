# Stage 277 ink revival: votes as the tau->1 limit

**INK_NO** slots=4338 vocab=11195 eval=338 tau=0.6 trained params **0**

| arm | top1 | median rank | 20-way strict |
|---|---|---|---|
| votes (incumbent) | 0.246 | 328 | 0.417 |
| sum (idf-weighted) | 0.139 | 1070 | 0.251 |
| sum + all-but-top | 0.130 | 1058 | 0.251 |
| **maxsim** | **0.198** | 713 | 0.379 |
| hybrid (a=0.25) | 0.237 | 398 | 0.432 |

- kernel reduces to votes at tau=0.999: **True** (maxsim 0.243 vs votes 0.246)
- votes silent on gold: **165/338** (0.488) -> near-string 103, purely semantic 62
- on the near-string half maxsim top1 0.000, top10 0.000; on the semantic half top10 0.016
- popularity floor: votes 0.000, maxsim 0.000
- typo 0.15: votes 0.243 -> maxsim 0.180, hybrid 0.201
