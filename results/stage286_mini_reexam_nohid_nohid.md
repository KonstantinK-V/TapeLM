# Stage 286 the tape as its own label (no hidden)

**EVIDENCE_OK**

| leave-one-out | coverage | accuracy | reward |
|---|---:|---:|---:|
| model | 0.36 | 0.84 | 0.725 |
| votes | 0.83 | 0.43 | 0.014 |
| return | 0.13 | 0.89 | 0.754 |

- unconditional silence is worth 0.75; answering pays only above accuracy 0.875
- at votes's coverage 0.83: model 0.52 vs judge 0.43
- at return's coverage 0.13: model 1.00 vs judge 0.89
- control: unknown AUC 0.809 on the last training tape against 0.560 held out; the pair says whether a failure is shift between the tapes or not a function of the features at all
- P(UNKNOWN) separates absent from present at AUC **0.560** (0.655 vs 0.549); at the argmax 0.714 vs 0.625
- training saw 2.07 candidates on average, 0.43 of examples unanimous
- exam is a consistency check only: oracle 0.932, model 0.892, gap 0.041

## Gates

- G_arc_enc_frozen: **True**
- G_task_exists: **True**
- G_judges_non_vacuous: **True**
- G_learns_evidence: **True**
- G_abstains_unknowable: **True**
- G_survives_lie: **True**
- G_beats_silence: **False**
