# Stage 286 the tape as its own label

**EVIDENCE_NO**

| leave-one-out | coverage | accuracy | reward |
|---|---:|---:|---:|
| model | 0.73 | 0.35 | -0.011 |
| votes | 0.83 | 0.43 | 0.014 |
| return | 0.13 | 0.89 | 0.754 |

- unconditional silence is worth 0.75; answering pays only above accuracy 0.875
- at votes's coverage 0.83: model 0.33 vs judge 0.43
- at return's coverage 0.13: model 0.67 vs judge 0.89
- control: unknown AUC 0.996 on the last training tape against 0.302 held out; the pair says whether a failure is shift between the tapes or not a function of the features at all
- contested evidence only (45 sets): model answers 0.82 of them at 0.19, against 0.56 at 0.79 on unanimous ones; at votes' contested coverage 0.73 model 0.15 vs votes 0.12
- P(UNKNOWN) separates absent from present at AUC **0.302** (-2.28 sigma) (0.075 vs 0.297); at the argmax 0.071 vs 0.321
- training saw 2.07 candidates on average, 0.43 of examples unanimous
- exam is a consistency check only: oracle 0.932, model 0.257, gap 0.676

## Gates

- G_arc_enc_frozen: **True**
- G_task_exists: **True**
- G_judges_non_vacuous: **True**
- G_learns_evidence: **False**
- G_abstains_unknowable: **False**
- G_survives_lie: **False**
- G_weighs_contested: **True**
- G_beats_silence: **False**
