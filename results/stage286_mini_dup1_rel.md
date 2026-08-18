# Stage 286 the tape as its own label

**EVIDENCE_PARTIAL**

| leave-one-out | coverage | accuracy | reward |
|---|---:|---:|---:|
| model | 0.39 | 0.78 | 0.675 |
| votes | 0.83 | 0.43 | 0.014 |
| return | 0.13 | 0.89 | 0.754 |

- unconditional silence is worth 0.75; answering pays only above accuracy 0.875
- at votes's coverage 0.83: model 0.43 vs judge 0.43
- at return's coverage 0.13: model 1.00 vs judge 0.89
- control: unknown AUC 0.841 on the last training tape against 0.482 held out; the pair says whether a failure is shift between the tapes or not a function of the features at all
- contested evidence only (45 sets): model answers 0.04 of them at 0.00, against 1.00 at 0.84 on unanimous ones; at votes' contested coverage 0.73 model 0.12 vs votes 0.12
- UNKNOWN's margin over the best candidate separates absent from present at AUC **0.486** (-0.16 sigma); the diluted probability form says 0.482 (0.478 vs 0.484); at the argmax 0.714 vs 0.589
- training saw 2.07 candidates on average, 0.43 of examples unanimous
- exam is a consistency check only: oracle 0.932, model 0.851, gap 0.081

## Gates

- G_arc_enc_frozen: **True**
- G_task_exists: **True**
- G_judges_non_vacuous: **True**
- G_learns_evidence: **True**
- G_abstains_unknowable: **False**
- G_survives_lie: **True**
- G_weighs_contested: **False**
- G_survives_duplicated_lie: **None**
- G_beats_silence: **False**
