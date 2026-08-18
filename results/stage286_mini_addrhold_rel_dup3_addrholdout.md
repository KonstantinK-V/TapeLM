# Stage 286 the tape as its own label

**EVIDENCE_PARTIAL**

| leave-one-out | coverage | accuracy | reward |
|---|---:|---:|---:|
| model | 0.68 | 0.57 | 0.335 |
| votes | 0.61 | 0.68 | 0.512 |
| return | 0.24 | 0.70 | 0.665 |

- unconditional silence is worth 0.75; answering pays only above accuracy 0.875
- at votes's coverage 0.61: model 0.64 vs judge 0.68
- at return's coverage 0.24: model 0.50 vs judge 0.70
- control: unknown AUC 0.896 on the last training tape against 0.754 held out; the pair says whether a failure is shift between the tapes or not a function of the features at all
- contested evidence only (23 sets): model answers 0.43 of them at 0.20, against 1.00 at 0.78 on unanimous ones; at votes' contested coverage 0.30 model 0.29 vs votes 0.43
- UNKNOWN's margin over the best candidate separates absent from present at AUC **0.754** (+2.64 sigma); the diluted probability form says 0.754 (0.335 vs 0.119); at the argmax 0.429 vs 0.259
- training saw 2.47 candidates on average, 0.46 of examples unanimous
- exam is a consistency check only: oracle 0.918, model 0.510, gap 0.409

## Gates

- G_arc_enc_frozen: **True**
- G_task_exists: **True**
- G_judges_non_vacuous: **True**
- G_learns_evidence: **False**
- G_abstains_unknowable: **True**
- G_survives_lie: **True**
- G_weighs_contested: **False**
- G_survives_duplicated_lie: **True**
- G_beats_silence: **False**
