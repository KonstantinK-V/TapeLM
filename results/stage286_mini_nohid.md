# Stage 286 the tape as its own label (no hidden)

**EVIDENCE_PARTIAL**

| leave-one-out | coverage | accuracy | reward |
|---|---:|---:|---:|
| model | 0.56 | 0.64 | 0.489 |
| votes | 0.83 | 0.43 | 0.014 |
| return | 0.13 | 0.89 | 0.754 |

- unconditional silence is worth 0.75; answering pays only above accuracy 0.875
- at votes's coverage 0.83: model 0.43 vs judge 0.43
- at return's coverage 0.13: model 1.00 vs judge 0.89
- P(UNKNOWN) separates absent from present at AUC **0.412** (0.229 vs 0.294); at the argmax 0.714 vs 0.375
- training saw 1.95 candidates on average, 0.47 of examples unanimous
- exam is a consistency check only: oracle 0.932, model 0.824, gap 0.108

## Gates

- G_arc_enc_frozen: **True**
- G_task_exists: **True**
- G_judges_non_vacuous: **True**
- G_learns_evidence: **True**
- G_abstains_unknowable: **False**
- G_survives_lie: **True**
- G_beats_silence: **False**
