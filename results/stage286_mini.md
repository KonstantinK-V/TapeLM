# Stage 286 the tape as its own label

**EVIDENCE_NO**

| leave-one-out | coverage | accuracy | reward |
|---|---:|---:|---:|
| model | 0.57 | 0.33 | 0.121 |
| votes | 0.83 | 0.43 | 0.014 |
| return | 0.13 | 0.89 | 0.754 |

- unconditional silence is worth 0.75; answering pays only above accuracy 0.875
- at votes's coverage 0.83: model 0.38 vs judge 0.43
- at return's coverage 0.13: model 0.22 vs judge 0.89
- P(UNKNOWN) separates absent from present at AUC **0.259** (0.143 vs 0.456); at the argmax 0.143 vs 0.500
- training saw 1.95 candidates on average, 0.47 of examples unanimous
- exam is a consistency check only: oracle 0.932, model 0.588, gap 0.345

## Gates

- G_arc_enc_frozen: **True**
- G_task_exists: **True**
- G_judges_non_vacuous: **True**
- G_learns_evidence: **False**
- G_abstains_unknowable: **False**
- G_survives_lie: **False**
- G_beats_silence: **False**
