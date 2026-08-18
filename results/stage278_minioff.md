# Stage 278 value baseline, honest ceiling, real abstain cost

**RETRIEVAL_UNUSABLE** · SMOKE · value head **True** · reward wrong -1.0 / abstain +0.1

| family (novel) | coverage | acc answered | abstain | teacher abstain | reads |
|---|---:|---:|---:|---:|---:|
| clean | 1.00 | 0.00 | 0.00 | 0.00 | 7.0 |
| decidable | 0.00 | nan | 1.00 | 0.00 | 7.0 |
| tie | 0.00 | nan | 1.00 | 1.00 | 7.0 |

- overall coverage 0.33 at accuracy 0.00 (276 with filter on: 1.00 at 0.67)
- reward: policy -0.407 vs teacher -0.107
- tie abstain **1.00** (276: 0.00), teacher **1.00** (276: 0.50)

## Gates

- G_arc_enc_frozen: **True**
- G_answer_is_slot: **True**
- G_retrieval_usable: **False**
- G_teacher_ceiling: **False**
- G_answers_when_decidable: **False**
- G_abstain_on_tie: **True**
- G_acc_when_answering: **False**
- G_beats_always_answer: **False**
- G_novel_tape: **False**
- G_reaches_teacher: **False**
