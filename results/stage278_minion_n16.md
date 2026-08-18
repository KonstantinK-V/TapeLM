# Stage 278 value baseline, honest ceiling, real abstain cost

**JUDGE_OK** · value head **True** · reward wrong -1.0 / abstain +0.75

| family (novel) | coverage | acc answered | abstain | teacher abstain | reads |
|---|---:|---:|---:|---:|---:|
| clean | 1.00 | 1.00 | 0.00 | 0.00 | 1.0 |
| decidable | 1.00 | 1.00 | 0.00 | 0.00 | 5.0 |
| tie | 0.00 | nan | 1.00 | 1.00 | 4.0 |

- overall coverage 0.67 at accuracy 1.00 (276 with filter on: 1.00 at 0.67)
- reward: policy 0.850 vs teacher 0.850
- tie abstain **1.00** (276: 0.00), teacher **1.00** (276: 0.50)

## Gates

- G_arc_enc_frozen: **True**
- G_answer_is_slot: **True**
- G_retrieval_usable: **True**
- G_teacher_ceiling: **True**
- G_answers_when_decidable: **True**
- G_abstain_on_tie: **True**
- G_acc_when_answering: **True**
- G_beats_always_answer: **True**
- G_novel_tape: **True**
- G_reaches_teacher: **True**
