# Stage 275 abstain

**ABSTAIN_PARTIAL** · SMOKE

| family (novel) | coverage | acc answered | abstain | teacher abstain | reads |
|---|---:|---:|---:|---:|---:|
| clean | 1.00 | 1.00 | 0.00 | 0.00 | 0.5 |
| decidable | 1.00 | 0.50 | 0.00 | 0.00 | 1.2 |
| tie | 0.25 | 0.00 | 0.75 | 1.00 | 3.0 |

- overall coverage 0.75 at accuracy 0.67
- reward: policy 0.393 vs teacher 0.508

## Gates

- G_arc_enc_frozen: **True**
- G_answer_is_slot: **True**
- G_answers_when_decidable: **True**
- G_teacher_abstains_on_tie: **True**
- G_abstain_on_tie: **True**
- G_acc_when_answering: **False**
- G_beats_always_answer: **True**
- G_novel_tape: **True**
