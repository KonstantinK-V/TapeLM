# Stage 276 search in loop (--subject-filter on)

**TEACHER_CANNOT_ABSTAIN** · retrieval precision 1.00, witness recall 1.00

| family (novel) | coverage | acc answered | abstain | precision | reads |
|---|---:|---:|---:|---:|---:|
| clean | 1.00 | 1.00 | 0.00 | 1.00 | 1.0 |
| decidable | 1.00 | 1.00 | 0.00 | 1.00 | 5.0 |
| tie | 1.00 | 0.00 | 0.00 | 1.00 | 4.0 |

- overall coverage 1.00 at accuracy 0.67
- reward: policy 0.500 vs teacher 0.563

## Gates

- G_arc_enc_frozen: **True**
- G_answer_is_slot: **True**
- G_retrieval_usable: **True**
- G_answers_when_decidable: **True**
- G_teacher_abstains_on_tie: **False**
- G_abstain_on_tie: **False**
- G_acc_when_answering: **False**
- G_beats_always_answer: **True**
- G_novel_tape: **True**
