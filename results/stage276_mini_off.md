# Stage 276 search in loop (--subject-filter off)

**RETRIEVAL_UNUSABLE** · retrieval precision 0.48, witness recall 1.00

| family (novel) | coverage | acc answered | abstain | precision | reads |
|---|---:|---:|---:|---:|---:|
| clean | 1.00 | 1.00 | 0.00 | 0.14 | 0.0 |
| decidable | 1.00 | 0.50 | 0.00 | 0.71 | 0.0 |
| tie | 1.00 | 0.00 | 0.00 | 0.57 | 0.0 |

- overall coverage 1.00 at accuracy 0.50
- reward: policy 0.350 vs teacher 0.060

## Gates

- G_arc_enc_frozen: **True**
- G_answer_is_slot: **True**
- G_retrieval_usable: **False**
- G_answers_when_decidable: **True**
- G_teacher_abstains_on_tie: **False**
- G_abstain_on_tie: **False**
- G_acc_when_answering: **False**
- G_beats_always_answer: **True**
- G_novel_tape: **True**
