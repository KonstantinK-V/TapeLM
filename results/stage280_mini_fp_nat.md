# Stage 280 the exam on raw text (hop fp)

**RAW_EXAM_INVALID** · teacher ceiling **0.367**

| family (held out) | n | coverage | acc answered | abstain | teacher acc | reads |
|---|---:|---:|---:|---:|---:|---:|
| clean | 6 | 1.00 | 1.00 | 0.00 | 1.00 | 5.2 |
| decidable | 7 | 1.00 | 0.71 | 0.00 | 0.71 | 4.9 |
| tie | 2 | 1.00 | 0.00 | 0.00 | 0.00 | 5.0 |

- policy 0.367 vs teacher 0.367; coverage 1.00 at accuracy 0.73
- words silent on 0.07 of episodes, 0.07 ink hops per episode
- retrieval precision 0.90, recall 0.96

## Gates

- G_arc_enc_frozen: **True**
- G_answer_is_slot: **True**
- G_all_families_present: **False**
- G_teacher_usable: **False**
- G_reaches_teacher: **True**
- G_answers_when_decidable: **True**
- G_abstain_on_tie: **False**
- G_acc_when_answering: **True**
- G_generalises_to_held_out: **True**
- G_hop_covers_silence: **True**
