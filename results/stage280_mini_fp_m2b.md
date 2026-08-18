# Stage 280 the exam on raw text (hop fp)

**RAW_EXAM_PARTIAL** · teacher ceiling **0.472**

| family (held out) | n | coverage | acc answered | abstain | teacher acc | reads |
|---|---:|---:|---:|---:|---:|---:|
| clean | 29 | 0.83 | 0.96 | 0.17 | 0.79 | 4.9 |
| decidable | 8 | 1.00 | 0.75 | 0.00 | 0.75 | 5.8 |
| tie | 20 | 0.75 | 0.00 | 0.25 | 0.00 | 4.6 |

- policy 0.227 vs teacher 0.472; coverage 0.82 at accuracy 0.62
- words silent on 0.07 of episodes, 0.07 ink hops per episode
- retrieval precision 0.59, recall 0.95

## Gates

- G_arc_enc_frozen: **True**
- G_answer_is_slot: **True**
- G_all_families_present: **True**
- G_teacher_usable: **True**
- G_reaches_teacher: **False**
- G_answers_when_decidable: **True**
- G_abstain_on_tie: **False**
- G_acc_when_answering: **True**
- G_generalises_to_held_out: **False**
- G_hop_covers_silence: **True**
