# Stage 280 the exam on raw text (hop fp)

**RAW_EXAM_OK** · teacher ceiling **0.625**

| family (held out) | n | coverage | acc answered | abstain | teacher acc | reads |
|---|---:|---:|---:|---:|---:|---:|
| clean | 29 | 0.76 | 1.00 | 0.24 | 0.83 | 3.6 |
| decidable | 8 | 0.62 | 1.00 | 0.38 | 0.75 | 4.5 |
| tie | 20 | 0.15 | 0.00 | 0.85 | 0.00 | 3.2 |

- policy 0.704 vs teacher 0.625; coverage 0.53 at accuracy 0.90
- words silent on 0.07 of episodes, 0.07 ink hops per episode
- retrieval precision 0.82, recall 0.95

## Gates

- G_arc_enc_frozen: **True**
- G_answer_is_slot: **True**
- G_all_families_present: **True**
- G_teacher_usable: **True**
- G_reaches_teacher: **True**
- G_answers_when_decidable: **True**
- G_abstain_on_tie: **True**
- G_acc_when_answering: **True**
- G_generalises_to_held_out: **True**
- G_hop_covers_silence: **True**
