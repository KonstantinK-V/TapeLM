# Stage 280 the exam on raw text (hop fp)

**TEACHER_UNUSABLE_ON_RAW** · teacher ceiling **0.165**

| family (held out) | n | coverage | acc answered | abstain | teacher acc | reads |
|---|---:|---:|---:|---:|---:|---:|
| clean | 29 | 1.00 | 0.86 | 0.00 | 0.79 | 4.9 |
| decidable | 8 | 1.00 | 0.75 | 0.00 | 0.75 | 5.8 |
| tie | 20 | 1.00 | 0.00 | 0.00 | 0.00 | 4.6 |

- policy -0.010 vs teacher 0.165; coverage 1.00 at accuracy 0.54
- words silent on 0.07 of episodes, 0.07 ink hops per episode
- retrieval precision 0.59, recall 0.95

## Gates

- G_arc_enc_frozen: **True**
- G_answer_is_slot: **True**
- G_all_families_present: **True**
- G_teacher_usable: **False**
- G_reaches_teacher: **False**
- G_answers_when_decidable: **True**
- G_abstain_on_tie: **False**
- G_acc_when_answering: **False**
- G_generalises_to_held_out: **True**
- G_hop_covers_silence: **True**
