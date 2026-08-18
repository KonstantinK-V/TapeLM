# Stage 280 the exam on raw text (hop none)

**TEACHER_UNUSABLE_ON_RAW** · SMOKE · teacher ceiling **-0.328**

| family (held out) | n | coverage | acc answered | abstain | teacher acc | reads |
|---|---:|---:|---:|---:|---:|---:|
| clean | 12 | 0.67 | 0.62 | 0.33 | 0.67 | 5.4 |
| decidable | 15 | 0.93 | 0.36 | 0.07 | 0.40 | 5.9 |
| tie | 9 | 0.78 | 0.00 | 0.22 | 0.00 | 4.0 |

- policy -0.210 vs teacher -0.328; coverage 0.81 at accuracy 0.34
- words silent on 0.00 of episodes, 0.00 ink hops per episode
- retrieval precision 0.32, recall 0.51

## Gates

- G_arc_enc_frozen: **True**
- G_answer_is_slot: **True**
- G_all_families_present: **True**
- G_teacher_usable: **False**
- G_reaches_teacher: **True**
- G_answers_when_decidable: **False**
- G_abstain_on_tie: **False**
- G_acc_when_answering: **False**
- G_generalises_to_held_out: **False**
- G_hop_covers_silence: **True**
