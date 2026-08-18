# Stage 271 controller

**CONTROLLER_NO** · episodes=6000 · actions=11

| arm | clean | lying |
|---|---:|---:|
| policy (novel tape) | **0.667** | **0.667** |
| fixed lookup | 1.000 | 0.667 |
| fixed majority | — | 0.667 |

- mean reads 0.00 of 6
- train tape lying 0.833 → novel 0.667

## Gates

- G_arc_enc_frozen: **True**
- G_answer_is_slot: **True**
- G_beats_lookup: **False**
- G_beats_majority: **True**
- G_clean_kept: **False**
- G_novel_tape: **False**
- G_reads_economical: **True**
