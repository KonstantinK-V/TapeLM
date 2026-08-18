# Stage 271 controller (frozen trunk)

**CONTROLLER_PARTIAL** · episodes=6000 · actions=11

| arm | clean | lying |
|---|---:|---:|
| policy (novel tape) | **1.000** | **0.500** |
| fixed lookup | 1.000 | 0.667 |
| fixed majority | — | 0.667 |

- mean reads 1.00 of 6
- train tape lying 0.833 → novel 0.500

## Gates

- G_arc_enc_frozen: **True**
- G_answer_is_slot: **True**
- G_beats_lookup: **False**
- G_beats_majority: **False**
- G_clean_kept: **True**
- G_novel_tape: **False**
- G_reads_economical: **True**
