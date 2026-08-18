# Stage 273 read-must-inform

**READ_INFORM_PARTIAL** · bc=4000 rl=0 · actions=11

| arm | clean | lying |
|---|---:|---:|
| policy (novel tape) | **1.000** | **0.667** |
| fixed lookup | 1.000 | 0.667 |
| fixed majority | — | 0.833 |

- mean reads 0.50 (clean 0.50 / lying 0.50)
- train tape lying 1.000 → novel 0.667

## Gates

- G_arc_enc_frozen: **True**
- G_answer_is_slot: **True**
- G_beats_lookup: **False**
- G_beats_majority: **False**
- G_clean_kept: **True**
- G_novel_tape: **False**
- G_reads_economical: **True**
- G_read_informed: **True**
