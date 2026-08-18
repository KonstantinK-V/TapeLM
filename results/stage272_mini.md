# Stage 272 BC controller

**CONTROLLER_PARTIAL** · bc=800 rl=0 · actions=11 · SMOKE

| arm | clean | lying |
|---|---:|---:|
| policy (novel tape) | **1.000** | **0.333** |
| fixed lookup | 1.000 | 0.333 |
| fixed majority | — | 0.333 |

- mean reads 0.92 of 6 (clean 0.83 / lying 1.00)
- train tape lying 0.667 → novel 0.333

## Gates

- G_arc_enc_frozen: **True**
- G_answer_is_slot: **True**
- G_beats_lookup: **False**
- G_beats_majority: **True**
- G_clean_kept: **True**
- G_novel_tape: **False**
- G_reads_economical: **True**
