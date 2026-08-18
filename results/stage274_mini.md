# Stage 274 truth-free oracle

**TEACHER_NO_BETTER_THAN_LOOKUP** · bc=400 rl=0 · 5 witnesses, 2 lying · SMOKE

| arm (novel tape) | clean | lying |
|---|---:|---:|
| policy | **1.000** | **0.667** |
| teacher (executable) | 1.000 | 0.500 |
| fixed lookup | 1.000 | 0.500 |
| fixed majority | — | 0.667 |

- reads: clean 1.00, lying 1.00
- train lying 0.833 → novel 0.667

## Gates

- G_arc_enc_frozen: **True**
- G_answer_is_slot: **True**
- G_teacher_useful: **False**
- G_policy_matches_teacher: **True**
- G_beats_lookup: **True**
- G_clean_kept: **True**
- G_novel_tape: **False**
- G_reads_informed: **True**
