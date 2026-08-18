# Stage 268 mind learns tape

**MIND_LEARNS_TAPE_PARTIAL** · steps=8000 · tapes=40 · bank≈1200+facts

| exam | EM | frozen-upper EM |
|------|---:|----------------:|
| last train tape | 1.000 | — |
| novel tape | **1.000** | 0.000 |

## Gates (read G_novel_tape first)

- G_novel_tape: **True** (novel 1.000 vs train 1.000)
- G_arc_enc_frozen: **True**
- G_beats_frozen_mind: **True**
- G_no_param_leak: **True** (empty=0.000)
- G_slot_delete: **True**
- G_lang_intact: **False** (hold 3.905→4.040)
