# Stage 269 open novel tape

**OPEN_NOVEL_TAPE_NO** · mode=upper · tapes=4 · open=20/tape · slots≈182 · SMOKE

| exam (open half) | top1 | median rank |
|---|---:|---:|
| train tape, trained query | 0.350 | 20 |
| **novel tape, trained query** | **0.250** | 44 |
| novel tape, zero-train votes | 0.450 | 2 |
| novel tape, shuffled keys | 0.000 | 122 |

## Gates (read G_headroom first)

- G_headroom: **True**
- G_novel_tape: **False**
- G_beats_votes: **False**
- G_arc_enc_frozen: **True**
- G_no_param_leak: **True**
- G_tape_causal: **True**
- G_lang_intact: **True**

- planted half EM 0.750, empty tape 0.000
- hold CE 4.139 → 4.179
