# Stage 281 what counts as an assertion

**NO_FRAME_SURVIVES** · trained parameters **0**

- frames 2/18 kept, assertions 18/240 (7.5%)
- teacher ceiling **0.483 -> nan** (silence pays 0.750; 280 measured -0.189)
- tie abstention by the teacher: 0.00 -> nan

| frame | n | confirm | anchors | values/anchor |
|---|---:|---:|---:|---:|
| `lder` | 6 | 0.83 | 1 | 1.00 |
| `mystery television series` | 4 | 0.75 | 1 | 1.00 |
| `entered` | 4 | 0.75 | 1 | 1.00 |
| `tico` | 7 | 0.71 | 1 | 2.00 |
| `was` | 12 | 0.33 | 4 | 2.00 |
| `and the` | 6 | 0.33 | 2 | 2.00 |
| `and` | 87 | 0.26 | 25 | 2.56 |
| `ois` | 5 | 0.20 | 1 | 4.00 |
| `the` | 81 | 0.16 | 19 | 3.58 |
| `near` | 4 | 0.00 | 1 | 4.00 |

## Gates

- G_frames_survive: **True**
- G_tape_shrinks: **True**
- G_kept_frames_functional: **True**
- G_ceiling_clears_silence: **False**
- G_ceiling_improves: **False**
- G_teacher_abstains_on_tie: **False**
