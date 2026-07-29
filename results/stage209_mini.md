# Stage209 — scaling + teacher probe

**Overall:** `STRUCTURAL_BLOCK_NO`

| scale | curve PAWS | gpt PAWS | Δ |
|-------|------------|----------|---|
| d128_L2 | 0.596 | 0.580 | +0.016 |
| d192_L4 | 0.602 | 0.624 | -0.022 |
| d256_L6 | 0.632 | 0.640 | -0.008 |

Teacher Pearson r @ d256: curve **0.256** vs gpt 0.270
gates: mono=True parity=True teacher=True signal=True