# Stage202 — B capability via PAWS (frozen encoder + attention head)

**Overall:** `SEM_B_NO`

- curve: PAWS test acc **0.644** | 179 para 0.359 / hard 0.758 (**inversion=False**)
- gpt baseline: PAWS acc 0.655 | para 0.401 / hard 0.728 (inv=False)
- lexical-overlap baseline: 0.558 (PAWS adversarial ~chance)

gates: {'g_paws': False, 'g_inversion': False, 'g_vs_gpt': True}

Non-destructive: P1 frozen; head is a separate branch. Confirms whether the curve representation CAN encode meaning over spelling given a meaning signal.