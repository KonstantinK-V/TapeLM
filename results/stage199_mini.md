# Stage199 — semantic invariance (CPC on frozen encoder)

**Overall:** `SEM_INV_PARTIAL`

- raw frozen baseline: para 0.705 / hard 0.889 (gap +0.185)
- scale trend (para / hard / gap hard−para):
  - budget 0.05: 0.648 / 0.820 / +0.172 (inv=False)
  - budget 0.25: 0.704 / 0.877 / +0.173 (inv=False)
  - budget 1.00: 0.714 / 0.877 / +0.163 (inv=False)

- non-destruct: next_tok(frozen) = 0.825 (generation/memory/calib untouched)
- gates: {'g_nondestruct': True, 'g_scale_monotone': False, 'inversion_at_full': False, 'head_beats_raw': True}