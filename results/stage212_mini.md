# Stage212 — instance / content-invariant channel

**Overall:** `THESIS_NO_AT_SCALE` (legacy label: THESIS_NO)

- T1 collision (4-way, chance 0.25): instance **0.378** | untrained 0.336 | fp_only 0.242 | ctx_blend 0.328 | soft_rerank 0.320
- T2 para=0.750 hard=0.937 inversion=False
- next_tok 0.82 -> 0.82 (delta 0.0), anti-CF True
- gates: {'g1_collision': False, 'g2_invariance': False, 'g3_learning_matters': False, 'g4_beats_soft_rerank': False, 'g5_no_ce_cost': True, 'g6_beats_ctx_blend': False}