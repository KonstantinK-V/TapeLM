# Stage197 — knowledge edit (anchored write + surprise-gated policy)

**Overall:** `EDIT_CLEAN_WIN`  (best mech: `M3_blend` = 1.000)

acquisition (4-way, chance 0.25):
- M1_meanctx: 0.460
- M2_subject: 0.970
- M3_blend: 1.000
- M4_multikey: 0.960
- gpt_parametric: 0.280  |  gpt+rag: 0.440

selective write (budget=100, novel=100): gated kept 0.980 vs uniform 0.200

gates: {'g_acquire': True, 'g_vs_rag': True, 'g_select': True}