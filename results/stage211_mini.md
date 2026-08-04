# Stage211 — internal slow tape vs endpoint (cross-doc)

**Overall:** `THESIS_NO_AT_SCALE` (legacy label: THESIS_NO)

- clean: internal **0.230** endpoint 0.280 external 1.000 gpt_ic 0.260
- noisy: internal **0.160** doc_id_wrong 0.990 (p=0.3)
- gates: {'g1_beats_endpoint': False, 'g2_near_external': False, 'g3_beyond_window': True, 'g4_not_metadata': False, 'g5_anticf': True}