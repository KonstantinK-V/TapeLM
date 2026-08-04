# Stage210 — SoftFollow in forward, token answers

**Overall:** `THESIS_NO_AT_SCALE` (legacy label: THESIS_NO)

- soft-follow token test: k2=0.250 k3=0.300
- free-form token test: k2=0.333 k3=0.217 (train k2=0.278)
- no_memory max=0.317 | external cosine {1: 1.0, 2: 1.0, 3: 1.0}
- next_tok delta=0.0 bit_diff=0.0
- gates: {'g1_generalize': False, 'g2_structure': False, 'g3_no_ce_cost': True, 'g4_bit_identity': True, 'g5_needs_memory': True}