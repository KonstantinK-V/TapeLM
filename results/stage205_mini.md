# Stage205 — targeted unlearning / provenance / audit

**Overall:** `UNLEARN_PROVENANCE_WIN`

| metric | curve before | curve after | GPT after memorize | GPT after unlearn |
|--------|--------------|-------------|--------------------|-------------------|
| target fact recall | 1.000 | **0.000** | 1.000 | 0.000 |
| retained fact recall | 1.000 | **1.000** | 1.000 | 0.200 |
| next_tok (collateral) | 0.825 | **0.825** | 0.808 | 0.733 |

- curve delete: 20 slots in 0.0 ms, no gradient
- GPT unlearn: 30 gradient steps, 2.1 s
- provenance attribution (curve): 1.000
- conflict audit: detection 1.000, false-positive 0.000
- gates: forget=True no_collateral=True gpt_collateral=True prov=True conflict=True