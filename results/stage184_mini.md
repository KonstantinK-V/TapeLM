# Stage184 — log-prob exam calibration

**Overall:** `CALIBRATED_NO_ENTITY_SIGNAL`

chance=0.25

- `ce_gpt_181`: next_tok=0.758 entity=0.180 ood=0.200
- `hybrid_182`: next_tok=0.650 entity=0.190 ood=0.267
- `random`: next_tok=0.275 entity=0.190 ood=0.200

If HARNESS_STILL_BROKEN: fix scorer/data before any curve claim. If CALIBRATED: entity_acc is now a trustworthy dataset-answer number; proceed to S2 (addressable tape).
