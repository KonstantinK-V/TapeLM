# Stage 279 write as a decision (model)

**SUPPORT_NOT_CALIBRATED** · SMOKE · trained parameters **0**

- 365 assertions -> **248** slots over 60 addresses (32.1% saved against an append)
- actions {"WRITE": 60, "CONFIRM": 117, "DISPUTE": 188}, dedup **0.321**, disputed addresses **0.933**
- replay control (identical second pass): {"WRITE": 0, "CONFIRM": 365, "DISPUTE": 0}
- model self-dispute **1.000**, consistency lift **nan** (low nan -> high nan)

## Gates

- G_no_false_dispute_on_replay: **True**
- G_dedup_happens: **True**
- G_tape_compresses: **True**
- G_disputes_found: **True**
- G_soft_match_merges_forms: **True**
- G_support_predicts_truth: **False**
