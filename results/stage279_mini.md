# Stage 279 write as a decision (corpus)

**WRITE_DECISION_OK** · SMOKE · trained parameters **0**

- 372 assertions -> **144** slots over 48 addresses (61.3% saved against an append)
- actions {"WRITE": 48, "DISPUTE": 96, "CONFIRM": 228}, dedup **0.613**, disputed addresses **0.833**
- replay control (identical second pass): {"WRITE": 0, "CONFIRM": 372, "DISPUTE": 0}

## Gates

- G_no_false_dispute_on_replay: **True**
- G_dedup_happens: **True**
- G_tape_compresses: **True**
- G_disputes_found: **True**
- G_soft_match_merges_forms: **None**
- G_support_predicts_truth: **None**
