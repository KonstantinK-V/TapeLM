# Stage 288 repair (address holdout)

**REPAIR_PARTIAL** · params **6562**

| held out | model | votes | random |
|---|---:|---:|---:|
| detect forged | 0.760 | 0.297 | 0.400 |
| detect dup | 0.984 | 0.057 | |
| clean pass | 0.528 | 0.553 | |
| repair reward | 0.478 | 0.478 | 0.283 |

- clean-margin AUC 0.859 (+11.25 sigma)
- UNKNOWN margin AUC 0.677 (+2.04 sigma) on 20 unrecoverable vs 26 recoverable
- observer: verdict restored 0.217 vs votes 0.217

## Gates

- G_arc_enc_frozen: **True**
- G_task_exists: **True**
- G_detects_forgery: **True**
- G_detects_dup: **True**
- G_flags_clean: **True**
- G_repairs: **False**
- G_honest_unrecoverable: **True**
