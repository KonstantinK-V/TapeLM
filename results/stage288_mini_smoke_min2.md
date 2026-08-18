# Stage 288 repair (corpus holdout)

**REPAIR_PARTIAL** · SMOKE · params **6562**

| held out | model | votes | random |
|---|---:|---:|---:|
| detect forged | 0.812 | 0.250 | 0.429 |
| detect dup | 1.000 | 0.000 | |
| clean pass | 0.875 | 0.875 | |
| repair reward | 0.750 | 0.750 | 0.875 |

- clean-margin AUC 0.977 (+5.34 sigma)
- UNKNOWN margin AUC 0.728 (+1.45 sigma) on 4 unrecoverable vs 28 recoverable
- observer: verdict restored 0.938 vs votes 0.938

## Gates

- G_arc_enc_frozen: **True**
- G_task_exists: **True**
- G_detects_forgery: **True**
- G_detects_dup: **True**
- G_flags_clean: **True**
- G_repairs: **False**
- G_honest_unrecoverable: **None**
