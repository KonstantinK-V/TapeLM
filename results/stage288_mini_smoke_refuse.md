# Stage 288 repair (corpus holdout)

**REPAIR_PARTIAL** · SMOKE · params **6562**

| held out | model | votes | random |
|---|---:|---:|---:|
| detect forged | 0.812 | 0.250 | 0.429 |
| detect dup | 1.000 | 0.000 | |
| clean pass | 0.875 | 0.875 | |
| repair reward | nan | nan | nan |

- clean-margin AUC 0.961 (+5.16 sigma)
- UNKNOWN margin AUC nan (+nan sigma) on 0 unrecoverable vs 0 recoverable
- observer: verdict restored nan vs votes nan

## Gates

- G_arc_enc_frozen: **True**
- G_task_exists: **True**
- G_detects_forgery: **True**
- G_detects_dup: **True**
- G_flags_clean: **True**
- G_repairs: **False**
- G_honest_unrecoverable: **None**
