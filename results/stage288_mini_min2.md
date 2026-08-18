# Stage 288 repair (corpus holdout)

**REPAIR_PARTIAL** · params **6562**

| held out | model | votes | random |
|---|---:|---:|---:|
| detect forged | 0.671 | 0.333 | 0.304 |
| detect dup | 0.940 | 0.162 | |
| clean pass | 0.504 | 0.453 | |
| repair reward | 0.085 | 0.184 | 0.546 |

- clean-margin AUC 0.731 (+7.05 sigma)
- UNKNOWN margin AUC 0.603 (+2.43 sigma) on 64 unrecoverable vs 170 recoverable
- observer: verdict restored 0.842 vs votes 0.803

## Gates

- G_arc_enc_frozen: **True**
- G_task_exists: **True**
- G_detects_forgery: **True**
- G_detects_dup: **True**
- G_flags_clean: **True**
- G_repairs: **False**
- G_honest_unrecoverable: **True**
