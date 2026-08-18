# Stage 288 repair (corpus holdout)

**REPAIR_PARTIAL** · params **6562**

| held out | model | votes | random |
|---|---:|---:|---:|
| detect forged | 0.671 | 0.333 | 0.304 |
| detect dup | 0.940 | 0.162 | |
| clean pass | 0.504 | 0.453 | |
| repair reward | 0.750 | -0.278 | 0.243 |

- clean-margin AUC 0.729 (+6.99 sigma)
- UNKNOWN margin AUC 0.476 (-0.36 sigma) on 26 unrecoverable vs 64 recoverable
- observer: verdict restored 0.000 vs votes 0.644

## Gates

- G_arc_enc_frozen: **True**
- G_task_exists: **True**
- G_detects_forgery: **True**
- G_detects_dup: **True**
- G_flags_clean: **True**
- G_repairs: **True**
- G_honest_unrecoverable: **False**
