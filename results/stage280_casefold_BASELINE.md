# Stage 280 casefold — project baseline on raw text

**`RAW_EXAM_OK`** · all 10 gates · `--min-mentions 2 --min-per-family 8 --run-tag casefold`  
(with `texts_lc` subject filter — the register bug fix)

## Held-out

| | |
|---|---:|
| reward_total | **0.704** |
| teacher ceiling | 0.625 |
| acc_answered_all | **0.900** |
| coverage_all | 0.526 |
| tie.abstain | **0.85** |
| clean / decidable acc_answered | **1.0 / 1.0** |
| retrieval_precision | 0.823 |

## Attribution

Growth m2b (0.227) → 282 MIND_OK (0.594) was mostly this search fix, not A/B/C.  
Fair bar for later 282: **beat or match 0.704**, not 0.227.

## Artifacts

- `results/stage280_decision_fp_casefold_BASELINE.json`
- `results/_stage280_full_fp_casefold_BASELINE.out`
- `checkpoints/stage280_raw_exam_fp_casefold_BASELINE.pt` (if present)
