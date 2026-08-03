# Stage 242 rehearsal dose

**REHEARSAL_DOSE_PARTIAL**

```json
{
  "stage": 242,
  "overall": "REHEARSAL_DOSE_PARTIAL",
  "gates": {
    "G_tape_retain_ge_0p80": true,
    "G_zero_rehearsal_below_target": true,
    "G_found_dose": false
  },
  "tape_A_after_B": 1.0,
  "gpt_A0": 0.625,
  "target_gpt": 0.95,
  "min_rehearsal_to_match": null,
  "curve": {
    "0.0": 0.53125,
    "0.05": 0.6875,
    "0.15": 0.75,
    "0.3": 0.75,
    "0.5": 0.8125
  },
  "W_align": 0.9947370886802673,
  "memorize_steps": 2000,
  "note": "Price of anti-CF in weights = fraction of A tokens mixed into B CE.",
  "timestamp": "2026-07-31T04:59:53.705090+00:00",
  "wall_s": 294.7210536003113
}
```
