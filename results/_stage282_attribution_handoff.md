# Attribution after MIND_OK (both-names)

Archived: `stage282_decision_MIND_OK_bothnames_pre_twowitness.json`

## Queue (`_queue_attr_280_282.py`)
1. **280 casefold control** — same knobs as m2b (`--min-mentions 2 --min-per-family 8 --run-tag casefold`); `texts_lc` already in 280. Separates search bugfix from A/B/C.
2. **282 full two-witness** — overturn tie needs ≥2 return mentions; ceiling should rise ~0.445 → ~0.7.
3. **282 `--no-probe`** — price of C. Skip `--no-edit` (edits always 0).

## Known open
- B dead (votes never empty)
- `stop_unknown` = 0; typed silence half-alive; touch after attribution
