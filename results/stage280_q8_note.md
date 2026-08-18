# Stage 280 — q8 (archived)

**Tag:** `q8` · `--min-per-family 8` · `--min-mentions 3` (default)  
**Artifacts:** `stage280_decision_fp_q8_pre_stallfix.json`, `_stage280_full_fp_q8_pre_stallfix.out`, `checkpoints/stage280_raw_exam_fp_q8_pre_stallfix.pt`

## Claim (pre-stallfix scoring)

Policy beat the executable teacher on held-out raw text: **reward 0.467 vs ceiling 0.367**, `acc_answered_all` **0.769** vs **0.733**. Two decidable abstentions where the teacher answered wrongly; one (`texas`) was a **stall** (ASK_Q loop to `max_steps`) paid as abstain +0.75 — fixed afterwards so stall → 0.0, only explicit STOP earns silence reward. The other (`united states`) is a real STOP.

## Why quota did nothing

Held-out tape identical to `nat`: 15 addresses, clean 6 / decidable 7 / **tie 2**. `--min-per-family` cannot invent ties. Nat vs q8 differ only in training item order.

## Stall re-score (from traces, no re-eval)

- `united states`: real **STOP** after 7 reads — keeps abstain reward.
- `texas`: stall (`ASK_Q`×5 after reads, no STOP) — under fixed rule reward drops by **0.75** on that item → mean reward **0.467 → ~0.417**, still **above** teacher **0.367**. `acc_answered_all` unchanged.

## Next

`--min-mentions 2 --min-per-family 8 --run-tag m2` — more addresses, natural ties, then `G_abstain_on_tie` measurable.
