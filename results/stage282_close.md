# Stage 282 close

**Verdict:** MIND_OK on paper, below the fair bar. Reference mind stays **280 casefold** (`RAW_EXAM_OK`, held-out reward **0.704**).

## What 282 was

One mind with A (typed silence: CONFLICT / UNKNOWN), B (query edit), C (ASK_VALUE return-path probe). Same 280 tape so deltas are from new actions.

## Attribution (fair bar = 0.704, not m2b 0.227)

| Run | Overall | Reward | Teacher | Acc | Tie abstain |
|---|---|---:|---:|---:|---:|
| 280 casefold | RAW_EXAM_OK | **0.704** | 0.625 | 0.900 | **0.85** |
| 282 two-witness (tie-probe era) | MIND_OK | 0.594 | 0.533 | 0.811 | **0.70** |
| 282 `--no-probe` | MIND_OK | 0.564 | 0.594 | 0.789 | 0.70 |

Early jump m2b→0.59 was mostly **casefold/search**, not A/B/C.

## Channel status

- **B:** dead (`mean_edits=0`; votes never empty → teacher never DROP/ADD).
- **A:** half-dead. CONFLICT works; `STOP_UNKNOWN=0` everywhere. Splitting STOP into two types bought typed silence on ties at **0.70 vs 280’s 0.85** — that gap, not C, is why 282 does not reach the bar.
- **C:** keep as **measurement** (`acc_when_probe_hit` vs `acc_when_probe_miss`). Do not sell probe as the product goal or as the default tie resolver.

## Code rule (post-close)

Default: **tie → STOP_CONFLICT without probing**.  
`--tie-probe` restores the old two-witness probe-on-tie rule (the run the 0.594 numbers come from).

## Keep / drop

| Keep | Drop / do not grow |
|---|---|
| 280 casefold baseline as reference mind | A in current form as the next product step |
| C as a hit/miss diagnostic | Probe-as-tie-breaker as the design |
| Tape / search / casefold fixes that made 280 honest | Resurrecting B before A/C are clean |

## Next (outside 282)

Grow the exam (scale, free answer, stream), not the game around current metrics. Provenance / return path can replace `answer_is_slot` later; verifier ≠ generator.
