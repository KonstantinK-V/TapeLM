# 344 amended: retry budget is per LEVER, not per step
# (Kostya's correction; HANDOFF.md locally truncated — this file holds the rule.)

ONE LEVER = one mechanism, 4 seeds vs declared gate. Failed → closed, never tuned.
A STEP = 3–5 levers, then the VIEW of the problem changes (mandatory).
New lever = different mechanism. Parameter retune ≠ lever.

## Step 1 levers

| | mechanism | status |
|---|---|---|
| L1 | raw co-occurrence count | SPENT — (b) 0.044 vs 0.127; (a) VOID at constrain_rate ~0.003; coverage matched ahead 3/4 |
| L2 | share of value's presence | SPENT — worse than count (singleton tail); (b) 0/4; coverage 0/4; (a) VOID |
| L3 | two lenses intersected (joint co-oc) | hold — ceiling via `_audit346_lens` first (measurement ≠ lever) |

Bug fix ≠ lever. Near-miss = directionally right ≥3/4 seeds below threshold (L1/L2 were not).
If support_2plus < 10%: no peak → do not build L3; try thicker places (fixed addresses) first.
If all three fail: generation cannot come from counting neighbours on this thin tape.

## _audit346_lens (2026-08-16)

30MB (default --lines 25000): support_2plus **0.160** (above 10% floor) but
pair_present@8 **0.088 <** one_present@8 **0.193** → **TWO LENSES DO NOT SHARPEN IT**.

Default 120MB was a no-op (same 25000-line cap → identical tape).
120MB --lines 100000: mentions/place 3.99 (not thicker), support_2plus **0.135**,
pair 0.097 < one 0.196 — still no sharpen; peak did not appear with more corpus.

HANDOFF.md truncated locally — lever/L2 prose not applied there; rule kept in this file.
Audit script `_audit346_lens.py` applied cleanly.

## 346 read / L3

Corpus `--bytes` did not thicken (window owns thickness). L3 closed unbuilt.
Next: `_sweep347_thick.py` on `--window-lines` and `--frame-max`.

## Power (decided)

`--reach-max-q 8000` both arms next comparison; do not touch `--addresses`.
Do not re-run 345 at 8000 (gates not marginal). See results/_arm_ctrl_next.txt.

## 342a free read

WALK-ONLY identical d32/d64 is mind-invariance (336), not a tag collision —
CONFIRM / GATE-WO / OTHER all moved. `margin_by_stage` ABSENT from those JSONs
(deflation question unchecked).
