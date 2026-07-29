# Plan — SOTE understanding mid-path (~20–30h, checkpoints)

**Goal:** Can SOTE show *understanding* (not just last→next), on a 3050, without 100B data.  
**Not:** BPE race, width/depth grid, soft morph, “accept falsify and stop.”  
**Unit chapter:** closed (166 SOTE≈BPE). This branch is **understanding**.

## Contract

| Axis | Lock |
|------|------|
| Model | **thin control** 1L d128 h2 + **deep arm** 4L d128 h4 (+ matched 0L); thin ≠ understanding lock |
| Orthography | **case + punct + digits + basic specials** (not a-z0-9 strip) |
| Primary gate | **battery / same-last / order** vs 0L & majority-from-last |
| Secondary | HOLD ALL% |
| Hops | OUT of LM CE gate (dual-channel cue is **separate emb channel**, not hop-into-CE) |
| Cue (CP3) | **dual-channel embedding**: language trunk + **CueChannel** (context/punct/case structure) fused at predict |
| Hardware | RTX 3050 — batch8, resume ckpts |
| Budget | ~20–30h wall, **kill switches per CP** |

## Checkpoints

### CP0 — infra (2–3h)
Rich Wiki corpus + battery builder + alphabet/meta for case/punct.  
**Out:** corpus, battery, `stage168_cp0_decision.json`, mini report.  
**Kill:** none (infra).

### CP1 — plain CE rich baseline (4–6h)
Thin 1L + 0L on rich text, **plain next-token CE**. Battery + probes.  
**Kill:** do **not** tune width/lr if dead — proceed to CP2 (expected that orthography alone may fail).

### CP2 — task/loss change + depth (8–12h)
**CE on ambiguous-last** + **contrast aux**, two trunks under **understanding-gate** (battery / lift vs maj / vs CP1 0L):
- `1L_ambig` — thin control (warm-start CP1)
- `4L_h4_ambig` — **deep arm** (hypothesis: depth for thought, not ALL)

**Kill:** if *both* arms lift vs CP1 &lt; +5pp → don’t ×3 steps; go CP3 (channel) with **best** arm as trunk.  
Note: thin won historical STORY — that selected memorization; depth re-opened only for understanding metrics.

### CP3 — dual-channel cue emb (6–8h)
**CueChannel** on **best CP2 trunk** (thin or deep): separate emb path, last blanked, fuse → logits.  
**Kill:** still last-token → `NEEDS_HARDER_INSTANCE_CHANNEL` (next research), not 500k soak.

### CP4 — report (≤1h)
Verdict: `UNDERSTANDS` / `PARTIAL_LOSS` / `PARTIAL_CUE` / `STILL_LAST_TOKEN`.

## Success (understanding)
On battery HOLD: 1L (or cue model) **≥ +5–10pp** vs majority-from-last **and** vs 0L on same-last/order.  
ALL% alone never counts as understands.

## Files
- Plan: this file  
- Runner: `_stage168_understand_mid_pipeline.py --cp N`  
- Logs: `results/_stage168_cpN_*.txt`  
- Decisions: `results/stage168_cpN_decision.json`  
- Mini reports: `results/stage168_cpN_mini.md`
