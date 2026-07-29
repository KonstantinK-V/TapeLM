# Plan — Wiki 50M tokens: 1L d128 h2 vs 0L vs BPE + understanding probes

**Budget:** ~10h wall clock. **Goal:** does attention matter when data is harder than TinyStories?

## Why this dig

On TS@460k: `0L last ≈ 1L` (ATTENTION_REDUNDANT). Possible causes: easy kids text / fat bigrams / small N.  
**Raise the bar with Wikipedia ~50M tokens** (not 0.5B params). Metric of interest = **gaps**, not absolute %.

## Arms (matched ritual)

| Arm | Arch | Notes |
|-----|------|-------|
| `word_1L_d128_h2` | 1L / 2H / d128 | thin primary |
| `word_1L_d256_h2` | 1L / 2H / d256 | **capacity / underfit probe @50M** |
| `word_0L_last_d128` | emb[last]→linear | bigram-like control |
| `bpe_1L_d128_h2` | GPT2 1L/2H/d128, V_bpe=8k | unit compare |

**Underfit read:** if `d256 − d128` ALL ≥ +2pp on wiki → thin was underfitting at 50M; if ≤0 → same “more width hurts/flat” as TS.

**Ritual lock:** batch8 / lr1e-3 / Adam / wd0 / **fat_frac=0.25** (light; not TS fat0.75) / max_len16 word / BPE maxlen48 / **80k steps** / warmup200.  
**Hops OUT.** Gate = word exact@1.  
**V_word:** top-40k train types + `<unk>` (log coverage).

## Probes (all arms, same HOLD)

1. **HOLD ALL / obj** — standard word exact@1  
2. **order-shuffle** — shuffle prefix tokens; report clean − shuffled (**order_drop**)  
3. **same-last** — prefixes sharing last word but different gold next; model vs majority-next-from-last baseline  

**Understanding signal:** 1L ≫ 0L on order_drop and/or same-last (not just ALL).

## Timeline (~10h)

| Step | What | Est. |
|------|------|------|
| **0** | Download WikiText-103 train; SOTE-filter; take first **50M tokens**; cut ≤8-word windows; write corpus | 1.0–2.0h |
| **1** | Train `word_1L_d128_h2` 80k | 0.75–1.5h |
| **1b** | Train `word_1L_d256_h2` 80k (**underfit probe**) | 0.75–1.5h |
| **2** | Train `word_0L_last_d128` 80k | 0.5–1.0h |
| **3** | Train BPE tok + `bpe_1L_d128_h2` 80k | 1.0–1.5h |
| **4** | Probes + decision.json + short report | 0.3–0.5h |
| **buffer** | download retries / VRAM / re-eval | ~1–2h |

If short on time: drop BPE (keep 0L vs 1L probes — main question). If long: + `word_1L_d128_h4` smoke.

## Data contract

- Source: **WikiText-103** train (not WT2 — too small for 50M).  
- Charset: SOTE a-z0-9 space (same as Stage85+).  
- Windows: min3–max8 words, max_word_len24.  
- Claim **50M tokens** = count of whitespace tokens in filtered stream before/while windowing (logged).  
- HOLD = natural held-out windows from same stream (no TinyStories fat path triples).

## Verdict rules

| Verdict | Condition |
|---------|-----------|
| `ATTENTION_MATTERS_ON_WIKI` | 1L − 0L ALL ≥ +3pp **or** order_drop_1L − order_drop_0L ≥ +5pp **or** same_last gap ≥ +5pp |
| `UNDERFIT_AT_50M` | word d256 − d128 ALL ≥ +2pp (width helps on wiki) |
| `STILL_REDUNDANT` | 1L − 0L ALL < +1.5pp **and** probe gaps < +3pp |
| `MIXED` | else |
| Also log | BPE − word_1L_d128 ALL (unit gap on wiki) |

## Files

- Plan: `results/plan_wiki50m_probes_10h.md` (this file)  
- Runner: `_stage166_wiki50m_0l_1l_bpe_probes.py`  
- Log: `results/_stage166_wiki50m_log.txt`  
- Decision: `results/stage166_wiki50m_0l_1l_bpe_probes_decision.json`  
- Corpus: `data/external_wikitext103_50m_tokens_85.txt`

## Order vs live queue

Runs **after Stage165** (hard floor d16/d8). Resume-safe if decision exists.

## Stretch (Stage 167) — if 166 finishes early **or** results are ambiguous

Auto-queued after 166. Policy by `wall_hours`:
- **&lt;6.5h FULL:** 0L@d256, 2L@d128, 1L@h4, 1L fat=0, bigram table
- **&lt;8.5h PARTIAL:** 0L@d256, 2L@d128, bigram
- **else:** bigram only

**Ambiguous 166 → long soak 300–500k** (even if wall was long):
- Triggers: `MIXED`; gray gaps (attention ~0.8–3pp, width ~0.8–2.5pp, order/same-last ~1.5–5pp); curve still climbing at 80k
- **300k** default on contenders: `1L d128`, plus `d256` and/or `0L` as disputed
- **500k** if ≥3 reasons / climbing / wall &lt;3.5h; or escalate after 300k still gray

Runner: `_stage167_wiki50m_stretch.py`
