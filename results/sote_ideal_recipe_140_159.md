# SOTE ideal recipe — generalization from Stages 140–159

**Scope:** word-path LM (exact@1 STORY / SEEN), TinyStories-scale (~100k), F85 letter-fp init, hops **OUT** of LM train/gate.  
**Not claiming:** wiki/large-N final, BDLM product, or “beats BPE forever.”

---

## 1. Generalization (what the data say)

### A. Gap is real but was inflated by ritual mismatch
| Compare | Word STORY | BPE−word gap |
|---------|------------|--------------|
| 149 GPT lock, BPE without fat | 19.4% | ~**+13–14pp** |
| 150 fat-matched, trunk **S** | 18.6% | **+4.3pp** |
| 150 fat-matched, trunk **G** | 19.5% | **+9.1pp** |
| 150 fat-matched, trunk **S+** | 14.7% | **+6.9pp** |

**Read:** under matched fat/holds, residual gap is **moderate (~4–9pp)**, not ~14. Ritual × unit matters as much as “BPE magic.”

### B. SOTE’s historical ritual is the right *family* — capacity headroom is not
- **S** (2L / d256 / batch8 / lr1e-3 / Adam) beats **S+** (4L8H / d512) by **~+4pp** word STORY @100k.
- 154: d-headroom lifts **negative** at 100k and 460k (−4.4 / −3.7pp word).
- 151: on S+, **layers 4→2 = +4.8pp** (largest single lever in the table).

**Read:** at this V/window/N, SOTE wants **thin**, not deep-wide. Dim is **not** closed forever for all futures — but **default ship recipe is thin**.

### C. What does *not* close the residual gap
| Dig | Result |
|-----|--------|
| Context / BPE budget match (152) | gap still **+7pp** |
| Weird streams (141–147) | PARITY / HARM |
| Alphabet-BPE (148) | HARM |
| Soft morph 3-aff (156), rare-drop (157), ComposeLayer (158) | PARITY / HARM / PARITY |
| Rich BPE-like tails, cov~45% (159) | **PARITY** (−0.3pp) |

**Read:** residual is **not** “add shared morph/tail rows into the word book” under current CE+eval. Prefer **unit/geometry** next, not more soft composition overlays.

### D. Batch & lr are ritual-coupled (not universal folklore)
- Under **G**: batch**16** ≫ batch8 (153); lr**1e-3** helps (+2.2 on 151); Adam helps (+1.1).
- Under **S**: batch**8** is the locked winner of the hist recipe; S+ mildly likes batch16 (+1.4) but S+ itself is weak.
- Early SOTE lore (batch8 @ small-N/fat) **≠** “always batch8 under every trunk.”

### E. Absolute word peak ≠ best SOTE identity
- Highest word STORY in clean table: **G ~19.5%** (≈149 word).
- Best **matched gap** (SOTE-shaped): **S +4.3pp**.
- Pick **S** as *SOTE recipe*; treat G as *GPT-ritual control*, not as “what SOTE is.”

---

## 2. Ideal SOTE parameter recipe (default lock)

### Recipe **S★** — ship / primary (validated @ ~100k)

| Axis | Value | Evidence |
|------|-------|----------|
| **Unit** | `word_fp` (F85 letter-fp → word emb) | 149–150; func_bias only +~3.5pp, not a path change |
| **Codebook** | **whole-word atoms only** | 156–159 soft morph/tails = no STORY lift |
| **d** | **256** | S ≫ S+; 154 headroom hurts |
| **layers** | **2** | S; 151 S+ 4→2 +4.8pp |
| **heads** | **4** | S; S+ heads↓ also hurts when paired wrong |
| **max_len** | **16** word slots | 152: longer BPE wire ≠ gap closer |
| **batch** | **8** | hist S; keep until S×batch16 one-factor |
| **lr** | **1e-3** | S; 90/93 lore; G also likes 1e-3 |
| **schedule** | warmup **200** → **constant** | 92 cosine FAIL; 93 warmup OK |
| **opt** | **Adam**, **wd=0** | S lock; AdamW helps *S+* not proven on S |
| **emb lr** | **full** (no ×0.2) | 94 FAIL |
| **fat_frac** | **0.75** | matched protocol; don’t compare without it |
| **steps** | **40k** (eval every 4k) | 150–159 lock |
| **data** | TS-100k mix, seed **272** | plan lock |
| **hops** | **OUT** of LM train/gate | contract |
| **gate** | word exact@1 STORY (+ SEEN obj/rel report) | contract |
| **ckpt** | best on (ev_story, obj); also log `story_all_last` | 150 protocol |

**Expected ballpark @100k (S★):** STORY ~**18–20%**, SEEN obj ~**40%**, matched BPE gap ~**4–5pp** (not 14).

### Recipe **G★** — only as GPT-ritual control (not “SOTE”)

| Axis | Value |
|------|-------|
| batch / lr / opt | **16 / 3e-4 / AdamW wd0.01** |
| depth | **4L / 4H / d256** |
| else | same fat/window/eval as S★ |

Word STORY can match or slightly beat S★; **gap vs BPE worse** (~+9pp fat-matched). Use for fair unit compares, not as SOTE identity.

### Optional **S★+** probes (not locked — need new one-factor on S, not on S+)

Do **not** sum 151 deltas. Candidate next single levers *on S★*:

1. batch 8→**16** (153 says G loves 16; S+ mild +1.4)
2. Adam→**AdamW+wd0.01** (helped S+, untested on S)
3. **Geometry only:** 2L×d512 or wide-shallow / pyramid (agreed next dig) — keep ritual S otherwise

If a probe wins ≥~+2pp STORY without SEEN collapse → promote into S★.

---

## 3. Explicit anti-recipe (do not ship)

| Don’t | Why |
|-------|-----|
| Default **4L / d512 / 8H** (S+) for 100k word LM | −4pp vs S; 154 still negative |
| Freeze “dim forever” from one flat@100k *or* blindly ↑d with N | 154: capacity not the bottleneck here |
| Train/compare BPE without **same fat_frac** | fake +13pp gap (149) |
| Alphabet / char-BPE as tokenizer (148) | HARM |
| Soft morph / rare-drop / compose / rich tails as STORY unlock (156–159) | PARITY–HARM |
| Reverse LM, noise emb, copy-token tricks (141/146) | HARM |
| Cosine / slow-emb schedules from early lore (92/94) | SEEN collapse |
| Put **hops inside** LM gate | contract |
| Claim from unmatched context (word16 vs BPE48) without 152-style note | confounder |

---

## 4. Residual theory (after recipe lock)

Ordered leftover for the ~**4–5pp** matched gap under S★:

1. **Unit** — whole-word CE vs reusable subword pieces (composition in the *stream*, not soft codebook add-ons)
2. **CE density** — last-position word vs full-seq BPE
3. **Eval surface** — word argmax vs piece→word greedy
4. **Geometry** — layer shape at fixed param budget (open)

**Not primary:** window length, “need more dim,” soft `+ing` rows in the word book.

---

## 5. One-block copy-paste defaults

```text
SOTE S★ (word LM @ TS~100k)
  unit=word_fp  V≈word_types  max_len=16
  d=256  n_layer=2  n_head=4  dropout=0.1
  batch=8  lr=1e-3  opt=Adam  wd=0
  warmup=200  schedule=constant  steps=40000
  fat_frac=0.75  seed_mix=272
  init=F85_letter_fp  hops=OUT  gate=exact@1_word
```

---

## 6. Success framing (unchanged)

- **(A)** absolute word STORY ~33%, and/or  
- **(B)** matched BPE−word gap ≤~5pp under S★ ritual  

S★ already sits near **(B)** (~4.3pp). **(A)** still open — next pressure: geometry + honest unit/CE, not more soft morph.

---

*Sources: `sote_summary_140_plus.md`, stage149–159 decision.json, `plan_150_plus_clean_compare.md`.*

---

## 7. Scale recipes — ~500k and ~1M (not the same claim as 100k)

### What is actually measured

| N (phrases) | Source | Word STORY (thin / S★-like) | vs headroom (4L d512) | Matched BPE gap |
|-------------|--------|------------------------------|------------------------|-----------------|
| ~100k | 150 / 154 hist | **18.6%** | headroom **−4.4pp** | **+4.3pp** (S) |
| ~460k | 154 hist | **19.4%** | headroom **−3.7pp** | **+4.8pp** (24.2−19.4) |
| ~460k | Stage101 (recipe98≈S) | **20.6%** | — | — |
| ~1M | — | **no dig** | — | — |

**Read @~500k:** same story as 100k — **thin S★ still beats fat S+**; STORY only **+~1pp** vs 100k (soft ceiling under this unit/window). Capacity↑ with N did **not** unlock; `premature_close` evidence = flat both N under this V/window.

### Recipe **S★₅₀₀** — default for ~460–500k (evidence-backed)

Same as S★ @100k, with scale hygiene:

| Axis | Value | Note |
|------|-------|------|
| arch / opt / lr / batch / fat | **identical to S★** | 154: hist beats headroom @460k |
| **V / tok** | **rebuild word list (and BPE tok) at this N** | 154 did; don’t reuse 100k V |
| steps | **40k minimum**; if wall-clock allows, probe **60–80k** at same lr | not factorial-proven; 101 used longer hist runs |
| eval | same holds protocol; report `story_all_last` | — |
| hops / gate | OUT / exact@1 | — |

**Do not** switch to S+ “because more data.” 154 says the opposite through 460k.

**Batch caveat:** 153’s batch16 win was under **G** and weak under **S+**. At 500k, **one-factor on S★: batch 8 vs 16** is the highest-value unset probe (more tokens/step may matter more as N grows). Until that dig: keep **batch8**.

### Recipe **S★₁ₘ** — working prior for ~1M (extrapolation only)

No 1M table in 140–159. Prior = **S★₅₀₀** until falsified:

```text
SOTE S★_1M (prior, unvalidated)
  start from S★_500 (2L4H d256, Adam, lr1e-3, fat0.75, max_len16)
  rebuild V @1M
  steps: scale with N (e.g. 80–120k) OR fixed tokens-seen target — pick one and lock
  first digs before changing arch:
    (1) batch 8 vs 16 on S★
    (2) geometry at fixed param budget (2L×d512 vs 2L×d256 vs pyramid)
    (3) only then consider mild width — not 4L8H S+ as default
```

**Promotion rule:** promote S+ / deeper stack to default **only if** a matched N×geometry dig shows ≥~+2pp STORY on word **and** no SEEN collapse, at that N. 100k and 460k both failed that bar for S+.

### Absolute expectations (honest)

- **500k:** word STORY still ~**19–21%** under S★-family — not suddenly 33%. Gap vs matched BPE stays ~**5pp** order.
- **1M:** unknown; data alone from 100k→460k gave ~+1pp — **don’t bet the plan on N alone**.
- Success **(A) ~33%** likely needs **unit/geometry/CE**, not “S★ + more TS lines.”

## 8. Stage 160–161 — geometry @500k

**160 DONE — PARITY_TABLE:** base **2L d256** wins (STORY **19.1%**).  
wide −0.5 · deep4 −0.8 · pyramid↑ −1.3 · **pyramid↓ −5.8**. No promote.

**161 DONE — ENVELOPE_MAPPED:** best **thin_4L_d128** STORY **20.2%** (+1.1 vs 160 base).  
Survivors: thin_4/2L_d128, 1L_d256. Collapse: 8L and ultra-wide. 1L_d512 −1.4; 2L_d768 −4.2.

**162 DONE — PARTIAL:** `1L d128@50k` STORY **21.6%** (+2.5pp vs base). `1L d256@50k` **19.6%** (~flat vs 40k +0.3). Thin 1L wins; longer run doesn’t save wide 1L.

**163 DONE — FLOOR_SOFT:** best **1L d128 h2** **21.7%** (~= h4). h1 21.4. **d64 ~20.7–21.0**, **d32 still 20.6** (−1.0). No collapse — floor soft; heads almost free.

**164 DONE — ATTENTION_REDUNDANT:** gap 1L−0L_last@128 only **+0.9pp**.  
`0L last d256` **22.5%** (!), `0L last d128` 20.7%, meanpool 19.4%. STORY mostly emb→linear (bigram-like).

**165 DONE — SOFT_COLLAPSE:** d16 ~**18.3–18.4%** (−3.2 vs d128); **d8 14.4%** (−7.2). Hard floor near **d8**; d16 soft drop.

**166** relaunched after WT103 download fix — wiki 50M + d128/d256/0L/BPE.

