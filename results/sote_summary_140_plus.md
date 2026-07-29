# SOTE summary — Stages ~140 → 159 (living)

**Cut date:** 2026-07-26. Exact@1 only. Hops OUT of LM train/gate. Dual-channel foundation F85 frozen.

**Program spine:** finish clean SOTE-as-LM digs → understand residual word↔BPE gap → then BDLM product layer (separate) + geometry digs.

---

## One-line verdict arc

| Block | Outcome |
|-------|---------|
| **140** | Budget-matched BPE still beats word (~+10pp HOLD) — gap not “more tokens in window” |
| **141–147** | Weird-unit digs mostly PARITY / HARM — no free STORY unlock |
| **148** | Alphabet-BPE **HARM** — broken as tokenizer path |
| **149** | Matched GPT ritual: word **19.4%** STORY, BPE ~**32–34%**, gap ~**+13–14pp** (BPE lacked fat — inflated) |
| **150–155** | Clean compare: fat-matched gap shrinks to **~4–9pp** by trunk; **primary=S**; dim headroom hurts @100k |
| **156–157** | Shared morph (3 aff) **PARITY/−0.9pp**; rare morph-only **HARM/−2.0pp** |
| **158** | ComposeLayer **PARITY** (~13.8% STORY, ≈ctrl) — preprocess alone ≠ lift |
| **159** | BPE-like tails **PARITY** (−0.3pp vs ctrl; story **13.6%**, cov~45%) — richer tails ≠ STORY lift |

---

## 140 — BPE budget vs word atoms

- Protocol: pack BPE to similar mean pieces/line as words (~7.83).
- Word HOLD_ALL **18.4%**, BPE **28.6%**, gap **+10.2pp** → `GAP_SIMILAR`.
- Space/punct contract logged (whitespace-BPE ≠ tiktoken).
- **Read:** gap survives budget match → composition/unit, not raw length.

---

## 141–147 — weird units (pipeline)

Upstream 140. Soft lifts vs local ctrl (STORY):

| Stage | Dig | Verdict | Lift |
|-------|-----|---------|------|
| 141 | Reverse LM word | **HARM** | −10.9pp |
| 142 | Masked word fill | PARITY | −2.9pp |
| 143 | Char stream → word confirm | PARITY | −6.8pp |
| 144 | Random bigram bank | PARITY | ~0 |
| 145 | Pos shuffle bag | PARITY | +0.6pp |
| 146 | Copy-token / noise emb | **HARM** | −11.1pp |
| 147 | Two-word blank space | PARITY | ~0 |

**Read:** exotic stream tricks do not close the word–BPE gap; several actively hurt.

---

## 148 — alphabet BPE + FP

- Verdict **HARM**. Alphabet/char-BPE path dead for claims until a tokenizer actually merges usefully.
- Out of scope for 150–155 (hparams-only).

---

## 149 — matched hparams (GPT lock all arms)

Ritual: batch16 / lr3e-4 / AdamW wd0.01 / 4L4H / d256 / 40k.

| Arm | STORY |
|-----|-------|
| word_fp | **19.4%** |
| ws_bpe_rand | **32.4%** |
| ws_bpe_fp | **33.6%** |
| func_bias_gpt | **22.9%** (~+3.5pp vs word) |

Gaps: BPE−word ~**+13–14pp**. **Confounder later fixed:** BPE mix lacked word’s fat0.75 → STORY inflated.

---

## Plan 150+ — method fix

Problem: SOTE hparams frozen early (~5k/100k), GPT got “its” ritual, sequential one-lever digs on muddy baselines, dim closed after flat@100k then data scaled.

**Hard rules:** one locked trunk; exact@1; negatives@N do not freeze axes for N≫N; claim only on matched axes.

**Trunks:**

| | G | S (hist) | S+ (headroom) |
|--|--|--|--|
| batch/lr/opt | 16 / 3e-4 / AdamW | 8 / 1e-3 / Adam | same as S |
| depth/d | 4L4H / d256 | 2L4H / d256 | **4L8H / d512** |

Within a trunk: word ↔ BPE matched (fat, holds, seeds, BPE tok@N).

---

## 150 — dual trunk (unit × ritual)

Fat-matched STORY gaps (BPE−word):

| Trunk | Word STORY | Gap |
|-------|------------|-----|
| **G** | 19.5% | **+9.1pp** |
| **S** | 18.6% | **+4.3pp** |
| **S+** | 14.7% | **+6.9pp** |

- S+ word **−3.9pp** vs S @100k → **default_primary = S**; dim **not** closed forever.
- Fat matching **shrinks** gap vs 149 (~14 → 4–9).

---

## 151 — one-factor (G and S+)

Notable deltas vs trunk base (pp STORY):
- **G:** lr→1e-3 **+2.2**; Adam **+1.1**; batch8 **−2.1**; heads8/layers2 small +.
- **S+:** layers4→2 **+4.8**; AdamW **+2.2**; batch16 **+1.4**; lr3e-4 **−2.1**; heads4 **−2.0**.

**Read:** batch8 not universal under G; depth excess on S+ hurts; don’t sum 151+152 deltas — combine in a new matched run.

---

## 152 — context / budget (S+)

- tpw≈1.26; BPE maxlen16 / budget16 gaps still **+7.0…7.3pp**.
- Gap **not** from long BPE window. Mild fat0.45 note.

---

## 153 — N × batch (word)

Under **G**, **batch16** beats batch8 for STORY. Batch8 ≠ universal SOTE law.

---

## 154 — capacity × data

Headroom (S+ vs thin) lifts:
- word: **−4.4pp @100k**, **−3.7 @460k**
- BPE: ~flat (−0.4 / −0.6)

`premature_close` framing: headroom **hurts** here; prefer thin S+ / geometry, don’t blindly ↑d.

---

## 155 — eval surface

Note only: word = argmax; BPE = greedy piece→word. Residual decode surface; no oracle boundary CE shipped.

---

## Residual gap theory (post-155)

Ordered suspicion for remaining **~4–7pp** matched gap:

1. **Unit / composition** (word atoms vs reusable pieces)
2. CE density (last-pos word vs full-seq BPE)
3. Eval surface
4. V size  
**Not:** context length alone / single hparam / “need more dim”

**Success criteria (still open):** (A) absolute word STORY ~33%, and/or (B) compress BPE−word gap to ≤~5pp under matched ritual.

---

## 156 — shared morph in one codebook

- Framing: codebook = one indexing entity; words kept + `$stem` + `+aff`.
- Mine via `crude_stem`+`AFFIXES` → **only 3 aff** (ed/s/ing), **420 stems**, V=5912, expand coverage **~8%**.
- Ctrl word S+ STORY **13.8%**; shared morph **12.9%**; lift **−0.9pp** → **PARITY**.

**Read:** soft morph add-on with tiny aff inventory does not help; inventory ≠ BPE-like.

---

## 157 — rare morph-only

- Drop rare decomposable word-ids; morph path only for those.
- STORY **11.8%**; vs ctrl **−2.0pp** → **HARM**.
- Dropping rare rows without richer composition hurts.

---

## 158 — ComposeLayer preprocess

- Offline compose index → LM on composed ids; word HOLD at eval.
- Same thin morph inventory as 157 path (V=5253).
- STORY **13.8%**, lift vs 156 ctrl **~0.0pp** → **PARITY**.
- **Read:** freezing compose as preprocess does not unlock STORY when the piece inventory is still 3-affix morph.

---

## 159 — BPE-like tails

- Char suffixes 1–4 by type-support; K=128 tails, 1531 stems, V=7148; expand coverage **~45%** (vs ~8% in 156).
- STORY **13.6%**; vs 156 ctrl **−0.3pp**; vs 156 shared-morph **+0.6pp** → **PARITY**.
- **Read:** richer shared-tail channel indexes more tokens but does **not** unlock word STORY under S+. Soft morph/tail expand in the same codebook is not the residual gap closer — next pressure is unit/geometry (or harder composition), not more soft disambig-style morph rows.

---

## Side branches (not LM gates)

| Branch | Doc / note | Role |
|--------|------------|------|
| **BDLM** | `sote_bdlm_product_layer.md` | Neuro-symbolic fact / product layer on dual-channel — after LM ceiling |
| Weights-as-exact-bank | `sote_weights_as_bank_note.md` | Side research; not SQL-in-weights product claim |
| **Geometry** | agreed post-queue | wide-shallow / pyramid / 2L×d512 vs 4L×d256 — one factor |

---

## What we believe now (working)

1. Matched ritual + fat shrinks the folklore “BPE forever” gap; residual is real but moderate.
2. SOTE primary ritual at 100k is **S** (2L d256), not deep/wide S+.
3. Context length / budget are **not** the main residual.
4. Soft morph (3 aff) **and** rich BPE-like tails (128, cov~45%) are both **PARITY** — shared-tail expand in-codebook ≠ STORY unlock under S+.
5. Do not bury SOTE on a 4–7pp matched gap — study unit/composition + **geometry** next.

---

## Pointers

- Plan: `results/plan_150_plus_clean_compare.md`
- Queue log: `results/_queue_150_158_log.txt` / `_queue_150_159_log.txt`
- Decisions: `results/stage14*_decision.json`, `stage15*_decision.json`
- Path replay (earlier): `results/sote_v2_path_replay.md`
