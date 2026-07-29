# Plan 150+ — clean compare (post-149)

**Problem we are fixing:** sequential one-lever digs on muddy baselines; SOTE recipe frozen early (~5k/100k) then reused; GPT got “its” ritual; capacity (dim/heads) closed after a **single N** and then scale kept growing; context / fat-CE / eval surfaces never matched.

**Hard rules**
1. One **locked trunk** at a time. No “heads↑ then later AdamW on reverted heads.”
2. Exact@1 only. Hops OUT of LM train/gate.
3. Negatives at N=X do **not** freeze axes for N≫X (capacity × data).
4. Claim only on **matched** axes named in the stage; unmatched axes listed as residual confounders.
5. Do not start Stage N+1 until decision.json for N exists.

**Upstream:** Stage149 = GPT-like lock on A/B/C/D (batch16 / lr3e-4 / AdamW wd0.01 / 4L4H / d256 / 40k). Wait for table before 150.

**Pipeline (wired):** `_stage150_155_clean_compare_pipeline.py`  
- waits for `stage149_matched_hparams_decision.json`  
- runs 150→155 sequentially; resume skips stages with existing verdict  
- Trunk G word/bpe reused from 149; S and S+ trained; primary default = **S+**  
- log: `results/_stage150_155_log.txt` · rollup: `stage150_155_clean_compare_pipeline_decision.json`

---

## Locked surfaces (name them every stage)

| Surface | Word (SOTE) | BPE | Matched when? |
|--------|-------------|-----|----------------|
| Phrase window | ≤8 words | same phrases | always (data) |
| Model context | max_len **16** word slots | n_pos **64**, seq~**48** | Phase 152 |
| Piece inflation | 1 tok/word | ~1.1–1.25 tpw on TS/wiki | Phase 152 (budget) |
| Fat / CE | fat_frac **0.75** lock; CE on word next | same mix; CE on **all** BPE positions | Phase 152b |
| Eval | word argmax | greedy piece→word | report both; claim on word HOLD only |
| Opt / batch / lr / depth | historically SOTE vs GPT rituals | — | 149 + 150 |
| Dim / heads | often frozen **256**/4H after “no lift @100k” | same width in 112/136 | **headroom from 150 (S+); full N×d in 154** |

**Policy — capacity headroom early (yes):** do **not** wait until 154 to give SOTE spare width. Old mistake was freezing d=256 from a flat 100k dig, then scaling data under that ceiling. From 150, SOTE primary path uses a **capacity-positive** trunk; 154 remains the clean N×d factorial, not the first touch of dim.

## STORY confounders (checked / fixed in pipeline)

| Risk | Effect on STORY | Status |
|------|-----------------|--------|
| BPE uniform mix vs word fat0.75 | BPE sees more natural story → inflated STORY gap | **Fixed:** BPE samples fat/story with same `fat_frac` |
| Word fin_story=500 vs BPE=300 | unequal noise / variance | **Fixed:** both EV=300 FIN=500 |
| Eval BPE encode max_len=48 on 152a@16 | train/eval mismatch | **Fixed:** `encode_max_len` |
| `hash()` seeds | non-reproducible across processes | **Fixed:** `zlib.crc32` |
| BPE tok@100k used on 460k | UNK on large-N story | **Fixed:** tok per N in 154 |
| Reuse 149 G (no BPE fat) into 150 | ritual×unit table biased | **Fixed:** retrain G in 150 |
| Best-ckpt on `ev_story` | optimistic STORY; arm-dependent peak | **Logged:** `story_all_last` alongside best |
| Word last-pos CE vs BPE full-seq CE | denser BPE updates / multi-step | **Residual** (152c / known surface) |
| Word argmax vs piece→word greedy | different error surface | **Residual** (155) |
| d512 init from d256 fp | dead high dims | **Fixed:** noise-pad + renorm |
| CUDA nondeterminism | small run-to-run STORY jitter | **Residual** |
---

## Phase 0 — close 149
- Table: word_fp / ws_bpe_rand / ws_bpe_fp / func_bias_gpt under **one** GPT-like recipe.
- Verdict = unit±emb under ritual G only. Not “BPE forever.”

---

## Stage 150 — dual-trunk + SOTE headroom (unit × ritual × capacity)

Freeze recipes (not one lever). **Three trunks:**

| | Trunk G (GPT-like) | Trunk S (hist SOTE) | Trunk **S+** (SOTE + headroom) |
|--|--|--|--|
| batch / lr / opt | 16 / 3e-4 / AdamW+wd0.01 | 8 / 1e-3 / Adam | **same as S** |
| depth/heads | 4L / 4H | 2L / 4H | **4L / 8H** (запас; one capacity bump, not sequential folklore) |
| d | 256 | 256 | **512** |
| steps / warmup | 40k / 200 | same | same |
| window / max_len / n_pos | residual word16 vs BPE48/64 — logged | same | same |
| data | TS-100k seed=272 | same | same |
| hops | OUT | OUT | OUT |

Arms per trunk: `word_fp`, `ws_bpe_rand` (optional `ws_bpe_fp` on G and S+ only if budget tight).

**Roles:**
- **S** = folklore control (“what we actually froze”).
- **S+** = **default SOTE primary** going forward — ritual S + capacity reserve so later scale isn’t pre-capped.
- **G** = GPT ritual control (d256 for 150; G+ optional later).

**Output:** unit × ritual table; plus S vs S+ = **early capacity delta @100k** (preview of 154, not a close).  
If S+ ≤ S @100k: **do not close dim** — only note “no lift yet @100k”; keep S+ as primary for later N (154).  
**Stop misuse:** do not interpret S-only or G-only as universal unit winner; do not bump only word and leave BPE thin.

---

## Stage 151 — orthogonal 1-factor (two bases)

**Base A = Trunk G** + `word_fp` (GPT ritual purity).  
**Base B = Trunk S+** + `word_fp` (SOTE primary with headroom — prefer this for “what we ship next”).

On each base, change **exactly one** axis; return to that base lock before next:

1. heads (G: 4→8; S+: 8→4 as *down* control — skip 16 unless VRAM OK)  
2. layers (G: 4→2; S+: 4→2)  
3. batch 16↔8 (lr fixed to that trunk’s lr)  
4. lr swap toward the other trunk (G: 3e-4→1e-3; S+: 1e-3→3e-4)  
5. opt swap (G: AdamW→Adam; S+: Adam→AdamW+wd)  

**Do not** re-litigate dim inside 151 (already S vs S+ in 150; full N×d in 154).  
**Forbidden:** chaining winners into a new muddy “best so far” without re-lock.

---

## Stage 152 — context / token-budget match (primary = S+)

Isolate “BPE has longer wire + more CE steps” from unit. Run on **S+**; optional thin replay on G.

- **152a Word-window matched:** both arms, phrases ≤8; word max_len=16; BPE truncated to **same #content tokens as word** (no free 48). `n_positions` may be >16 (pos-table headroom only); train+eval encode cap = 16.  
- **152b Token-budget matched:** same phrases; pack/pad so mean CE positions ≈ equal (target budget B, e.g. 16 or 32); log mean_words / mean_bpe / tpw.  
- Eval claim = **word HOLD exact@1** only; piece-greedy (`eval_bpe_word_holds`) reported as diagnostic; encode_max_len must match train cap.

If tpw≈1 (TS), expect small effect — still closes the confounder on paper.

---

## Stage 152c — fat / CE surface (optional, after 152a/b)

Same trunk **S+**, `word_fp` vs `ws_bpe_rand`:

- fat_frac ∈ {0.75, 0.45, 0.0} **or** equalize CE: word CE on all prefix positions vs BPE CE only at word-boundary pieces.  
One change at a time. Goal: separate fat-lock from “CE denser under BPE.”

---

## Stage 153 — scale × batch (word only)

Two thin grids (word_fp only, no BPE):
- Lock **G** @ d256: N ∈ {20k, 100k} × batch ∈ {8, 16}
- Lock **S+** @ d512: N ∈ {20k, 100k} × batch ∈ {8, 16}

Thesis: batch8 was for small-N / fat-SEEN, not universal — and may interact with capacity.

---

## Stage 154 — capacity × data factorial (not first touch)

**Why still needed:** 150 gives S vs S+ only @100k. Full claim needs **N × d** so we don’t repeat “flat@100k ⇒ close forever.”

**Design:**

| Factor | Levels |
|--------|--------|
| N | 100k, **~460k or 1M** |
| d | 256, **512** |
| unit | `word_fp` primary; `ws_bpe_rand` on **S+ ritual** at both d — **same d both units** |

Ritual for the grid = **S+ opt/batch/lr** (SOTE primary), with capacity pairs:  
- d256 → 2L/4H (hist-shaped)  
- d512 → 4L/8H (S+ shape)  
Do **not** cross d512×hist-heads in the main grid; optional corner later.

Also run the **same N×d grid under Trunk G** if 150 gap is ritual-sensitive (else skip to save compute).

**Interpretation rules:**
- Flat d@100k + lift d@large N → old close was **premature**; S+ headroom was justified.  
- Flat both N → capacity not the bottleneck under this V/window (then can thin S+ back).  
- Word vs BPE interaction with d — report; never average away.

Optional 154b: V/param note — word V~5k vs BPE V~8k can mimic ceiling.

---

## Stage 155 — eval surface (thin)

Same checkpoints from 150/154: report word-HOLD for both; for BPE also oracle word-boundary CE if cheap.  
No new train unless discrepancy >ε suggests decode-only gap.

---

## Order (strict)

```
149 done
 → 150 G + S + S+   (unit × ritual; SOTE headroom starts here)
 → 151 1-factor @G and @S+
 → 152 context/budget on S+ (and G if needed) (+152c fat/CE)
 → 153 scale × batch on G and S+
 → 154 N × d factorial (proves or drops headroom)
 → 155 eval surface (if needed)
 → 156 shared-morph-in-codebook
 → 157 freq-gated rare morph-only
 → 158 ComposeLayer preprocess + LM
 → 159 BPE-like tails in codebook (rich char-suffix inventory)
```

**Default train going forward after 150:** Trunk **S+** (unless 150 shows S+ harmful vs S on SEEN *and* STORY — then fall back to S and document).

**After SOTE-as-LM ceiling:** separate branch **BDLM** (product + research) on dual-channel — see `sote_bdlm_product_layer.md`. Not part of 150–159 gates.

**Stance (agreed):** do **not** bury SOTE-as-LM on a moderate matched gap (~4–7pp). Study and understand — ritual, unit, CE/eval, capacity **geometry** (wide-shallow / pyramid vs deep-wide).  
**After current queue + analysis:** digs on layer shape (e.g. 2L×d512 vs 4L×d256 vs simple pyramid) under locked ritual; one factor at a time.

**Stage 160 (wired):** `_stage160_geometry_500k.py` — S★ ritual @ N≈460k; arms base / wide-shallow / thin-deep / pyramid_up / pyramid_down.

**Stage 161 (after 160):** `_stage161_extreme_geometry_500k.py` — envelope extremes: d128×2/4/8L, 8L×d256, 1L×d256/512, 2L×d768.

## Explicitly out of scope for 150–155
- Hops back into LM gate  
- Soft@5 gates  
- Alphabet/char-BPE until tokenizer actually merges (148 dead)  
- Unmatched capacity (word fat, BPE thin)  
- Morph-as-BPE **claims** inside 150–155 (hparams only) — morph digs start at **156**

## Success criteria for the *plan* (150–155)
We can answer without folklore:
1. How much of word–BPE gap is **ritual** vs **unit**? (150)  
2. Does SOTE need early capacity headroom @100k, or only later? (150 S vs S+; 154)  
3. Which single hparams move the needle on clean locks? (151)  
4. How much is **context/CE density**? (152)  
5. Is batch8 scale-specific / capacity-interactive? (153)  
6. Was **dim closed too early** relative to later scale? (154)

---

## After hparams — morph codebook (your framing + frequency fork)

### Shared understanding
**Codebook = one entity** for indexing the dataset into SOTE’s language (ids the LM sees).

Today word-path: almost every surface type is its **own** atom → ending `-ing` is not a shared unit; `running` / `playing` / `looking` are unrelated rows.

BPE-analog inside **our** codebook (what you mean):
- keep **words as atoms** (they stay in the book — indexing still knows the word);
- **also add** a data-driven inventory of **shared morph atoms** (`+ing`, `$run`, …) sized from the corpus (order-of-magnitude: hundreds–low thousands, not “one unique ending per wordform”);
- when indexing, reusable endings map to **the same id**, not to a fresh “word-like” row for the suffix alone.

That is: morphs join the **same** dictionary entity; they are not a second parallel system. Softmax / stream can then use shared endings the way BPE reuses pieces — without pretending we deleted the word list.

### Two pipeline digs (after 155; trunk = S+ lock from 150)

| Stage | Idea | Codebook | Indexing / train | Tests |
|-------|------|----------|------------------|-------|
| **156** | **Your reading** — shared morphs *in* the book | `V = words ∪ morphs_data` (words kept) | Prefer expand to `$stem`+`+aff` (+ optional surface) when data says affix is shared; endings = shared ids | STORY vs plain word@S+ and vs BPE@S+ |
| **157** | **Freq fork** (stronger BPE-analog) | Frequent surfaces keep word-atoms; **rare** surfaces **no** active word-id (morph-only path) | Rare forms only morph composition; eval still word HOLD (compose→surface confirm) | Isolates “composition without rare word row” |

156 alone can still leave ~5.5k word classes in play (soft residual).  
157 asks whether STORY lift needs **dropping** rare word rows, not only **adding** morph rows.

### Morph inventory (data-driven, not hand list only)
From train mix (same 100k seed as lock):
- mine stem/affix via existing `crude_stem` / `surface` where `surface(stem,aff)==word`;
- keep affixes with support ≥τ; stems with support ≥τ;
- cap morph add-on (e.g. top 512 affix/stem slots or budget so `V ≈ word_V + K`);
- log: `#shared_aff`, `#stems`, `%tokens_expanded`, collision with raw word strings (`+ing` vs word `ing`).

### Order (extended)

```
… → 155
 → 156 shared-morph-in-codebook (words stay)
 → 157 freq-gated rare = morph-only
 → 158 ComposeLayer preprocess (separate indexing layer → LM on composed ids)
 → 159 BPE-like tails (char suffixes by type-support; K≈128; longest-match)
```

**Pipeline files:**
- `_stage150_155_clean_compare_pipeline.py` — hparams (waits 149)
- `_stage156_157_morph_codebook_pipeline.py` — 156→157→158 (waits 150–155 rollup)
- `_stage159_bpe_like_tails_codebook.py` — rich shared tails (waits 158)
- `_run_queue_150_159.py` (alias `_run_queue_150_158.py`) — sequential queue

### Stage 159 — BPE-like tails in codebook

**Why after 156:** 156’s “data-driven” affixes collapsed to **3** (`ed`/`s`/`ing`) via `crude_stem`+`AFFIXES` — coverage ~8%. That is not a BPE-like shared-piece channel.

**Idea:** mine character suffixes (len 1–4) by **distinct word-type** support; keep top-K (`≈128`) as `+tail`; stems `$stem` under longest-match; expand `[$stem,+tail,surface]` when match. Words stay in the same codebook.

**Gate:** STORY vs 156 ctrl (S+) and vs 156 shared-morph. Same trunk S+.

### Stage 158 — Compose layer as separate preprocess (agreed perspective)

**Idea:** fully split **composition** from **sequence LM**.

1. **ComposeLayer (offline, dataset processing):**  
   word stream → canonical composition ids (`$stem`, `+aff`, optional frequent whole-word atoms).  
   Writes indexed corpus + compose codebook. LM never sees raw mixed rules ad hoc per batch.

2. **LM layer:** train CE only on composed id sequences (S+ trunk).

3. **Word identity:** kept for decode/confirm/HOLD exact@1 (and future hops API) — not necessarily every CE step.

**Arms:**
- `158_ctrl_word` — plain word (or reuse 156 ctrl if present)
- `158_compose_lm` — LM on precomposed corpus
- optional compare stub vs BPE@S+ numbers from 150

**Gate:** STORY word HOLD exact@1 vs ctrl.  
**Not:** hops in LM; not merging compose into hparam digs.

### What 156 is *not*
- Not “morph replaces codebook.”  
- Not char-BPE (148).  
- Not hops in LM.  
- Not claiming 132/139 already were this: those were close cousins; 156 locks **S+ hparams** + explicit “one codebook entity” indexing contract + matched STORY protocol from 150+.
