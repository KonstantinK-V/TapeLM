# SOTE fp-language contract

**Status:** **V1 = RESERVE (frozen).** **V2 = active** (charter + instrumentation).  
**V1 reserve index:** `results/SOTE_V1_RESERVE_FROZEN.txt`  
**V2 charter:** `results/sote_v2_charter.md`  
**Rule:** do not build a palace on an unlabeled draft — V1 claims stay; V2 adds slots/metrics.

**Not a freeze of generative LM.**  
**IO edges frozen in spirit:** encode B (text→fp) + decode D (fp→text NN). Hops/memory stay the answer path until fp-seq gates go green.

---

## 1. Claim

Language *inside* the model is not BPE and not raw characters.

```
text ──encode──► fp-seq ──(learn here)──► fp-seq ──decode──► text
```

- **In:** Letter/Word/Morph/Phrase composers (frozen stack where possible).
- **Native tongue:** sequences of `word_fp` (primary), optional `phrase_fp` / slot keys (memory objects).
- **Out:** closed-set NN codebook (Stage 38 D). Char-head = orthography probe only, not the trunk.

Learning (next-fp, masked-fp, contrastive) happens **on fp sequences**, not on token ids from an external LM vocab — though we *may* keep integer indices as handles into our own codebook (see §4).

---

## 2. Unit of thought / unit of training

### 2.1 What moves

A **thought step** is one advance in fp-space, not one byte.

| Layer | Unit | Role |
|---|---|---|
| Surface | characters | only at encode/decode edges |
| **Primary train unit** | **word** (surface match ↔ one `word_fp`) | next / mask / contrastive |
| Separator | **space** | explicit boundary; order of words = order of fps |
| Memory object | phrase / slot | hops, store, retrieve — not default LM token until separable |
| Graph step | bridge / continuation | protocol next among local candidates |

### 2.2 Order and spaces

- Phrase `"cat on mat"` → word sequence `["cat","on","mat"]` → fp sequence `[fp_cat, fp_on, fp_mat]`.
- **Space = hard segment boundary**, not a soft BPE merge.
- Order is left-to-right in that segmentation; shuffle of word order must hurt next-fp (control).
- Morph forms (`playing`, `played`) are **distinct surfaces** → distinct codebook entries unless morph channel explicitly ties them.

### 2.3 Explicit word match

Training targets are **exact surface words** in the lexicon sense:

- Gold next word `w*` ↔ `word_fp(w*)`.
- Success after decode: `NN(pred_fp) == w*` (same string), not “same stem” / fuzzy.
- Soft gate vs chance/shuffle stays (Stages 40+); not GPT-level exact on HOLD.

This keeps the loop honest: **same unit in, same unit out**.

---

## 3. Encode / decode edges

### Encode (already green)

`normalize → words.split(space) → word_fp / phrase_fp`  
Stage 37 B: recon 100% on in-bank strings.

### Decode (primary)

Closed-set **codebook**: `fp → nearest label` (Stage 38/39).  
Holdout OOV exact ≈ 0 for pure NN is expected — grow the codebook, don’t invent strings by default.

### Char readout (side channel)

Stage 42: word TF/exact above chance; phrase free-run exact ~0.  
**Allowed:** diagnostics, later OOV assist.  
**Not allowed (yet):** replace D as answer decode; open phrase LM.

---

## 4. Dataset after translation: indexing (not silly)

Translating the corpus into fp-language **should** produce an indexed table. That is the model’s lexicon, not a second BPE.

### 4.1 What to store

For each split (`train` / `hold`):

| Field | Meaning |
|---|---|
| `word_id` | dense index `0 … V_w-1` into word codebook |
| `word` | surface string (explicit match key) |
| `word_fp` | vector `[dim]` (or store once per id) |
| `seq_id` | utterance / phrase / chain id |
| `pos` | position in word sequence (space-segmented) |
| `phrase_id` (optional) | index into phrase codebook when row is a full phrase |
| `edge` (optional) | `(a_id → b_id)` graph continuations for hops |

Example row (conceptual):

```text
seq=17  pos=0  word_id=4   word=cat   fp=[…]
seq=17  pos=1  word_id=11  word=on    fp=[…]
seq=17  pos=2  word_id=9   word=mat   fp=[…]
```

### 4.2 Why index

- **Stable handles** for batching next-fp / masked-fp without re-encoding every step.
- **Decode = id → string** (and id → fp); training can use fp targets *or* contrast over ids in a shortlist.
- Growing language = append new `(id, surface, fp)` when a new word appears — same as a living codebook.
- Not silly: ids are *names inside our language*; vectors are the *meanings*. External BPE ids are what we refuse.

### 4.3 What not to do

- Don’t train a big softmax over 50k BPE pieces.
- Don’t throw away surfaces (need explicit match for D and reports).
- Don’t index phrase_ids as the only LM tokens while siblings collide (Stage 41).

---

## 5. Learning objectives (in fp-language)

All on indexed fp-seq unless noted.

1. **Next-word-fp** — prefix `[f0…f_{t-1}] → f_t`; decode NN vs gold word; gate vs shuffle (Stage 40 soft PASS on word).
2. **Masked-word-fp** — recover masked position from context fps.
3. **Contrastive / bag-neg** — true next vs scrambled order / wrong word in window.
4. **Protocol phrase-next** (optional) — only among `left == right(A)` shortlist + instance; not full phrase vocab (41 FAIL without separability).
5. **Memory hops** — unchanged contract: observe_strict, chain-link, what_tail, left_instance; answer text via D.

---

## 6. Soft gates (not GPT)

| Probe | Soft green | Red / defer |
|---|---|---|
| Encode recon | ≥95% in-bank | fix compose IO |
| D word/phrase NN | ≥95% closed-set | don’t claim decode |
| Next-word HOLD | gap vs shuffle ≥10pp, abs ≥25% | word dynamics not ready |
| Masked-word | beat chance by similar margin | — |
| Phrase shortlist next | gap ≥10–15pp in local set | keep phrase as memory only |
| Char free-run phrase | diagnostic only | never blocks A–D |

---

## 7. Curriculum (“return” path)

1. **Freeze narrative:** answers = hops + D; no phrase LM claim (post-41).
2. **Build indexed fp corpus** from hop bank / hold+train phrases (§4).
3. **Grow word-level fp-LM:** next + mask + order controls.
4. **Only then** phrase-as-token or char trunk — if siblings separable / word assembly works.
5. Open text without codebook = out of scope until OOV policy exists.

---

## 8. Out of scope (for now)

- GPT-style open generation from phrase_fp.
- Replacing hops with a decoder-only stack.
- Full-vocab phrase next as primary metric.
- Treating char-LM as the native language (chars are edge alphabet; words are tokens).

---

## 9. One-liner

**Space-split words are the training atoms; fps are the meanings; indices are our lexicon handles; text is only the border.**

---

## 10. Pointers

- D decode: `results/stage38_decode_contract_FROZEN.txt`
- D stress: `results/stage39_decode_stress_contract_FROZEN.txt`
- Word next soft / phrase full NN fail: `results/stage40_next_fp_PARTIAL.txt`
- Shortlist chance: `results/stage41_phrase_shortlist_FAIL.txt`
- Char probe: `results/stage42_char_readout_PARTIAL.txt`

### Indexed corpus (Stage 43 PASS)

- `checkpoints/fp_language_corpus.pt` — word/phrase surfaces+fps, `token_rows`, `seq_rows`, `edges`, `next_word`
- `results/fp_language_corpus_report.txt`
- `results/fp_language_corpus_meta.json`
- `results/fp_language_corpus_preview.jsonl`

Counts (hop25 train∪hold): **107 words**, **215 phrases**, **637 tokens**, **401 edges**, **422 next-word pairs** (232 train / 190 hold). D self-decode 100/100.

### Stage 44 minimal next-word (PASS soft)

- `checkpoints/stage44_fp_next_word.pt`
- `results/stage44_fp_next_word_report.txt`
- HOLD: acc **33.2%**, shuffle 12.1%, **gap +21.1pp**; D word 100%. Gates: gap≥10pp, abs≥25% — green.
- Confirms Stage40 word signal on indexed corpus; not GPT.

### Stage 45 masked-word + longer prefix (PASS soft, with caveat)

- `checkpoints/stage45_mask_long.pt`
- `results/stage45_mask_long_report.txt`
- Next HOLD all **35.8% / +24pp**; **L=1: 63.5% / +31pp**; **L≥2: 7.4% / +4pp** (no length gain — mostly first→second/rel).
- Mask HOLD **25.5% / +19pp** (soft floor); train 64%.
- Read: word-unit dynamics real; longer prefix not yet helpful on this bank.

### Stage 46 tiny seq (FAIL L≥2 → data)

- L≥2 **7.4% / +3pp** after GRU+pos on short phrases → template world, not dead fp-language.

### Stage 47 longer paths same lexicon (PASS/PARTIAL)

- `checkpoints/fp_language_corpus_long.pt` (+path2/path3); `checkpoints/stage47_long_seq.pt`
- HOLD: L1 **70.5%/+23pp**; **L≥2 32.8%/+22.5pp**; L≥3 **39.3%/+21pp** (was ~7%/+3pp).
- Order drop still ~+4pp. Data fix worked for L≥2 gap; order sensitivity next.

### Stage 47/48 FROZEN (axis status)

- `stage47_long_seq_FROZEN` + `fp_language_corpus_long_FROZEN`: L≥2 soft-green on long streams.
- `stage48_order_ctrl_FROZEN`: order drop **+24pp** (was +4); L≥2 held ~34%/+23pp.
- **Status:** next-atom on word fp-language learns at L≥2 when data has length; order can be trained in loss without new corpus.

### Stages 49–51 (diverse → mask → hop junction)

- **49** `fp_language_corpus_diverse.pt`: +adj/polarity/verb/list/morph seqs (1426 beyond-path). L≥2 **34.8%/+23pp** PASS; beyond-only **18.8%/+8pp** (harder). Order drop +20pp.
- **50** mask on diverse: HOLD **27%/+18pp** PASS soft.
- **51** junction: hops joint3 **95.8%** + atom on paths **42.5%/+25pp** PASS. Channels separate; no phrase/char trunk.

### Stage 52 deepen beyond-path (PASS/PARTIAL)

- `fp_language_corpus_beyond.pt` (+frame_grid / verb_noun_chain denser).
- Beyond-core L≥2 (excl. ambiguous `verb_rel`): **26.0%/+12.5pp** (was ~19%/+8). Path L≥2 **38%** held.
- `verb_rel` 3-grams stay ~chance; order drop on beyond still weak (~+5pp).
- Read: structured beyond frames learnable; open v-rel-n not yet.

### Stage 52 FROZEN + Stage 53 tiny external (PASS)

- Freeze: `stage52_beyond_contract_FROZEN.txt` (core beyond + path).
- External: `data/tiny_external.txt` (34 lines); grow D V=107→113 (`fox`,`hat`,`bed`,…).
- Zero-shot L≥2 weak (expected); **light FT → HOLD L≥2 46.4%/+32pp**, order drop +21pp.
- Claim: encode→next-atom→D axis transfers to tiny external with light FT; not zero-shot GPT.

### Stage 54 mid external scale (FAIL soft abs — wall)

- `data/external_mid.txt` **618 lines**; V=135 (+28); D 100%.
- Zero-shot L≥2 **27.6%/+17pp**; FT L≥2 **24.2%/+12pp** (abs &lt;25% gate; tiny was 46%/+32).
- FT did not lift vs zero-shot; L1 worsened. Wall=`FT_L2_soft` at hundreds of lines.
- **Stop scale here** before hops-on-external; dig FT/curriculum or narrower mid set — not phrase/char.

### Stage 55 curve dig (PASS — overshoot)

- Same 618; HOLD L2 every 150; restore peak; arms lr 1e-3 / 1e-4.
- **Peak lr=1e-3 @2850: L2 28.3%/+16pp ≥ ZS 27.6%**; end@3000 was 24.2% (overshoot +4pp = Stage54 fail).
- lr=1e-4: peak@0 (=ZS), never lifts. Curriculum not needed for abs gate.
- Residual: order_drop weak (+3pp); L1 still soft. Next optional: order harden / rehearsal — not 2k scale.

### Stage 56 order harden (PARTIAL)

- Parent 55; shuf-perm harden + tiered peak (3=L2≥28∧order≥10, 2=L2≥25∧order≥10).
- Hard PASS empty; **best soft: L2 26% / order_drop +11pp** (lr=5e-4 @1500).
- Tradeoff: order rises only as L2 leaves 28% band. Next: rehearsal before scale; L1 still separate.

### Stage 57 rehearsal (PASS hard gate)

- Parent 55; short path (len 3–5 on/to) + tiny×6; rehearse_frac=0.55; mild order.
- **L2 29.5%/+17pp, order_drop +12.9pp** (tier 3). Broad path≈mid failed earlier.
- L1 still ~9% (diagnostic). Next: L1 dig, then ~1k scale + peak-restore.

### Stage 58 L1 dig (WALL — majority ceiling)

- Parent 57; L1-heavy FT + L2/order protect. Majority L1≈9.3%; model 8.8%.
- No peak with L1≥15% without killing L2/order. Accept L1 residual; do not block scale.

### Stage 59 ~1k scale (PASS via zero-shot)

- `data/external_1k.txt` (1000). Parent 57 on HOLD: **L2 27.9%/+20pp, order +14pp** ≥25%.
- FT with order-max selection dipped to 25% &lt; ZS — restore parent. Scale holds without FT.
- L1 residual ~12%. Next: optional freeze / hops-on-external / kinds — not layers.

### FREEZE 57+59 external band (2026-07-24)

- Artifacts: `stage57_rehearsal_FROZEN.pt`, `stage59_scale_1k_FROZEN.pt`,
  `stage57_rehearsal_contract_FROZEN.txt`, `stage59_scale_1k_contract_FROZEN.txt`,
  `stage57_59_external_band_FROZEN.txt`.
- Claims locked: mid joint L2∧order; 1k ZS ≥25% L2 + order; scale ≠ fragile FT.
- Not claimed: L1≥15%, FT lift on 1k, hops-on-external.
- Next: **kinds dig B** (path vs tail on 1k) → then hops-on-external A.

### Stage 60 kinds dig (DONE)

- 1k HOLD ~99% path (on|to); no_rel n=3 (untested as tail).
- LIVE: path_short/long; subkinds **noun_rel** (29%) + **polarity** (35%).
- DEAD: **verb_ing** L2 18% (gap ok, abs soft-fail).
- Implication for hops-on-external: path noun/polarity facts first; skip verb_ing as first surface.

### Stage 61 dual-channel probe (PASS stretch)

- Fact bank: `data/external_facts_live.txt` (120 atomic LIVE facts; 354 pairs).
- Hop2 external **79.7%** joint (≥70% stretch); hop3 joint3 64.6% (diag).
- Atom LIVE L2 **28.1%/+14pp**. Dual green → expand fact bank next.
- Internal hop2 regression 83.3% held.

### Stage 62 expand bank (FROZEN PASS)

- `external_facts_live_exp_FROZEN.txt`: **818** facts (+698), pairs 11.5k.
- Hop2 **84.4%** (+5pp vs 61); hop3 joint3 **92.2%**; atom L2 held 28%.
- verb_ing still excluded. Freeze expanded bank.

### Stage 63 hop3 / polarity / junction

- Hop3 stress: joint3 **85.9%** (h2inst 87.5%); hubs mostly hub3p.
- Polarity (143 no/not facts): hop2 **93.8%** ≈ noun_rel 95.3% (Δ~1.6pp) — enough data; hops OK on negation keys.
- Junction rel-shortlist **44% ~ chance** — channels stay separate (as Stage51).
- Negation ≠ logical NOT; retrieval of polarity phrase keys only.

### FREEZE 62+63 external hops band (2026-07-24)

- Artifacts: `stage62_ext_hop_expand_FROZEN.pt`, `stage63_hop3_polarity_junction_FROZEN.pt`,
  `external_facts_live_exp_FROZEN.txt`, `stage62_63_ext_hops_band_FROZEN.txt`.
- Locked with 57/59: atom+1k ZS + expanded LIVE hops + hop3/polarity stress.
- Not claimed: verb_ing, NLI negation, junction gate.

### Stage 64 dirty hop (FROZEN PASS)

- Bank 818; binders seed 272; clean vs dirty_bind vs dirty_hard (no FT).
- Hop2: clean 84.4% / dirty_bind **86.5%** / dirty_hard **78.1%**.
- Hop3: clean 92.7% / dirty_bind **92.7%** / dirty_hard **91.7%**.
- Soft gates all green (walls: none). Mild noise ≥ clean; hard drop ≤7pp hop2.
- Artifacts: `stage64_dirty_hop_FROZEN.pt`, `stage64_dirty_hop_contract_FROZEN.txt`,
  band → `stage62_64_ext_hops_band_FROZEN.txt`.
- Next (optional): **true-neg dig** (logical NOT vs polarity-key retrieval); verb_ing deferred.

### Stage 65 true-neg dig (DONE wall)

- Cue `what_tail` with both `X rel Y` and `no/not X rel Y` in episode (same structured left).
- Conflict: pos **45%** / neg **54%** ≈ coin (n=96; 6 natural co-present, rest synth).
- Alone: pos_only/neg_only **99%** — Stage63 retrieval intact.
- Full-surface cue: pos→pos 75%, neg→neg 68% — soft, not clean disambig.
- **Wall:** no logical NOT; polarity = key retrieval only. Artifacts: `stage65_true_neg_DONE.txt`.
- Next: hold / verb_ing deferred / or other axis work — not NLI FT.

### Stage 66 verb_ing isolated dig (RESIDUAL)

- Same 1k HOLD as 60; FT only on verb_ing train (≤1200); protect noun/polarity ≤5pp.
- ZS verb_ing L2 **18.3%/+13.5pp**; FT best **17.5%/+14.3pp** — gap OK, abs &lt;25%.
- Protect held. Not hop surface. `stage66_verb_ing_DONE.txt`.
- Next: hold band 57/59+62–64; verb_ing stays residual unless new surface idea.

### Stage 67 -ing as separate atom (PARTIAL)

- Remap `looking`→`look ing`; TinySeq only (encode freeze).
- Split ZS cold 8.3%; FT+protect best **24.4%/+11.5pp** (+6pp vs opaque 18%).
- Peaks ~37% without protect crush LIVE neighbors — stop at protect.
- Idea helps a bit; still not soft LIVE / not hop bank. `stage67_ing_atom_DONE.txt`.

### Stage 68 verb_ing error + morph align (DONE)

- All **14** corpus `-ing` types already in `MORPH_LEX`; morph≈plain cos **0.997**, NN 14/14.
- L2 failures: **right/object 5.7%** (n=88); rel after *ing ~47% (on↔to); not “unknown -ing encode”.
- Object preds collapse to hubs (land/hand/bed/cap). Dig: `stage68_verb_ing_err_DONE.txt`.

### Stage 69 tail as one atom (FAIL)

- Remap `on|to + object` → one `phrase_fp` atom on verb_ing only.
- Remap/FT: verb_ing L2 **2.6%**, tail-target **0%** (vs opaque 18%). Same phrase-next wall as 40/41.
- Do not adopt. Object hole stays a word-level / disambig problem, not “one phrase chunk”.

### Stages 70–72 follow-ups (DONE)

- **70** ing-atom+rehearse: PARTIAL **23.5%/+10.6pp** protect OK (not soft 25%).
- **71** side-hint phrase_fp(prefix): FAIL (object 5.7%→2.3%).
- **72** object shortlist: FAIL (5.7%, |cands|~43, gap +4.5pp only).
- Best remaining signal still 67/70-style ing-atom under protect; object slot not fixed by hint/shortlist.

### RESIDUAL LOCK verb_ing (2026-07-24)

- overall vi next: **PARTIAL ~23%** (70); object-after-rel: **WALL ~6%**.
- Hops stay noun_rel|polarity. Trunk 57/59/62–64 untouched.
- 69/71/72 hypotheses dropped. Object only with new denser data, not same 1k.
- Artifact: `stage66_72_verb_ing_RESIDUAL_LOCK.txt`.
- Dual path+hops axis not weakened.

### Stage 73 ~2k scale ZS (PASS)

- `data/external_2k.txt` (2000). Parent57 ZS only (no FT).
- ALL L2 **26.6%/+15.1pp** (Δ1k −1.3pp); no_vi **27.9%**. Soft ≥25% held.
- verb_ing still ~18%. Scale did not break trunk. `stage73_scale_2k_FROZEN.pt`.

### Stage 74 dense (verb,rel,right) (FAIL)

- `external_verb_dense.txt` (224 triples × repeats). Object-slot ZS~6% → FT best **~12–14%** (≈chance/8 rights).
- Soft 25% not hit. Density ≠ unlock. Semi-cognitive: path+hops ok, verb_ing object still wall.
- `stage74_verb_dense_DONE.txt`. Residual lock unchanged.

### Stage 77 / V2.1 baseline + V2 Path Replay R0–R9 (FROZEN)

- V1 labeled **RESERVE**: `SOTE_V1_RESERVE_FROZEN.txt`. V2 charter: `sote_v2_charter.md`.
- Stage 77: role + hit@1/@5 diagnostic on 57 (`stage77_v2_baseline_*`); right-after-rel hole systemic.
- Path replay **exact@1 only, no top-5 gates**: `sote_v2_path_replay.md` →
  `SOTE_V2_PATH_REPLAY_BAND_FROZEN.txt` (Stages 78–81).
- LIVE: path next@1 + hops; **R8 = HOPS_PRIMARY_FOR_RIGHT** (next@1 right residual not LIVE).
- Harness: `sote_v2_metrics.py`.
- **Stage 82 FAIL:** RightHead(left+rel) on frozen TinySeq — right@1 4.2→5.6% only;
  dig confusion ~87% other_rightish. Soft ADD-on ≠ unlock.
- **Stage 83 PARTIAL:** fat noun_rel repeats (`external_noun_rel_repeat.txt`) —
  SEEN right@1 3.3→**25%**, rel→75%; RARE right~1.7% stuck.
- **Stage 83b PASS:** denser copies (80×, 8k steps) → SEEN right@1 **50.8% LIVE**;
  RARE still dead. Density unlocks seen object next; rare → hops/overlap.
- **Stage 84 WEAK:** real TinyStories (filtered) — ALL@1~8%, rel high, object~0%.
  Open story text + OOV lexicon ≠ 83b fat repeats.
- **Stage 85 PASS:** new foundation lineage — charset a-z+0-9+space, **dim=256**,
  max_word_len=24; word/digit/morph/encode_B gates 100%. Checkpoint
  `stage85_foundation.pt`. V1/V2 dim64 reserve incompatible (kept).
- **Stage 86 WEAK:** capacity on TinyStories (digits kept) + TinySeq@256 —
  ALL@1 8.6%, rel~84%, right/obj 0%; vs bigram ALL 24%. Adequacy: handles rel;
  fails rare/object/unseen. Same frequency cliff as hyp BPE on same data;
  dim256 did not close story capacity. `stage86_capacity_WEAK.txt`.
- **Stage 87 WEAK_LIFT:** fat repeats on TinyStories-mined path triples + story mix —
  SEEN obj 0→**22%** (gate PARTIAL≥25% miss), head→53%, rel 84%; RARE obj~12%;
  STORY natural still weak. SOTE SEEN right ≈ bigram (~22%). Confirms 83b:
  density pulls SEEN object; does not unlock rare/unseen. `stage87_ts_repeat_WEAK_LIFT.txt`.
- **Stage 88 PARTIAL:** word-id causal Transformer (emb init from SOTE fp, CE next-id)
  on same 87 mix — SEEN obj **28%** (+6pp vs TinySeq@87), head 56%, rel 84%;
  RARE~13%; STORY still weak. PARTIAL gate hit; LIVE miss. Smoke: SOTE as
  tokenizer/codebook + Transformer next-atom head is viable. `stage88_word_tf_PARTIAL.txt`.
- **Stage 89 FAIL:** batched CE (batch=32, lr 3e-4) on SOTE word_id prefixes —
  SEEN obj 24.7% (PARTIAL miss), SEEN rel collapsed to 0%, head 12%; STORY ALL
  rose ~12%; swing still ~25pp. Batch of atoms works mechanically; did not beat
  Stage88 SEEN lock. `stage89_word_tf_batch_FAIL.txt`.
- **Stage 90 PARTIAL:** same batch=32, lr **1e-3** (ablate vs 89) — SEEN obj **28%**,
  rel **86%** restored, head 57%; parity with Stage88 SEEN; STORY~10%; swing still
  noisy. Conclusion: 89 fail was **lr too low**, not batch-of-atoms. `stage90_word_tf_batch_lr_PARTIAL.txt`.
- **Stage 91 FAIL:** lr **2e-3** (denser-up vs 90) — SEEN obj 12.7%, rel 47% at best
  joint; denser batch ≠ free higher lr. Next: warmup/cosine at 1e-3, not hotter
  constant. `stage91_word_tf_lr2e3_FAIL.txt`.
- **Stage 92 FAIL:** warmup500+cosine+emb×0.2 @ peak 1e-3 — SEEN never locks
  (rel/obj~0); STORY improves (~19% obj). Fat-SEEN needs aggressive constant-ish
  lr; gentle SOTE schedule underfitting that mode. `stage92_sote_sched_FAIL.txt`.
- **Stage 93 PARTIAL:** warmup200→constant 1e-3, full emb — SEEN obj **33.3%**,
  rel 74%; swing still ~33pp. Short warmup OK; cosine+slow-emb were the problem
  in 92. `stage93_warmup_only_PARTIAL.txt`.
- **Stage 94 FAIL:** emb_lr×0.2 isolate (no cosine) — obj~27% but **rel 0%**.
  Slow emb alone kills SEEN rel; recipe stays full emb + warmup200. Next: batch
  sweep on Stage93. `stage94_emb_slow_FAIL.txt`.
- **Stage 95 PARTIAL:** batch {8,16,64} on recipe93 — winner **batch=8** SEEN
  obj **40%** rel **95%**; b16 unstable joint; **b64=0%** underfits fat-SEEN.
  Swing still ~38pp. Default batch → 8 for next atom work. `stage95_batch_sweep_PARTIAL.txt`.
- **Stage 96 PARITY:** emb_lr×2 on batch8 — obj **40.7%** rel **96.7%** ≈95;
  F85→task emb drift large but no clear gain vs ×1. ×0.2 still the failure mode.
  Keep emb×1 default. `stage96_emb_fast_PARITY.txt`.
- **Stage 97 PASS:** dual channel on F85 — WordId TF atom (batch8) SEEN obj40%/rel95%
  + SoftPhraseMemory hops via CueBinder (no V1 binders) — clean hop2 joint **94.8%**;
  dirty weak (~17–19%). Contract: atom≠facts. Next: reduce atom swing. `stage97_atom_hops_PASS.txt`.
- **Stage 98 STABLER_PARTIAL:** fat_frac 0.75 + joint patience on batch8 — obj **41.3%**
  rel **96%**; swing_when_rel≥50% **13.3pp** (was ~38pp @95). Stabler joint window
  without losing peak. `stage98_swing_STABLER_PARTIAL.txt`.
- **Stage 99 PASS:** dirty-hop FT @256 — dirty_bind/hard joint **95.8%/91.7%** (V1
  Stage64 floors); clean 95.8%. Mini-corpus STORY~10% looked like soft ceiling;
  hops are separate graph channel. `stage99_dirty_hop_ft_PASS.txt`.
- **Stage 100 PASS:** ~98k TinyStories windows + recipe98 (50k steps) — SEEN held
  (obj **42.6%** / rel **95.8%**); **STORY ALL 19.6%** (+10.4pp vs Stage98 mini).
  Mini soft-ceiling was **data scale**, not atom recipe. RARE flat. F85 FROZEN
  untouched. `stage100_scale_100k_PASS.txt`.

### Freeze (F85 dual-channel)
Band lock: `results/SOTE_F85_DUAL_CHANNEL_BAND_FROZEN.txt`.
Core FROZEN weights: `stage85_foundation_FROZEN.pt`, `stage98_swing_FROZEN.pt`,
`stage99_dirty_hop_ft_FROZEN.pt`. Do not overwrite in place; V1/V2 reserves untouched.
- **Stage 101 CEILING:** ~500k scale — STORY 20.6% (lift +1.0pp vs 100); branch→capacity. `stage101_scale_500k_CEILING.txt`.
- **Stage 102 CEILING (capacity):** STORY 20.7% (lift +0.1pp). `stage102_capacity_4L8H_CEILING.txt`.
Stage100 scale atom is a living dig (`stage100_scale_100k.pt`), not a freeze overwrite.

## 11. Target architecture (axis shared with next-token LMs)

```
text → encode → id/fp stream
              → (trainable) next-atom head
              → decode via index (codebook D)
facts / long memory → graph (hops), not stuffed into head weights
```

**Same axis as GPT-style next-token learning:** context → predict next atom → (soft) exact after decode.

| | Typical LM | SOTE fp-language |
|---|---|---|
| Atom | BPE / subword id | **word** (space-split) ↔ `word_id` / `word_fp` |
| Inside atom | embedding table row | composed fp (letter→word; frozen encode) |
| Train signal | next-token CE | next-fp (+ InfoNCE); exact after **D** |
| Soft green | — | gap vs shuffle ≥10pp (Stages 44+) |
| Long facts | mostly in weights / KV | **hops + memory graph** |

Difference is **what the atom is and how it is built**, not the learning game.

**Status on that axis:** atom = word; L=1 and **L≥2** next-atom > chance on longer in-lexicon streams (47). Order-control still weak. Phrase generative / char trunk stay off-axis. Hops = fact channel; head = local next-atom.

