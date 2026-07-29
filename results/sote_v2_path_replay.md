# SOTE V2 Path Replay (exact-only)

**Mode:** walk the **same ladder as V1**, more carefully.  
**No top-5 claims.** Metrics = **hit@1 (exact codebook)** + **role tables** + controls (shuffle/gap where V1 had them).  
**V1 weights:** RESERVE (`SOTE_V1_RESERVE_FROZEN.txt`). Replay may **copy** then train; never overwrite reserve in place.

---

## Rules

1. One layer at a time. Tag **LIVE / DRAFT / RESIDUAL / RESERVE**.
2. Every next-eval: print **by role** (`rel` / `right` / …). Overall % alone is not a gate.
3. Path LIVE gate = noun_rel (and polarity) with **hit@1**; `right` reported but not silent inside ALL.
4. No phrase-LM, no dim bump first, no logical NOT via next.
5. Hops stay separate from next.

---

## Ladder (same spirit as V1)

| ID | V1 echo | V2 replay dig | Gate (exact) |
|---|---|---|---|
| **R0** | — | Charter + reserve lock | docs |
| **R1** | morph/encode 9+ | Encode recon + morph surface NN | recon/in-lexicon high |
| **R2** | D 38/39 | Decode codebook word/phrase hold | ~V1 floors |
| **R3** | next 40/46/47 | Next hit@1 + roles on in-lex / long | path ALL & rel; **report right** |
| **R4** | order 48 | Order drop on L≥2 | ≥10pp soft |
| **R5** | external 53–59 | Mid → 1k ZS exact@1 + roles | path soft ≥25% @1; right residual OK |
| **R6** | scale 73 | 2k ZS exact@1 + roles | no >5pp path regress |
| **R7** | hops 61–64 | Ext hops noun/polarity (+ dirty) | hop2/3 floors; no verb_ing required |
| **R8** | — | Right-after-rel protocol (all path) | right @1 lift **or** hops-primary; protect rel/path |
| **R9** | band freeze | V2 band FROZEN | all LIVE green |

Top-5 may exist only as private diagnostic later — **not in gates or claims** for this replay.

---

## Status

| ID | Status |
|---|---|
| R0 | DONE (charter + V1 reserve) |
| R1 | **PASS** Stage78 — encode recon 100%, word NN 100%, morph-ing NN 100% |
| R2 | **PASS** Stage78 — word/phrase exact 100%; holdout OOV 0% expected |
| R3 | **PASS** Stage78 — noun_rel ALL 29.3% @1, rel 54%, right **4.2% residual**; order_drop +14.5pp |
| R4 | **PASS** Stage79 — noun_rel order_drop +14.5pp |
| R5 | **PASS** Stage79 — 1k ALL 27.9% / path 29.3% @1; right residual |
| R6 | **PASS** Stage79 — 2k ALL 26.6% (d −1.3pp); path 28.1% |
| R7 | **PASS** Stage80 — hop2~84–88% hop3~93%; polarity≈nounish; dirty green |
| R8 | **PASS** Stage81 — **HOPS_PRIMARY_FOR_RIGHT**; next right@1 not LIVE |
| R9 | **DONE** `results/SOTE_V2_PATH_REPLAY_BAND_FROZEN.txt` |

Post-band dig: **Stage 82 RightStep FAIL** — RightHead +1.4pp (4.2→5.6%); confusion = other_rightish ~87%. Reinforces hops-primary. See `stage82_right_step_FAIL.txt`.

**Stage 83 PARTIAL (scale thesis):** fat noun_rel repeats → SEEN right@1 **3.3%→25%**, rel **75%**; RARE right stuck **~1.7%**. Repeats help seen; rare still hops/data. `stage83_nr_repeat_PARTIAL.txt`.

**Stage 83b PASS:** copies 30→80, steps 8k → SEEN right@1 **50.8% LIVE**; RARE still ~0%. `stage83b_nr_dense_PASS.txt`.

**Stage 84 WEAK:** real TinyStories chunk filtered a-z → ALL@1 1.9→8%, rel→78%, **right/obj ~0%**. Natural stories ≠ fat repeats. `stage84_tinystories_WEAK.txt`.

**Stage 85 PASS (new foundation):** digits+dim256+max_word_len24 — word/digit/morph/encode_B **100%**. V1 dim64 reserve untouched. `stage85_foundation_PASS.txt`. Next: capacity dig (no object-FT yet).

**Stage 86 WEAK:** TinyStories+85 foundation — SOTE ALL **8.6%**, rel **84%**, object **0%**; bigram ALL 24%. dim256 ≠ story unlock. Same freq cliff as hyp BPE. `stage86_capacity_WEAK.txt`.

**Stage 87 WEAK_LIFT:** TinyStories + fat path-triple repeats (150×60) on 85 — SEEN obj **0→22%** (missed PARTIAL≥25%), head **53%**, rel **84%**; RARE obj **11.7%**, natural STORY obj **~14%** / ALL **~7%**. Matches SEEN bigram right (~22%). 83b thesis on story-mined triples: repeats pull SEEN object; RARE/story still hard. `stage87_ts_repeat_WEAK_LIFT.txt`.

**Stage 88 PARTIAL:** SOTE word-id Transformer smoke (fp→D, CE next-id) on same 87 mix — SEEN obj **28%** (+6pp vs TinySeq), head **56%**, rel **84%**; RARE obj **13%**; STORY ALL still weak. Gates: PARTIAL≥25% hit, LIVE≥50% miss. Confirms axis: SOTE=dict/encode, head=Transformer. `stage88_word_tf_PARTIAL.txt`.

**Stage 89 FAIL:** same TF + **batch=32** SOTE word_id prefixes (lr 3e-4) — SEEN obj **24.7%** (miss PARTIAL), **rel 0%**, head 12%; STORY ALL **~12%** (up vs 88); curve swing still ~25pp. Batch≠free win: averaged steps underfit fat-SEEN lock that 88 hit. `stage89_word_tf_batch_FAIL.txt`.

**Stage 90 PARTIAL:** ablate lr only (batch32, **1e-3** like 88) — SEEN obj **28%**, rel **86%** (restored vs 89), head **57%**; ≈ parity with Stage88 SEEN; STORY ALL ~10%; swing still ~28pp. Low lr caused 89 collapse, not batch itself. `stage90_word_tf_batch_lr_PARTIAL.txt`.

**Stage 91 FAIL:** denser-up lr **2e-3** (batch32) — best SEEN obj **12.7%**, rel **47%** (PARTIAL miss); peak rel alone ~97% but never locks with object like 90. Swing smaller (~13pp) only because obj stayed low. **1e-3 sweet spot**; 2e-3 too hot without schedule. `stage91_word_tf_lr2e3_FAIL.txt`.

**Stage 92 FAIL:** SOTE schedule @1e-3 (warmup500 + cosine→0.1× + emb_lr×0.2) — SEEN stuck **~0%** rel/obj; STORY obj rose **~19%**. Too gentle for fat-SEEN lock that constant-1e-3@90 hit. `stage92_sote_sched_FAIL.txt`.

**Stage 93 PARTIAL:** warmup**200** then constant 1e-3 (full emb) — SEEN obj **33.3%** (>90), rel **74%**, head 54%; swing still ~33pp. Cosine/slow-emb were the 92 killers; short warmup alone OK / slight obj lift. `stage93_warmup_only_PARTIAL.txt`.

**Stage 94 FAIL:** same as 93 + **emb_lr×0.2** (no cosine) — SEEN obj **27%** but **rel 0%** (89-style). Slow fp-emb alone breaks rel lock; keep **full emb lr**. Cosine was extra poison in 92. `stage94_emb_slow_FAIL.txt`.

**Stage 95 PARTIAL:** batch sweep on recipe93 — **batch=8** wins SEEN obj **40%** rel **95%** (best so far); batch16 mixed/joint-unstable; **batch=64 dead** (0%). Swing still high (~38pp). Smaller batch helps fat-SEEN lock. `stage95_batch_sweep_PARTIAL.txt`.

**Stage 96 PARITY:** emb_lr **×2** on batch8 recipe — SEEN obj **40.7%** rel **96.7%** ≈ Stage95; emb drift vs F85 init large (~1.24 L2). Spectrum: ×0.2 kills rel; ×1–×2 both OK, no clear lift from faster. Stick **×1** unless re-check. `stage96_emb_fast_PARITY.txt`.

**Stage 97 PASS:** dual channel — atom from Stage95 batch8 (obj **40%** rel **95%**) + F85 SoftPhraseMemory/CueBinder hops on frozen 818 bank — clean hop2 joint **94.8%** (H1 99% / H2 96%); dirty_bind/hard weak (~17–19%, no dirty FT). No V1 dim64 binders. `stage97_atom_hops_PASS.txt`. Next: atom swing.

**Stage 98 STABLER_PARTIAL:** fat_frac=0.75 + patience on batch8 — SEEN obj **41.3%** rel **96%**; swing when rel≥50% **13.3pp** (vs ~38pp @95). Early-stop didn't fire (kept improving). Atom peak held + much stabler joint window. `stage98_swing_STABLER_PARTIAL.txt`.

**Stage 99 PASS:** dirty-hop FT (DirtyWhatRefine + Hop2@256) — clean/dirty_bind joint **95.8%**, dirty_hard **91.7%** (V1 floors cleared; ≥ Stage64). Note: Stage97 dirty~17% used harsher whole-cue noise; last-token dirty (V1-style) already strong ZS on F85 CueBinder. Atom ceiling note kept: TinyStories-mini ≠ hop graph limit. `stage99_dirty_hop_ft_PASS.txt`.

**Stage 100 PASS:** scale atom ~**98k** TinyStories windows (recipe98: batch8 / warmup200 / lr1e-3 / fat_frac0.75; 50k steps) — SEEN obj **42.6%** rel **95.8%** (parity with 98); **STORY ALL 19.6%** vs 98 **9.2%** (**+10.4pp**); RARE obj ~12% flat. Mini soft-ceiling was data scale, not recipe. F85 dual-channel FROZEN untouched. `stage100_scale_100k_PASS.txt`.


**Stage 101 CEILING:** ~460k TinyStories (recipe98) — SEEN obj **40.7%** rel **97.1%**; STORY ALL **20.6%** (lift +1.0pp vs 100). `stage101_scale_500k_CEILING.txt`.


**Stage 102 CEILING (capacity):** SEEN obj **41.6%** rel **96.3%**; STORY ALL **20.7%** (lift +0.1pp vs ref). `stage102_capacity_4L8H_CEILING.txt`.


**Stages 103–106 (after 101 CEILING):** fat0.90 obj=40.5% (PARITY_OBJ); fat400×60 obj=33.2% (CEILING); fat0.95 obj=38.4% (PARITY); **1M×4L8H** STORY=20.5% obj=38.6% (CEILING). Best SEEN obj: Stage 101=40.7%. `overnight_followups_decision.json`.


**Stages 107–108 (CE reweight dig):** corpus on|to in 70% windows; 107 full w(on)=0.14000000059604645 → SEEN obj 40.2% (PARITY); 108 rel_only w=0.15 → obj 40.1% (PARITY). `overnight_reweight_decision.json`.


**Stages 109–110:** slot-dyn CE → SEEN obj 42.2% (PARITY); TF baseline (rand emb) obj 42.2% (PARITY_BASE). `stage109_110_decision.json`.


**Stages 111–112:** fat0.45+slot_dyn obj=40.8% (PARITY); BPE baseline obj=41.3% STORY=34.0% (BPE_HIGHER). `stage111_112_decision.json`.


**Stages 113–114 (sample composition):** rel@0.5× → obj 42.6% (FAIL_REL); rare_long → obj 39.7% (PARITY). `stage113_114_decision.json`.


**Stage 115 (context):** win16/max24 → obj 38.8% STORY 15.0% (PARITY); max32@win8 ctrl → obj 42.1% (PARITY). `stage115_context_decision.json`.


**Stages 116–117:** morph diag STORY gap +0.1pp; hops_rerank PARITY obj=42.6%; hops_loss PARITY obj=41.8%. `stage116_117_decision.json`.

**Stage 120 NULL_MEM:** phrase-mem 2nd channel — STORY_LOCAL 20.8% → STORY_MEM 21.3% (gain **+0.5pp**); SEEN obj 28.8% rel 75.8%. SoftPhraseMemory boost not a STORY lever. `stage120_decision.json`. Next: A–D → H → E/F/G′.

**Priority digs 125–126 DONE:** C′ NULL; #5 PARTIAL obj 45.2% (+2.6pp), gap~40pp; G′ PARITY STORY 19.1%.

**128 ABORTED** → **130 compare** → **131 func-bias** → **132 selective mid/rare+aux** → **133 honest hop-rank@top5** (pure fp/mem, then FT fingerprint adapter; NOT filter+atom).
1. **One-page A–D** → `results/POST120_VERDICT_ABCD.txt`
   - **A** mem gain (`STORY_MEM − STORY_LOCAL`, Stage120)
   - **B** speed SOTE vs BPE (Stage119a)
   - **C** mean BPE toks / 8-word window
   - **D** rare-bucket word-level SOTE vs BPE
2. Dig **H** soft top-5 + morph-in-top5 diag on Stage100 (+ BPE beam@5) — **NOT a gate**; exact@1 claim unchanged. Artifact `stage124_H_top5_decision.json`.
3. Digs **E → F → G′** (sequential):
   - **E** who/where entity slots (refine 120) — strongest if A LIVE/WEAK
   - **F** fp projection adapter 256→256 — if geometry/rare still open
   - **G′** denser attn w/o BPE — **morph-root positions** (stem+affix as seq tokens); char-pieces fallback. Strongest if C/D LIVE.
4. Artifacts: `stage121_123_EFG_decision.json`, ckpts `stage121_E_*` / `stage122_F_*` / `stage123_G_*`.

**Hyp G′ (morph-as-input, not morph-as-eval):** Stage116 showed morph@1≈exact@1 (miss ≠ wrong inflection). Still open: change **atom input** — expand `w1…wk` → `[stem_i, affix_i?, surface_i]…` so attn gets BPE-like density + stem sharing; target stays whole-word exact@1 at word boundaries. Distinct from 117 hops-prior.


**Stage 118:** hop_conflict_rerank PARITY obj+hop=41.7%; hop_hard_mask PARITY obj+hop=42.2%; phrase_mix_bank PARITY obj+hop=40.9%. `stage118_decision.json`.


**Stage 119 (SOTE edges):** speed SPEED; freeze_emb EFFICIENT (rel90@0); curriculum PARITY (obj35@18000). `stage119_decision.json`.


**Stage 120 (phrase-mem channel):** STORY_LOCAL 20.8% → STORY_MEM 21.3% (gain +0.5pp); SEEN obj 28.8%. NULL_MEM. `stage120_decision.json`.


**Post-120 A–D + H/E/F/G:** see `POST120_VERDICT_ABCD.txt`; H NEAR_MISS_LIVE, E PARTIAL, F PARITY, G PARITY. `stage121_123_EFG_decision.json` / `stage124_H_top5_decision.json`.


**Priority digs 125–126:** C′ NULL; #5 PARTIAL obj_lift=+2.6pp; G′ PARITY STORY_lift=-0.5pp. `stage125_127_priority_decision.json`.


**Stage 129 composition diag:** func_hyp=SUPPORT_local_freq; hops_obj=UNLIKELY; bpe_only Δfunc=+13.8pp. `stage129_top5_composition.json`.


**Stage 132 selective morph+aux:** PARITY STORY=20.6% (vs100 +1.0pp). Completes mid/rare+aux CE principle. `stage132_selective_morph_decision.json`.


**Stage 133 honest hop-rank@top5:** PARITY atom_obj=45.2% pure_fp=17.0% ft_fp=41.3% lift=-3.8pp. `stage133_honest_hop_rank_decision.json`.


**Stage 134 codebook-tok + hops-assist (eval):** PARITY SEEN A=41.6% best_B=38.8% fact@1=3.8%. `stage134_codebook_tok_hops_assist_decision.json`.


**Stage 135 CE-only (hops OUT):** POLICY_OK SEEN obj=39.8% STORY=15.9% (no hops in train/eval). `stage135_ce_only_no_hops_decision.json`.


**Stage 136 WikiText word vs BPE:** GAP_SMALLER word=10.7% bpe=18.7% gap=+8.0pp. `stage136_wikitext_word_vs_bpe_decision.json`.


**Stage 137 dynamic bigram input:** PARTIAL word=18.2% bg=19.9% lift=+1.7pp. `stage137_dynamic_bigram_input_decision.json`.


**Stage 138 piece-fp CE / word confirm:** HARM word=18.6% piece=6.1% lift=-12.4pp. `stage138_piece_fp_word_confirm_decision.json`.


**Stage 139 BPE-tail-cut (keep words):** PARITY word=18.5% tail=18.0% lift=-0.4pp. `stage139_bpe_tail_cut_words_decision.json`.


**Stage 140 BPE-budget pack / word atoms:** GAP_SIMILAR word=18.4% bpe=28.6% gap=+10.2pp (budget=32). `stage140_bpe_budget_word_atoms_decision.json`.


**Stages 141-147 weird digs:** 141:HARM, 142:PARITY, 143:PARITY, 144:PARITY, 145:PARITY, 146:HARM, 147:PARITY. `stage141_147_weird_pipeline_decision.json`.


**Stage 148 alphabet-BPE + letter-fp:** HARM word=18.0% abpe=1.8% lift=-16.2pp. `stage148_alphabet_bpe_fp_decision.json`.


**Stage 149 matched hparams:** word=19.4% bpe_rand=32.4% bpe_fp=33.6% func_gpt=22.9% (batch=16 lr=0.0003 4L). `stage149_matched_hparams_decision.json`.


**Stage 150-155 clean compare pipeline:** primary=S headroom@100k=-3.9pp premature_close=False See `plan_150_plus_clean_compare.md` + `stage150_155_clean_compare_pipeline_decision.json`.

**F85 dual-channel FREEZE:** `results/SOTE_F85_DUAL_CHANNEL_BAND_FROZEN.txt` — foundation85 + atom98 + dirty-hop99 locked; V1/V2 FROZEN bands untouched. Stage100 is scale dig, not freeze overwrite.

**BDLM (agreed):** (1) next **product** layer on dual-channel; (2) **separate research branch after SOTE-as-LM ceiling** — not mixed into 150–158 gates. See `results/sote_bdlm_product_layer.md`.

**Side note (weights-as-bank):** exact data in weights/slots without classic index; text≠numbers roles — `results/sote_weights_as_bank_note.md` (not a live gate).

Band freeze (V2 replay only): `results/SOTE_V2_PATH_REPLAY_BAND_FROZEN.txt`. No top-5 claims.

---

## Non-goals for replay

- Soft@5 as success  
- Retrain everything end-to-end in one soup  
- Throwing out hops or word atoms  
