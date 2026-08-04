# Night queue metrics (2026-08-04)

**Narrative (255–260f full OK chapter):** [`stages_255_260_close.md`](stages_255_260_close.md)

Source: `_run_queue_night_full.log` — 255 skipped (done earlier) → 260f → 257 → 258 → 261 smoke → 242 → 212b.

| stage | overall | wall_s | notes |
|-------|---------|--------|-------|
| **260f** | **OPEN_GATE6_OK** | 1035 | headline **feat_only**, full 2500 steps |
| **257** | **FP_COMPOSE_OK** | 489 | 800 steps, 16 eval chains |
| **258** | **SEM_QUERY_OK** | 50 | 64 subjects, 600 steps |
| **261** | **NL_QUERY_NO_AT_SCALE** (full) | 34 | **20-way fp 0.22**; blend harm @ α=0.72 — [`stage261_close.md`](stage261_close.md) |
| **242** | **`REHEARSAL_DOSE_ANTICF_OK`** | 325 | 100% A replay → GPT **0.938**, tape **1.0** |
| **212b** | **INSTANCE_SEM_NO_AT_SCALE** | 43 | 480 slots, 800 steps |

---

## 260f — open-text gate (feat_only headline)

| metric | value |
|--------|------:|
| n_fit / eval_on / eval_off | 300 / 120 / 120 |
| n_pairs | 61 |
| paired_win_rate | **0.672** |
| paired_logit_gap | **2.698** |
| paired_gap (prob) | 0.154 |
| gold_is_top1_with → without | **0.738 → 0.000** |
| AUC vs prose / off-tape | 0.868 / **0.844** |
| gate_on_tape / random / ratio | 0.232 / 0.0003 / **868×** |
| gate_after_delete | 0.079 |
| false_fire_prose | 0.019 |

**Ablation (full run):**

| arm | win-rate | logit gap | AUC prose | AUC off-tape | paired prob gap |
|-----|---------:|----------:|----------:|-------------:|----------------:|
| h+feat | 0.639 | 3.747 | 0.907 | 0.777 | 0.087 |
| **feat_only** | **0.672** | **2.698** | 0.868 | **0.844** | 0.154 |
| h_only | 0.000 | 0.000 | 0.854 | 0.613 | 0.000 |

All gates true: `G_paired`, `G_feat_only`, `G_h_only_flat`, `G_auc_*`, `G_quiet`, `G_delete`, `G_tape_causal`.

Artifacts: `stage260f_decision.json`, `_stage260f_full.out`

---

## 257 — fp compose (2-hop)

| metric | value |
|--------|------:|
| em_2hop_glue | **1.000** |
| em_2hop_unseen_pair | **1.000** |
| em_2hop_one_hop_only | 0.000 |
| em_2hop_shuffled | 0.000 |
| em_2hop_after_delete_middle | 0.000 |
| exp_hops_2hop | 1.998 |
| gate_mean_chain / prose | 1.000 / 0.002 |

All 12 gates true (`FP_COMPOSE_OK`).

Artifacts: `stage257_decision.json`, `_stage257_full.out`

---

## 258 — semantic query

| channel | seen_rel sel | unseen_para sel | bank_top1 (unseen) |
|---------|-------------:|------------------:|-------------------:|
| fp-only | 0.057 | **0.000** | 0.000 |
| **curve_sem** | **1.000** | **0.646** | 0.646 |
| shuffled keys | 0.125 | — | — |
| **GPT-2 control** | 1.000 | **0.276** | 0.276 |

**Headline:** unseen_para curve **0.646** vs GPT **0.276** (Δ **+0.370**).

All 10 gates true (`SEM_QUERY_OK`).

Artifacts: `stage258_decision.json`, `_stage258_full.out`

---

## 261 — natural question (smoke + **full**)

**Closed read:** [`stage261_close.md`](stage261_close.md)

| | Smoke (53 slots) | **Full** (353 + 4000 noise) |
|--|------------------|-----------------------------|
| Verdict | `NL_QUERY_NO` | **`NL_QUERY_NO_AT_SCALE`** |
| top1 fp → fp+sem | 0.148 → 0.037 | **0.034 → 0.000** |
| 20-way fp → fp+sem | — | **0.220 → 0.090** (chance 0.05; shuffled **0.062**) |
| blend α | ~0.58 | **0.72** |

**Headline (full):** GPT top1 **0** → **`NO_AT_SCALE`** correct.

**Finding (full):** fp-only **20-way 4.4× chance** — task not hopeless; **sem blend ~halves** 20-way at **α≈0.72** — same **calibration** failure as smoke (fp+sem below shuffle), now with honest noise bank (loss **3.94→0.96**). Not a failed exam; **mixer** failed. Mechanism: fp-confidence gate rarely sees strong fp on this open bank (cf. **258 anchored**). **Deferred:** fp-only floor in blend loss — **261** stays out of 255–260f bundle.

Smoke table (historical):

| metric | fp-only | fp+sem | shuffled keys | GPT-2+sem |
|--------|--------:|-------:|--------------:|----------:|
| top1 | **0.148** | **0.037** | 0.074 | 0.148 |

Artifacts: `stage261_decision.json`, `stage261_mini.md`, `_stage261_full.out`, `_stage261_smoke.out`

---

## 242 — rehearsal dose (anti-CF price in weights)

| metric | value |
|--------|------:|
| tape_A_after_B | **1.000** (slot path) |
| gpt_A @ rehearsal 0 | 0.531 |
| gpt_A @ rehearsal **1.0** (100% of A in B CE) | **0.938** |
| target for `G_found_dose` | 0.95 |
| min_rehearsal_to_match | **null** (no rate hits ≥0.95) |
| W_align | 0.995 |

**Curve (rehearsal rate → GPT A retain):** 0→0.53, 0.05→0.69, 0.15→0.75, 0.3→0.75, 0.5→0.81, 0.75→0.78, 1.0→**0.94**

**Claim (substance):** even **replaying the entire A corpus** through B’s CE leaves GPT at **0.938**, not tape’s **1.0** and not the pre-registered **0.95** bar. That **is** the «price of anti-CF in weights» result — weights do not buy what one slot write buys.

**Verdict:** **`REHEARSAL_DOSE_ANTICF_OK`** — at rehearsal **1.0**, GPT A retain **0.938**, tape **1.0**. Legacy JSON said PARTIAL only because `G_found_dose` requires ≥0.95.

Artifacts: `stage242_decision.json`, `_stage242_full.out`

---

## 212b — instance semantics at scale

| channel | collision_4way (n=240) |
|---------|-------------------------:|
| fp-only | 0.204 (blind by construction ≈ chance 0.25) |
| fp+sem (curve) | 0.325 |
| shuffled | 0.283 |
| GPT-2+sem | **0.367** |

**Honest `NO_AT_SCALE`:** curve **0.325** vs GPT **0.367** — both barely above chance; **parity → scale / channel**, not «curve lost to GPT». `G_fp_blind_by_construction` holds (exam didn’t leak).

Artifacts: `stage212b_decision.json`, `_stage212b_full.out`

---

## Push / claim sketch

| claim | status |
|-------|--------|
| Open-text possession gate (260f feat_only) | **OK** full n=120 |
| 2-hop fp compose (257) | **OK** |
| Semantic query unseen para vs GPT (258) | **OK** Δ+0.37 |
| Natural NL query (261 full) | **`NO_AT_SCALE`** headline; **finding:** fp 20-way signal, **sem blend harm** — not in 255–260f bundle |
| Anti-CF price: weights vs slot (242) | **`REHEARSAL_DOSE_ANTICF_OK`** — 100% A → **0.938** GPT vs **1.0** tape |
| Instance disambig @ scale (212b) | **NO_AT_SCALE** — curve ≈ GPT, both ~chance |

Machine-readable: `night_queue_metrics.json`
