# Stages 255–260f — closed story (full runs, gated)

**Status:** Five **`_OK`** verdicts with pre-registered controls — not a work-in-progress queue.  
**Trunk:** `stage253_joint_l02.pt` (joint curve + glue after stream ingest **255**).  
**Machine-readable:** `results/stage255_decision.json` … `stage260f_decision.json` · minis `stage*_mini.md`.

This page is the **narrative layer** the JSONs were missing. It does **not** open new stages; negatives and partials (261 open-domain NL, 212b instance sem, 242 rehearsal) stay in their own decisions.

---

## One paragraph (for README / abstract)

After **255** ingests ~54M wiki tokens into a **bounded** canonical tape without forgetting P1, the **253** joint trunk supports four capability claims with full gates: **(258)** on held-out subject paraphrases, curve+semantic query reaches **0.646** relation accuracy vs **0.276** for matched GPT-2+sem — the first **beat**, not parity, on a semantic probe at this scale; **(257)** external two-hop composition is **1.0** EM with **0.0** one-hop-only and shuffled controls; **(260f)** open-text “possession” is read from **five z-scored retrieval features**, with **h_only** paired win-rate **0.000** and random-key gate **~868×** below on-tape; **(259)** slot value hot-swap in **~116 µs**, **zero** gradient steps, keys bit-identical.

---

## Headline claim — Stage 258 (semantic query)

**Verdict:** `SEM_QUERY_OK` · wall ~50 s · 64 subjects · 600 train steps.

Each exam item fixes one **subject anchor** (shared fp) and asks which **relation** holds; the query paraphrases the relation with **no shared content word** with the stored line. **fp-only** is pre-registered at chance on unseen paraphrase (`G_fp_only_at_chance`); if that gate failed, the exam would leak.

| Channel | seen_rel (sel) | **unseen_para (sel)** |
|---------|---------------:|----------------------:|
| fp-only | 0.057 | **0.000** |
| **curve + W_sem blend** | **1.000** | **0.646** |
| shuffled keys | 0.125 | — |
| **matched GPT-2 + sem** | 1.000 | **0.276** |

**Why this is a separate claim (not a footnote to 256):** Stages **210–212** and fair RAG comparisons showed **curve ≈ GPT** on semantic probes — `NO_AT_SCALE` meant “the small GPU is not the bottleneck.” **258** adds a **matched GPT-2 control on the same blend recipe** and a **hard exam** (anchor-shared bank, paraphrase holdout). Curve **wins by +0.37** on `unseen_para` while fp-only stays at **0**. That is **operational semantics in fp-space** (frozen keys, train only `W_q`, `W_sem`, blend), not lexical overlap.

Controls (all gates true): sem beats fp on seen and unseen; bank-wide top1 tracks selective acc; anchored rows do not collapse; tape causal (shuffled keys); keys frozen.

Artifacts: [`stage258_decision.json`](stage258_decision.json) · [`stage258_mini.md`](stage258_mini.md)

---

## Stage 257 — two-hop composition (external loop)

**Verdict:** `FP_COMPOSE_OK` · 16 eval chains · 800 steps.

| Arm | EM |
|-----|---:|
| 2-hop glue (re-anchor on retrieved value) | **1.000** |
| one-hop-only baseline | **0.000** |
| head-only / unseen relation pair | **0.000** / **1.000** |
| shuffled / empty / no-edge1 bank | **0.000** |
| after delete **middle** edge | **0.000** (local causal) |

**Mechanism metric:** retrieval@cue reports hop0/hop1 top1 and chain_complete **without** trusting decode EM alone (cue tails end on **of** so greedy decode is not forced to emit `the`). Expected hops ≈ **2.0** on 2-hop queries vs **1.0** on 1-hop. Prose CE unchanged when glue runs (~3.99).

Artifacts: [`stage257_decision.json`](stage257_decision.json) · [`stage257_mini.md`](stage257_mini.md)

---

## Stage 260f — open-text gate (feat_only)

**Verdict:** `OPEN_GATE6_OK` · headline arm **feat_only** · 2500 steps · 300 fit / 120 eval on-tape / 120 off-tape.

Early **260** trained a gate on hidden state alone; shuffled-key AUC equaled real AUC — the gate ignored the tape. **260d–f** z-score **five retrieval scalars** (max, mean, margin12, max−mean, coverage) and ablate:

| Arm | paired win-rate | paired logit gap | reads tape? |
|-----|----------------:|-----------------:|:-----------:|
| h+feat | 0.639 | 3.747 | yes |
| **feat_only** | **0.672** | **2.698** | yes |
| **h_only** | **0.000** | **0.000** | **no** |

**Causal read:** `gate_on_random_keys` **0.0003** vs on-tape **0.232** (~**868×**); slot delete drops gate **0.232 → 0.079**; false fire on prose **1.9%** over 3129 positions. Feature probe: **gold_is_top1** **0.738 → 0.000** when gate features zeroed — possession tracks **retrieval geometry**, not prose logits.

Artifacts: [`stage260f_decision.json`](stage260f_decision.json) · [`stage260f_mini.md`](stage260f_mini.md)

---

## Stage 259 — hot swap (zero-train edit)

**Verdict:** `HOT_SWAP_OK` · glue `stage256_slot_bias.pt` · 4 held-out facts.

| Check | Result |
|-------|--------|
| EM before edit | 0.750 |
| new value after edit | **1.000** |
| old value after edit | **0.000** |
| neighbours | 0.750 (unchanged) |
| second edit (latest wins) | newest 0.750, superseded 0.000 |
| keys / glue params | **bit-identical** |
| gradient steps | **0** |
| mean edit wall time | **~116 µs** |

Update is **in-place slot value** with ctx_fp excluding self at write — not re-indexing keys.

Artifacts: [`stage259_decision.json`](stage259_decision.json) · [`stage259_mini.md`](stage259_mini.md)

---

## Stage 255 — substrate (stream ingest)

**Verdict:** `STREAM_INGEST_OK` · wiki **12** chunks · ~**53.6M** tokens · **2738** slots (**1.4 MB** tape).

Chunked single-pass ingest; canonical frozen keys; reservoir replay only for past text. Exam next-token **0.833 → 0.883**; hold CE vs P1 does not regress (`G_no_forget_vs_P1`). **W_q:** trainable query adapter per chunk (ingested-entity pairs); **recall gates** on held-out probe facts — keys stay frozen.

Artifacts: [`stage255_decision.json`](stage255_decision.json) · [`stage255_mini.md`](stage255_mini.md)

---

## How the five fit together

```text
255 stream ingest  →  bounded tape, P1 intact
        ↓
253 joint trunk    →  shared curve state for glue + query heads
   ┌────┴────┬────────────┬────────────┐
 257       258          260f         259
 2-hop     sem query    gate reads   O(1) value
 compose   beats GPT    retrieval    edit, 0 grad
           on unseen    feats        keys frozen
           paraphrase
```

| Stage | Question answered | vs matched GPT / RAG |
|-------|-------------------|----------------------|
| **258** | Does curve **semantic** query generalize paraphrase on unseen subjects? | **Yes — 0.646 vs 0.276** |
| **257** | Does **multi-hop** work as fp retrieval, not one-hop cheat? | Architecture (external hop API) |
| **260f** | Does open-text gate **read the tape**? | Architecture (feat-only causal) |
| **259** | Can facts **update** without training? | vs parametric finetune (197/205 family) |
| **255** | Can the tape **grow** under budget without CF? | vs continual weight training |

---

## Reproduce (full)

```bash
python _stage255_stream_ingest.py    # already OK; long run
python _stage257_fp_compose.py       # full (non --smoke)
python _stage258_semantic_query.py
python _stage259_hot_swap.py
python _stage260f_open_gate.py
```

Night queue reference: [`night_queue_metrics.md`](night_queue_metrics.md) (260f → 257 → 258 full order).

---

## Stage 261 — natural NL query (full; not part of 255–260f bundle)

**Verdict:** `NL_QUERY_NO_AT_SCALE` · full run · [`stage261_close.md`](stage261_close.md)

| Layer | One line |
|-------|----------|
| Headline | GPT and tape **0** open-domain top1 → scale label **`NO_AT_SCALE`** is correct. |
| Finding | **20-way:** fp-only **0.22** (4.4× chance **0.05**); shuffled **0.062**; fp+sem **0.09** at **α=0.72** — **blend destroys fp signal**; failed **calibration**, not “exam hopeless.” |

Do not fold 261 into the **258** headline; 258 is selective subject–relation; 261 is open bank + natural contexts. Deferred mixer fix documented in `stage261_close.md`.

---

## Explicitly out of scope for this bundle

| Stage | Verdict | One line |
|-------|---------|----------|
| **261** | `NL_QUERY_NO_AT_SCALE` (full) | See **[stage261_close.md](stage261_close.md)** — blend harm, fp 20-way signal |
| **212b** | `INSTANCE_SEM_NO_AT_SCALE` | Instance channel ≈ GPT @ 3050 |
| **242** | **`REHEARSAL_DOSE_ANTICF_OK`** | GPT **0.938 @ 100%** replay, tape **1.0** — [`VERDICT_VOCABULARY.md`](../docs/VERDICT_VOCABULARY.md) |

See [`extension_closed_branches.md`](extension_closed_branches.md) and [`docs/STAGES.md`](../docs/STAGES.md) for the full index.
