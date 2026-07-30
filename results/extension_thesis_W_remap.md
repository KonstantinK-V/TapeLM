# Thesis: W-remap as change-of-basis (221–223)

Working note — strong-track narrative + branch boundaries. Numbers from full CPU runs unless noted.

---

## 221 — structured fp space, not chaos

**Verdict:** `FP_REMAP_ADAPTER_YES`

| Metric | Value |
|--------|------|
| cos(W·fp_old, fp_new) on ~800 core | **0.997** |
| Recall via W @ key_old (221 protocol) | **0.783** |
| Oracle reindex (same run, SEED 221) | **0.867** |
| Bottleneck r=32 recall | **0.70** (~16k params) |
| Probe: hold-out / OOV after W | **~0.993 / ~0.987** (raw cos **~0.68**) |

**Claim (philosophy):** Continual learning often fights **weight** forgetting. Variant A contract: **weights may drift; slot contents need not.** W is not an anti-forgetting regularizer — it is a **change-of-basis** (approximate linear pushforward) on a **frequency core**, learned after the shift. Probe suggests **structure**, not a 800-word lookup table.

**Branch boundary:** W trained **after** a deliberate arc_enc shift (control). Production path is **freeze + optional W**, not “shift encoder routinely.”

---

## 222 — deploy modes; intra-run paradox

**Verdict:** `FP_DEPLOY_MODES_MIXED` · shift cos(word) **~0.68** · SEED **222** (fact set ≠ 221)

| Mode | Recall | Meaning |
|------|--------|---------|
| M7 oracle (new keys + new query) | **0.283** | Rebuild entire bank under **shifted** encoder |
| **M3 W @ keys only, old query** | **0.733** | Old bank + restore key-side metric |
| M2 old keys, new query, no W | **0.183** | New encoder queries, legacy keys |
| M5 221 (W keys + W old query) | **0.233** | Full legacy remap on both sides |
| M1 old / old | **0.483** | Baseline on **this** exam draw |

### Paradox (valid **within 222**)

Oracle **<** W-on-old-keys on the **same** 60 facts: reindexing with the **Stories-finetuned** encoder rebuilds keys in a geometry that **does not preserve** the discriminative layout the exam needs. **W** maps old keys back toward the **pre-shift wiki-era metric** that the slot bank was written in.

**Strategic reading (strong track):** To keep **domain-A facts** while adapting **generation** toward domain B: prefer **frozen slot bank + W** over **full reindex** under a B-shifted encoder — in this shift scenario, oracle reindex is actively worse.

**Caveats (branch / instrument):**

1. **221 on SEED 221** had oracle **0.867** and old/old **0.95** — same shift strength, different fact sample. Headline paradox must be reported with **fixed exam JSON**, not only RNG fakes.
2. One-off mismatch measure (old keys + new query **without** W) once showed **~0.97**; 222 M2 **0.18** — scoring path and seed differ. Deploy story needs **one frozen protocol** (222 modes on `stage191_exam_v3` subset).
3. M1 **0.48** shows this exam draw is **harder than chance** but **below** 221 baseline — compare like-for-like before paper-strong claims.

**Interpretation nuance:** W is not only “translator new→old.” It acts as **restorer of the old metric** under which episodic keys were written — sometimes more informative than the shifted encoder’s fresh fp for **legacy** facts.

---

## 223 — domain switch or universal unwarp?

**Verdict:** `DOMAIN_W_SWITCH_PARTIAL`

- Matched: B query + W_B keys **0.98**; C query + W_C keys **0.98**
- “Wrong” adapter: **≥0.97**; cross-drop **~0.017** (4-way noise scale)

**C in protocol:** wiki **P1 training windows** (`load_data` flat), not a third corpus. **B:** TinyStories. Both English narrative — **domians are close**.

**Hypotheses (open branches):**

| H | If true |
|---|---------|
| H1 Domains too close | W_B ≈ W_C; switch unnecessary on this pair |
| H2 W ≈ universal unwarp to **base** P1 frame | One W for all shifts toward “canonical” fp |
| H3 Registry W_domain | Only visible when shifts are **far** (code, med, …) |

**Needed test (224 sketch):** Same A-era slot bank; arc shifts on **code** / **med** / **legal** text (or synthetic char stats); cross-drop **matched vs wrong W** on **fixed** exam; align **‖W_B−W_C‖** and singular vectors.

**224 script:** `_stage224_far_shift.py` — Stories + synthetic Python + wiki-med lines; matrix recall[query domain × W adapter].

**224 full result:** `W_DOMAIN_PARTIAL` — max cross-drop vs best wrong W **~0.12** (code); stories drop **~0.03**; med matched **0.85** but stories-W on med query **0.90** (noise / weak med shift cos **~0.92**). W matrices differ (cos_flat **~0.43–0.53**). Not yet `W_REGISTRY_NEEDED` (≥0.20) nor `CANONICAL_W` (<0.05 everywhere). **Code** is the domain that most justifies a **separate W**.

If cross-drop stays **<0.05** on far domains → document **“canonical W”** branch (may merge registry). If **≥0.20** → **W_domain registry** becomes strong-track product architecture.

---

## Strong trunk vs branches

| Idea | Trunk | Branch |
|------|-------|--------|
| Change-of-basis memory migration | 221 + fixed-exam 222 | — |
| “Don’t reindex under B-shift” | 222 M3 vs M7 (fixed exam) | Oracle high in 221 → exam sensitivity |
| W_domain registry | After far-domain 224 | Universal W hypothesis |
| Mixed pretrain → less W | Open 224-mix | Toy B/C only proved mechanism |

**SOTE analogy:** Joint/hop **dead** for headline; **fp + shift structure** **reuse** for migration story — same pattern as tape from old collision work.

---

## Next experiments (minimal)

1. **222b:** All modes on **frozen** `data/stage191_exam_v3.jsonl` entity subset (no new fakes per seed).
2. **224_far_shift:** One non-prose domain + cross-W + **‖W−I‖** vs 221 W.
3. **Paper one-liner:** *External memory slots are basis-dependent; a tiny learned W realigns coordinates after encoder drift without rewriting the bank.*
