# SOTE V2 Charter

**Status:** active specification (2026-07-24)  
**Parent:** V1 reserve frozen — `results/SOTE_V1_RESERVE_FROZEN.txt`  
**Rule:** do not build a palace on an unlabeled draft. V1 = labeled reserve. V2 = same axis + explicit slots/metrics.

---

## 0. One sentence

Same SOTE axis (word atoms, frozen letter→word where possible, next on `word_fp`, hops separate, closed codebook D); **change the contract of slots and metrics**, not the tokenizer or an end-to-end rewrite.

---

## 1. Foundation kept from V1 (do not rip out)

| Piece | Keep |
|---|---|
| Atom | space-split **word**; morph surfaces distinct unless morph channel ties them |
| Encode | letter→word (morph) frozen stack as default |
| Native learn | next / order on **fp sequences** |
| Memory | **hops** separate from next; observe_strict + binders |
| Decode | closed-set codebook D (grow lexicon; no open string invent) |
| Train craft | peak-restore, zero-shot scale, rehearsal / protect LIVE neighbors |
| Path LIVE | noun_rel (+ polarity) as primary next LIVE gate |

---

## 2. What V2 changes (the path already showed)

### 2.1 Roles, not only flat next

After a path frame (`on`/`to`), **rel** and **object (right)** are different tasks.

- Flat next on right → ~4–6% @1 even on **noun_rel** (Stage 77) — not only verb_ing.
- verb_ing made the hole obvious; V2 treats **right-after-rel** as its own slot for all path frames.
- V2: tag steps `left | verb | rel | right | other` (and polarity when present).
- Prefer **role-aware step** or separate right protocol / what_tail-style move — not one anonymous “next word”.

### 2.2 Metrics from day one

Always report on HOLD:

| Metric | Use |
|---|---|
| **hit@1** | primary claim for path/noun_rel freeze |
| **hit@5** | soft column; **candidates = codebook only** (not arbitrary strings) |
| **by role** | tables; never hide object death inside overall % |
| gap vs shuffle / chance@5 | keep honesty controls |

Interpretation (locked by Stage 76a):

- path @1 primary; @5 side column (already ≫ chance).
- object @1 dead, @5 ~4× chance → ranking/geometry, not total zero — soft metric allowed; exact-only alone is too harsh for that slot.

### 2.3 phrase_fp

Optional symbol / **fact key** (hops). **Not** primary next. Do not revive open phrase-LM (40/41/69).

### 2.4 Data

Architecture will not save rare `(verb, rel, right)`.

- First: path + polarity density.
- Verb frames: only when **right repeats** enough (dense triples), or route object via **hops** (Stage 75: retrieve ~81%).

### 2.5 Dim / depth

Not the first lever. Raise dim only if codebook or hop collisions force it.

---

## 3. Non-goals (explicit)

- New tokenizer instead of words  
- Phrase-LM instead of word-next  
- One end-to-end soup (next eats hops)  
- Logical NOT / NLI via next  
- Putting verb_ing object into LIVE gate while @1 wall stands  
- Overwriting V1 reserve weights in place  

---

## 4. Slot policy (V2)

| Slot | Success | LIVE gate? |
|---|---|---|
| path noun_rel next | hit@1 (+ report @5) | **yes** (V1 bar) |
| polarity next | hit@1 / hops | yes as now |
| verb_ing rel | hit@1/@5 by role | diagnostic / soft |
| verb_ing **right** | hit@5 soft and/or **hops retrieve** | **not** exact@1 LIVE until green |
| hops facts | joint hop2/3 (incl. dirty) | yes (62–64 band) |
| true-neg conflict | — | residual wall (65) |

---

## 5. Ordered roadmap (execute in order)

| Step | Name | Done when |
|---|---|---|
| **V2.0** | Freeze V1 reserve + this charter | artifacts + labels exist |
| **V2.1** | Baseline instrumentation: role + hit@1/@5 on V1 weights | Stage 77 report; no FT |
| **V2.2** | Eval contract in code: `sote_v2_metrics.py` role+@k | harness importable |
| **V2.3** | Right-after-rel protocol (all path frames, not only verb_ing) | right @5↑ and/or hops; protect path @1 / rel |
| **V2.4** | Dense rights data only if V2.3 needs it; hops-primary object path | policy choice logged |
| **V2.5** | Optional soft@5 train-signal on `right` only | must not collapse path @1 |
| **V2.6** | Revisit letter/morph harden **only if** collisions/OOV block V2.3 | labeled dig, not silent rewrite |
| **V2.7** | V2 band freeze when path+hops+slot policy all green | new FROZEN band |

---

## 6. Tier labels (anti-palace rule)

Every component must be tagged:

- **RESERVE** — V1 frozen, proven enough for its claim  
- **LIVE** — current v2 gate  
- **DRAFT** — exploratory; cannot carry silent weight under a LIVE claim  
- **RESIDUAL** — known wall; do not pretend fixed  

If unsure → **DRAFT**, not LIVE.

---

## 7. Immediate next

**Stage 77 = V2.1:** run role + hit@1/@5 baseline on `stage57_rehearsal_FROZEN` + 1k/2k HOLD; write `results/stage77_v2_baseline_*`; no training.

Then V2.2/V2.3 — role protocol — without touching encode freeze.
