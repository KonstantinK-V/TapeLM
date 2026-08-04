# Pre-publish frontier (210–212)

**Framing:** value is an **alternative substrate** with **measured parity** and a **complete map** of what was tried on internalization — not a single “beat RAG” headline.

**Verdict vocabulary**

| Verdict | Meaning |
|---------|---------|
| `THESIS_YES` | Property not reducible to “external RAG key” without changing the substrate |
| `ENGINEERING_ONLY` | Works but equivalent to external zero-train loop |
| **`THESIS_NO_AT_SCALE`** | Gates failed **@ frozen P1 d256/6L** with controls in JSON — **not** permanent falsification; see [`VERDICT_VOCABULARY.md`](../docs/VERDICT_VOCABULARY.md) + `stage210_212_matched_gpt_ladder.json` |
| `THESIS_NO` | **Do not use** — legacy alias = `THESIS_NO_AT_SCALE` for 210–212 |

**Execution order:** 210 → 211 → 212 (frozen P1, anti-CF, small trainable modules only).

**Shared invariants:** P1 frozen; disabled path bit-identical to P1; introspection signals detached from CE (188); each stage writes `results/stage2XX_decision.json`.

---

## Stage 210 — structured composition in forward, answer as tokens

**Goal:** composition inside the same forward as text generation, answer via CE head — not external cosine hop, not next-fp (207).

**Implementation sketch:** `SoftFollowBlock` + zero-init inject; train `W_inject`, gate, temperature; slots non-gradient; answer scored as tokens (203 protocol).

**Gates:** generalize on unseen chains; free-form overfits; no CE cost; bit-identity at gate=0; needs memory.

**Result (2026-07-29):** `THESIS_NO_AT_SCALE` — external cosine hops 1.0; token inject ~chance; matched-GPT ladder: `python _stage210_212_matched_gpt_ladder.py --run-gpt`; see `stage210_decision.json`.

---

## Stage 211 — beyond-window internal slow tape vs endpoint

**Goal:** addressable slow-channel buffer across documents vs endpoint-only memory.

**Gates:** internal beats endpoint; near external slots; beyond GPT window; not reducible to doc-id oracle.

**Result (2026-07-29):** `THESIS_NO_AT_SCALE` — internal 0.23 < endpoint 0.28; external 1.00; **gpt_incontext 0.26** (matched control in-run); see `stage211_decision.json`.

---

## Stage 212 — instance / content-invariant channel

**Goal:** collision disambiguation + para/hard inversion via a small instance channel on the same tape.

**Gates:** collision accuracy; para > hard (bonus); not metadata; beats soft rerank; no CE cost.

**Result (2026-07-29):** `THESIS_NO_AT_SCALE` — collision modest lift; para 0.75 < hard 0.94; semantic parity vs GPT: cross-ref **209**; see `stage212_decision.json`.

---

## Intentionally not reopened pre-publish

| Skip | Reason |
|------|--------|
| Another fair-RAG grid | Covered 196–206 |
| Variant B / next-fp | Falsified 207 |
| Hybrid fp rerank | Closed 208 |
| PAWS scaling alone | 209 trajectory |
| “Beat RAG on raw recall” | Wrong headline for this work |

---

## Frontier status (210–212)

All three internalization bets **failed gates @ d256/6L** for distinct reasons — documented in JSON with **`THESIS_NO_AT_SCALE`**. **Not reopened as product path**; **not closed at all scales** (209: curve tracks matched GPT on PAWS). **P1 unchanged** (Δ next_tok = 0).

**Publication hook:** variant A as **one fp-space** (generation + memory + calibration + edit on frozen P1) **plus** the memory trunk (221–230, 226c). Internalization inside forward is **not** part of the shipping product; external fp loops and canonical memory are.

**Open hardware track:** stronger encoder pretrain + meaning objective (**209** parity trajectory at 3050 scale). Refresh matched controls: `_stage210_212_matched_gpt_ladder.py`.
