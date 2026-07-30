# Extension pipeline — explored directions & open tracks

Directions **tested outside the v1 product claim** for variant A (213+). The shipping stack is 191–205 plus canonical memory (221–230, 226c).

---

## Closed: partial freeze inside `arc_enc` (Stage 216)

**Idea:** freeze `char_emb` + mean-pool; train only post-pool FF (linear vs GELU) for domain shift while preserving fp geometry.

**Result (2026-07-30, `_stage216_split_arc_ff.py`):** **`SPLIT_FF_NO`**

- After TinyStories FF-only training with **frozen emb**, min cosine(fp_old, fp_new) on probe words ≈ **0.18** (linear FF re-init).
- **GELU FF** path (emb frozen, FF trained) ≈ **0.67** — still large drift; neither matches gate “>0.95 linear”.
- **Conclusion:** partial FF freeze is **not** a reliable domain-adapt path for **stable lexical fp** without an explicit remap. Do not pursue as alternative to full `arc_enc` freeze (213) or external **W**.

**Stronger positive (213):** full **`arc_enc` frozen** + upper finetune → fp drift ~**10⁻⁷**.

---

## Closed: naive recency `ctx_fp` (Stage 214)

**Result:** **`RECENCY_CTX_NO`** — mean-pool **0.947**; any λ>0 **hurts** (wrong entity anchor in exam ctx).

---

## Weak / NO: first-pass domain adapter (Stage 215)

**Result:** **`DOMAIN_ADAPTER_NO`** — toy contrastive W did not meet gates (domain recall, old-bank retention). Superseded by **221** (remap on core vocab after explicit arc_enc shift).

---

## Closed for domain adapt: partial freeze branch

| Approach | Verdict | Use instead |
|----------|---------|-------------|
| Train FF only, frozen emb (216) | **Closed** | Full freeze arc_enc (213) or **W-remap (221)** |
| Train full arc_enc without remap (213-B) | fp drift ~0.36 | **W** on keys/queries or reindex slots |

---

## Open (scale): multi-domain `arc_enc` **pretraining**

**Idea:** pretrain `arc_enc` (or full P1) on a **mixture** (Wiki + Stories + Code + Med + …) once at scale; at inference **freeze arc_enc**; domain shift via **domain token / prompt / adapter on upper layers only**.

**Why it fits TapeLM:** fp vocabulary stays one geometry; no per-domain reindex if mixture covers scripts.

**Why not in this repo yet:** needs **data + compute** (same wall as 209 semantic B) — not fixable on RTX 3050 + frozen 191-P1 alone.

**Honest status:** **future track**, not falsified — **unattempted at scale** here.

---

## Active: fp-remap adapter (Stage 221) — **YES**

After **intentional** `arc_enc` shift (TinyStories finetune), learn **W** on **core vocab** (~800 words):

- Mean cos(**W fp_old**, fp_new) ≈ **0.997**
- Old fact recall: raw old bank **0.95** → after shift without W **collapses**; **W @ key_old** **0.783** vs oracle reindex **0.867** (≥80% of oracle gate)

**Verdict:** **`FP_REMAP_ADAPTER_YES`** — explicit remap beats “partial freeze” for keeping legacy slots under encoder drift.

Script: `_stage221_fp_remap_adapter.py`, `results/stage221_decision.json`.

*(Stage **214** is recency ctx — unrelated; adapter remap is **221**, not 214.)*

---

## Other extension verdicts (213–220 snapshot)

| Stage | Overall |
|-------|---------|
| 213 | ARC_ENC_FREEZE_PARTIAL (fp stable; wiki CE drop on TS-only upper) |
| 217 slow-endpoint | SLOW_ENDPOINT_NO |
| 218 snap hop | SNAP_HOP_NO |
| 219 stream decay | STREAM_DECAY_WIN |
| 220 sem sidecar | SEM_SIDECAR_NO |

---

*Update this file when extension stages change; link from `docs/EXTENSION_PIPELINE.md`.*
