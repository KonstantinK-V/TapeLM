# Stage183 plan — Dataset-answer exam (not LM ceiling)

**Status:** DRAFT 2026-07-28  
**North star:** train may use non-text *principle*; **win** = model can answer what was in the dataset.  
**Not the win:** beat GPT on CE / A-wipe / paraphrase LM probes.

Related: `results/plan_curve_dynamics.md`  
Controls: Stage181 CE GPT (LM ceiling reference only).

---

## Two exams (do not mix)

| Exam | Question | Role |
|------|----------|------|
| **LM ceiling** (181) | How well does next-token CE use this corpus? | Reference only |
| **Dataset-answer** (183) | After training, can readout recover facts/links from hold data? | **Primary win** |

Hygiene (optional, not win): light A — memory not fully wiped.

---

## Protocol (smoke, same wiki 20M chars / hold split)

### Data build (S0)
From hold paragraphs, auto-build small packs (no hand-label marathon):

1. **Cloze-fact** (N≈80–120)  
   - Take sentence with a rare-ish content span (capitalized entity / number / quoted title).  
   - Mask that span; 1 gold + 3 distractors from other docs.  
   - Item stores: context window (without answer), gold string, distractors.

2. **Doc-link** (N≈60)  
   - Pair two windows from same doc vs different docs.  
   - Task: same-doc? (binary) from pooled states.

3. **OOD negative** (N≈40)  
   - Cloze where gold is **not** in train corpus (synthetic / held-out wiki slice never trained).  
   - Model should not beat chance much — anti-hallucination check.

Save: `data/stage183_exam.jsonl` + manifest.

### Readout (S1) — thin, stop-grad backbone
For each candidate system (curve / hybrid / CE control):

- Freeze backbone.  
- Tiny classifier / scorer on final (or pooled) state:  
  - Cloze: score(context_state, candidate_emb) → pick max; report accuracy vs chance 25%.  
  - Doc-link: bilinear/cos on pair → AUC / accuracy.  
- Train readout **only** on exam train split of items (or few-shot linear); never CE on full LM vocab as the stage win.

### Verdict rules (smoke)

| Signal | Call |
|--------|------|
| Cloze ≥ chance+10pp on in-corpus AND OOD ≤ chance+5pp | `DATASET_ANSWER_SIGNAL` |
| Cloze ≈ chance on in-corpus | `NO_DATASET_ANSWER` |
| Cloze high on in-corpus **and** OOD | `LEAK_OR_PRIOR` (bad win) |
| Doc-link ≫ chance | supporting `DOC_BINDING` |
| Beats 181 CE on same exam | interesting, not required |
| Loses to 181 on exam but A pretty | substrate ok, teacher weak for facts |

**Do not** declare failure because B_FORM or ablΔ ≪ GPT.

---

## Systems to score (same exam)

| Tag | Backbone | Notes |
|-----|----------|-------|
| `ce_gpt_181` | Stage181 ckpt | LM ceiling on **this exam** |
| `hybrid_182` | Stage182 ckpt | text-CE on slow (compromise) |
| `dual_180` | Stage180 ckpt | non-CE dual (closer to north star) |
| later: `curve_vq` | geometric codes only | true non-text teacher |

---

## Stage map add-on

| Stage | Intent |
|-------|--------|
| **183a** | Build exam jsonl + chance baselines |
| **183b** | Freeze backbones → train thin readout → score table |
| **183c** | Mini report: dataset-answer vs LM-ceiling; next principle tweak |

---

## Explicit non-goals for 183

- Fluency / story / ALL%  
- Matching GPT ablation Δ  
- Declaring “understands language” from para/hard  
- Reviving 169 word-CE as north star  

---

## Files (target)

| Path | Role |
|------|------|
| `results/plan_stage183_dataset_answer.md` | this plan |
| `data/stage183_exam.jsonl` | exam items |
| `_stage183_dataset_answer_exam.py` | build + score runner |
| `results/stage183_decision.json` | table + verdict |
