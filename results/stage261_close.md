# Stage 261 — natural NL query (full run, closed read)

**Verdict (headline):** `NL_QUERY_NO_AT_SCALE` · 353 exam + 4000 wiki noise · 177 eval · [`stage261_decision.json`](stage261_decision.json)

**Verdict (substance):** The **open-domain top1** exam is hard for everyone (GPT **0.000**, fp+sem **0.000**), so **`NO_AT_SCALE` is the right scale label**. The **interpretable finding** is different: **fp-only already carries signal**, and the **semantic blend destroys it** — a **channel-calibration** failure, not proof that natural query over a large bank is hopeless.

---

## Numbers that matter

| Channel | top1 (4353) | **acc_20way** (chance **0.05**) | blend α |
|---------|------------:|----------------------------------:|--------:|
| fp-only | **0.034** | **0.220** (~**4.4×** chance) | 0 |
| fp+sem | 0.000 | **0.090** (~half of fp-only) | **0.72** |
| shuffled keys | 0.000 | **0.062** ≈ chance | 0.72 |
| matched GPT-2+sem | 0.000 | — | — |

- **20-way** scores gold vs **19 fixed random distractors** (not full-bank top1). Shuffled keys at **0.062** → fp-only lift is **tape-causal**, not vacuous.
- Open-domain **top1** stays near zero for fp-only too (**0.034**); the bar “beat thousands of keys” is separate from “is there any ranking signal?”

---

## Read

1. **Task:** Not hopeless at the prior question. **Fp path alone** is **4.4×** 20-way chance with **4000 noise slots** in the bank — weak open-domain top1, but **not** random retrieval.
2. **Blend:** Training pushes **α ≈ 0.72** toward **W_sem**; **20-way drops 0.22 → 0.09** — semantic channel **halves** what fp-only had. Same pattern as **261 smoke** (sem below shuffle on tiny bank); full run with noise closed the shortcut but **did not fix calibration**.
3. **Mechanism (same family as 258 `anchored`):** The blend uses an **fp-confidence** gate to decide when to trust fp vs sem. On this exam, fp is **weak everywhere** (top1 **0.034**); fit never sees a reliable **“fp confident, retreat to fp”** regime. The gate cannot learn to **back off** when sem hurts — especially under open-domain noise where fp hits are sparse.
4. **What failed:** **Channel calibration**, not the exam design (wiki noise + natural write/ask contexts did their job: loss **3.94 → 0.96**, no tiny-bank overfit).

**258 contrast:** On the **selective** relation exam, curve+sem **beats** matched GPT (**0.646 vs 0.276** on unseen paraphrase) with frozen keys and a closed candidate set per subject. **261** is the **hardest** line: open bank, natural contexts — headline stays **`NO_AT_SCALE`**, but **do not** read it as “no signal”; read it as **“sem blend broke fp signal.”**

---

## Headline vs finding

| Layer | Statement |
|-------|-----------|
| **Headline / overall** | `NL_QUERY_NO_AT_SCALE` — GPT and tape both **0** top1; scale statement. |
| **Finding** | **20-way:** fp **0.22**; **blend harm** at **α=0.72**. Fix the **mixer**, not the bank size first. |

`NL_QUERY_NWAY_ONLY` did not fire on baseline because fp+sem 20-way was **0.09** (gate was **≥0.10**); the signal lives on **fp-only**. After mixer ablations, **`NL_QUERY_NWAY_ONLY` is tightened** (causal 20-way + fp ≥ 0.12) — see [`stage261_mixer_ablation.md`](stage261_mixer_ablation.md).

---

## Deferred (not in 255–260f chapter)

Minimal direction if revisiting: make **α → 0** a **live option** at eval — e.g. add a loss term penalizing **fp+sem** when it **loses to fp-only** on the same batch (or cap α when fp margin is low). **Intentionally postponed:** **260f / 257 / 258** are three full **OK** lines with controls; **261** remains documented here, not expanded into new stages until the closed chapter is shipped.

---

## Reproduce

```bash
python _stage261_nl_query.py          # full
python _stage261_nl_query.py --smoke  # small bank
```

Related: [`stages_255_260_close.md`](stages_255_260_close.md) (261 explicitly **out of scope** for the joint-trunk bundle).

---

## Line closed — 261f word votes (zero train)

**Verdict (261 line):** **`WORD_VOTES_BEATS_MEAN`** · [`stage261f_decision.json`](stage261f_decision.json) · script `_stage261f_word_votes.py`

The trainable semantic channel **failed on three trunks** (261 baseline + mixer/tape ablations + Qwen trunk swap): fp-only had 20-way signal, **blend always hurt**. The win came from **zero train** — one fingerprint slot per word, **IDF-weighted votes**, not averaging forty context fingerprints into one vector.

| | 20-way (strict `>`) | top1 | low-overlap top1 | tie_at_zero |
|--|-------:|-----:|-----------------:|-----------:|
| 261 ctx_fp **mean** (trained keys, still averages) | 0.226 | 0.034 | 0.000 | — |
| **261f word votes** | **0.432** | **0.246** | **0.024** | **0.488** |
| 261f soft + **15% typo** | *(rerun)* | — | — | — |

**Legacy note:** published 20-way **0.601** used `gold >= distractor`, which counted all-zero ties as wins. Ranks/top1 were already strict (`v > gold`). After the fix: 20-way **0.432**, popularity floor 20-way **0.021** (was ~0.29). Permanent **`silence`** block: `tie_at_zero_frac=0.488`; on low-overlap **`tie_at_zero_frac_low_overlap=0.864`**; of low-ov misses **`low_overlap_miss_is_silence_frac=0.885`** — the hole is silence, not bad ranking. When gold>0 on low-ov: **`top1_low_overlap_given_vote=0.174`**.

**Read:** Context averaging was the bottleneck; the fingerprint is a **character identifier** and works as an **address** in an inverted index, not as a term in a mean vector. **Low-overlap top1 0.024** is almost entirely **index silence** on those queries (86% gold=0), not a separate ranking failure.

**Silence → routing (next):** Low-ov is not a ranking failure — **`tie_at_zero_frac_low_overlap=0.864`** vs high **0.112**; of low-ov misses **88.5%** are gold=0. When votes fire, **`top1_low_overlap_given_vote=0.174`**. Silence is the switch signal for a semantic channel (258), not a blend (264).

**Controls:** Repointed postings (`popularity_floor`) preserve posting-count distribution. Causal read is on **top1**: signal **0.246** vs floor **0.000** (`G_causal_top1`). Signal still **beats** popularity on strict 20-way (`G_beats_popularity_20way`). **`G_open_top1`** (≥0.30) is false — headline stays **BEATS_MEAN**, not full open-top1 OK. **`G_low_overlap_works`** uses **`top1_low_overlap_given_vote`**, not raw low-ov top1.

**261 remapped via 266:** headline **`MIXER_OVERFIT`** / remap **`MIXER_DEFECT`**. On every trunk fit top1_sem=**1.000**, eval=**0.000** — mixer memorizes h→slot; `NO_AT_TRUNK_SCALE` was reading the broken channel. Healthy fp_only: Instruct **0.153→0.210**. Prompted keywords→votes remain the positive zero-train signal.

Detail log: [`stage261_262_research.md`](stage261_262_research.md).
