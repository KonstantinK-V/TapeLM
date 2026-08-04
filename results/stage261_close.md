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

`NL_QUERY_NWAY_ONLY` did not fire because the gate requires **fp+sem** 20-way ≥ **0.10** (got **0.09**); the signal lives on **fp-only**, which the overall branch does not name.

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
