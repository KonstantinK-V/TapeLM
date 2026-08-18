# Stage 261 — mixer ablations (open bank)

**Baseline:** fp-only 20-way **0.220** → fp+sem **0.090**, α≈0.72, shuffled **0.062**, `NL_QUERY_NO_AT_SCALE`.  
Full bank: 353 exam + 4000 wiki noise, 177 eval, `--no-gpt-control`.

## Read the numbers, not the gate headline

Several early runs looked “OK” on **`NL_QUERY_NWAY_ONLY`** while **`read`** contradicted them.

| Trap | What went wrong |
|------|------------------|
| **`fp20` from init W_q** | Snapshot at **init** (before train) scored **~0.226** 20-way; **trained** W_q on the same run was **~0.10–0.21**. Verdict used init → **`sem_over_fp` looked neutral** while blend used a **weaker** fp leg. **Fixed:** `read.acc_20way_fp` = **trained fp-only**; `acc_20way_fp_init_Wq` logged separately. |
| **`NL_QUERY_NWAY_ONLY` without causal 20-way** | fix1: sem **0.107** but shuffled **0.039** (< chance) → not tape-causal on 20-way. **Fixed:** `NL_QUERY_NWAY_ONLY` requires **`G_tape_causal_20way`** and **fp ≥ 0.12**. |
| **Strong floor + co-train W_q** | fix1p: aux CE on **init** W_q + joint updates **collapsed** trained fp to **0.10**; max-fusion could not recover baseline **0.22**. |
| **α cap only at train** | fix1m needs **`_alpha_eval` cap** (in code) — eval α now **0.35** when capped. |

**Honest columns:** use **`read.acc_20way_fp`** (trained fp-only), **`read.acc_20way_sem`**, **`read.acc_20way_shuffled`**, **`read.sem_over_fp_20way`**, not bare gate booleans.

## Recipe matrix (full runs)

| recipe | fp 20-way (trained) | fp+sem 20-way | shuffled 20-way | α eval | overall (new logic) |
|--------|----------------------:|--------------:|----------------:|-------:|---------------------|
| **baseline** | 0.220 | 0.090 | 0.062 | 0.72 | `NL_QUERY_NO_AT_SCALE` |
| **fix1** (soft fp floor) | 0.209 | 0.107 | **0.040** | 0.69 | `NL_QUERY_NWAY_FP_ONLY` (was `NWAY_ONLY`) |
| **fix2** (+ RRF) | 0.209 | 0.085 | — | 0.69 | `NL_QUERY_NO` |
| **fix3** (+ 260f gate) | 0.209 | 0.079 | — | ~0.7 | `NL_QUERY_NO` |
| **all** (+ warmup / early-stop) | 0.209 | 0.079 | — | lower | `NL_QUERY_NO` |
| **fix1p** (strong floor + cap + max eval) | **0.102** | 0.113 | 0.051 | 0.35 | `NL_QUERY_NWAY_FP_ONLY` |
| **fix1m** (soft floor + cap + max eval) | 0.113 | **0.130** | 0.051 | 0.35 | `NL_QUERY_NWAY_FP_ONLY` |
| **fix1q** (800 fp pretrain, freeze W_q, then sem) | 0.102 | 0.124 | 0.040 | 0.35 | `NL_QUERY_NO` (fp never reached 0.12) |

**Substance:** No recipe **restores** baseline fp **0.22** and **beats** it with sem at open bank. Best harm reduction vs baseline blend: **fix1** (0.09→0.11) and **fix1m** (sem **0.13** but only because **fp co-training fell** to 0.11). **Cap α + max(fp, blend)** helps vs uncapped blend when fp stays strong; it does **not** replace a mixer that **backs off** when sem loses to fp on the same query.

## Code

`_stage261_nl_query.py --recipe {baseline,fix1,fix1m,fix1p,fix1q,fix2,fix3,all}`

Artifacts: `stage261_decision*.json`, `_stage261_full_fix*.out`, `stage261_mini_*.md`.

See also [`stage261_close.md`](stage261_close.md) (headline vs blend-harm finding).

## Tape-style (no α): rerank → dualkey → symkey

Concept paths in `_stage261_nl_query.py` (same 261 exam). Shuffled control **permutes keys only** (gold slot index fixed, as baseline).

| recipe | fp 20-way | tape 20-way | tape top1 | shuf 20-way | read |
|--------|----------:|------------:|----------:|------------:|------|
| **tape_rerank** (v2 metrics) | ~0.10 | **= fp**‡ | **0.27** | **0.056** | top1 shuf **0.23** → not causal at scale; **G_works** almost (0.27 vs 0.30) |
| **tape_dualkey** | 0.107 | **0.158** | low | 0.18 | max(fp, sem keys); sem > fp, still ≪ baseline **0.22** fp |
| **tape_symkey** | 0.096 | **0.141** | low | **0.034** | symmetric `K_fp+h_write` / `W_q+h_q`; causal shuffle, modest lift |

‡Rerank **open-bank top1** uses read head; **20-way** in JSON intentionally = **fp** on the 20-draw (rerank pool artifact fixed). For rerank lift on 20-way, need a separate metric (rerank within the 20).

**Takeaway:** Rerank lifts **top1** without blend harm to fp scores, but **shuffle top1 ~0.23** → read not tape-causal at open bank. Dual-key modest 20-way gain. Symkey best **causal** shuffle.
