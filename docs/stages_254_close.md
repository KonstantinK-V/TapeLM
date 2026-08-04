# 254 — closed (smoke, operators-only)

| Mode | Verdict | Evidence |
|------|---------|----------|
| **Joint upper** (shared CE+CPC) | `CONTINUAL_UNDERSTAND_NO` | `results/stage254_decision_smoke.json`, `stage254_mini_smoke.md` |
| **Operators-only** (frozen P1 + W_query) | `CONTINUAL_UNDERSTAND_OK` | `results/stage254_decision_operators_smoke.json`, `stage254_mini_operators_smoke.md`, `_stage254_operators_smoke.out` |

Smoke: 150k tok/domain, 4 domains wiki→stories→med→news, bank 22 slots, 6 cross-domain 2-hop chains. Full (4M tok/domain) optional for joint upper retry only.

Product line: **operators-only** is the explicit “frozen upper + tape operators” continual path; joint upper remains the 253-scale experiment (forget/leak gates).

## Log excerpts (smoke)

### Joint — `stage254_mini_smoke.md`

```
overall=CONTINUAL_UNDERSTAND_NO
exam: 0.850 -> 0.825 -> 0.850 -> 0.800 -> 0.800
max forget vs P1 (hold CE): +0.432 | vs first phase: +0.322
min mem @bank 22: 1.000 | max leak: 0.400
cross-domain 2-hop: 4way=1.000 strict=0.667 n=6

gates: G_no_forget_vs_P1=false  G_peak_hold_regress=false
       G_understanding_holds=false  G_no_param_leak=false
       G_mem_holds_full_bank=true  G_cross_domain_hop=true
```

### Operators-only — `_stage254_operators_smoke.out`

```
operators_only=True  budget/domain=150000
baseline exam_nt=0.850  hold CE unchanged per domain after all phases

PHASE 1 wiki   … upper frozen — skip arc/W_bwd and joint train  exam=0.850 mem=1.00
PHASE 4 news   … bank=22  exam=0.850 hop2=1.000  all domains mem=1.00 hold CE = P1

overall=CONTINUAL_UNDERSTAND_OK
exam_curve=[0.85, 0.85, 0.85, 0.85]
hop: four_way=1.0 strict=0.667 hop1=0.833 n=6
forget vs P1 / vs first: 0.0 | leak Δ vs P1: 0.0
wall ~91s (vs ~10 min joint smoke)
```

## How

**Claim (254)** — One curriculum walks four domains with a **shared** memory story: facts → hop-gated slots on a growing bank; understanding = held-out CE + exam next_tok + cross-domain 2-hop over planted chains; no internal latent hop thesis (210/212 closed).

**Scoring (current code, `_stage24x_lib`)**
1. **mem** — canonical slot read via **W_query** + local inject/mask (`inject_and_mask`), full accumulated bank.
2. **Leak** — parametric probe vs **Δ vs frozen-P1 baseline** (same seed), not a fixed 0.40 absolute cutoff.
3. **Forget** — held-out doc CE vs P1 and vs value after first phase that domain was trained (`G_peak_hold_regress`).
4. **2-hop** — external fp loop across domains (subject from domain *i−1* appears in domain *i*).

**Joint upper (default)** — Per phase: strip fact bindings from CE text, admit slots, **train_joint** (CE + λ·CPC, replay), **W_bwd** arc shift. Mem can stay 1.0 while **weights move** → exam dip and hold CE regression (smoke: stories/med leak/forget gates red).

**Operators-only (`--operators-only`)** — **P1 upper frozen**; no joint train, no arc/W_bwd. Only **W_query** fits on wq_train facts; tape bank grows. Hold CE stays at P1 baseline; exam flat; leak Δ = 0. This is the smoke gate that matches the frozen-upper product framing (255 stream ingest + 256 glue + 257/259 operators).

**Run**

```bash
python _stage254_continual_understand.py --smoke --operators-only
# joint comparison (slow):
python _stage254_continual_understand.py --smoke
```

**Fix note** — `tape_recall_metrics` uses fp32 `K @ qq` (fp16/float mismatch had crashed recall on some paths).
