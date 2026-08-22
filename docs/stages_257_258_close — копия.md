# 257 / 258 — closed (smoke)

| Stage | Verdict | Evidence |
|-------|---------|----------|
| **257** | `FP_COMPOSE_OK` | `results/stage257_decision.json`, `_stage257_smoke.out` |
| **258** | `SEM_QUERY_OK` | `results/stage258_decision.json`, `stage258_mini.md` |
| **259** | `HOT_SWAP_OK` | `results/stage259_decision.json`, `stage259_mini.md` |

Smoke-scale. Full (non-smoke) optional confirmation only.

## Log excerpts (smoke)

### 257 — `_stage257_smoke.out`

```
decode … gold=Gregorian got=Gregorian  hop0=Elton p_stop=0.011 → hop1=Gregorian
decode … gold=Jadwiga   got=Jadwiga    hop0=Admiral → hop1=Jadwiga
decode … gold=Coast     got=Coast      hop0=Comedic → hop1=Coast

retrieval@cue 2hop:     chain_complete=1.0  hop0=1.0 hop1=1.0  halt_correct=1.0
retrieval@cue unseen:   chain_complete=1.0
retrieval@cue shuffled: chain_complete=0.0

overall=FP_COMPOSE_OK
em_2hop_glue=1.0  em_unseen_pair=1.0  em_shuffled=0.0  em_empty=0.0
exp_hops_2hop≈1.94  exp_hops_1hop=1.0   # G_stop_selective
```

### 258 — `stage258_mini.md` / decision

```
fp-only:   seen_rel=0.028  unseen_para=0.000  anchored=0.729
curve+sem: seen_rel=1.000  unseen_para=0.583  anchored=0.688
           alpha para=0.478 vs anchored=0.264   bank_top1=0.583
gpt2+sem:  unseen_para=0.250

overall=SEM_QUERY_OK
# all result gates true: unseen_para, bankwide, selective, anchored_intact, …
```

### 259 — `stage259_mini.md`

```
before=0.750 → new=1.000 old=0.000  neighbours=0.750
second edit: newest=0.750 superseded=0.000  empty_tape=0.000
keys bit-identical | glue params bit-identical | gradient_steps=0
overall=HOT_SWAP_OK
```

## How

**257 two-hop**
1. Measure **retrieval@cue** (hop top1 / chain_complete) separate from decode EM — first run had EM=0 with chase already 3/3.
2. Cue tails end on **of** (not bare **in**) so LM does not force `the`.
3. **Soft train** (StopGate needs mixture grads); **decode** = soft metrics + **span-lock** emit `tape.tok_ids` of answer hop (`hop_targets[-1]`).
4. Hard-commit-only train killed stop (`p_stop→1` on hop0) — do not use.

**258 semantic query**
1. Validity = exam only (fp@chance, keys frozen, tape causal) — not anchored regression.
2. Hold out **paraphrase B** of fit relations (`para_hold`), not new relation types (those were InfoNCE negatives).
3. Fit on `para` + `para_b` + **anchored**; blend L1 on anchored∧fp-hit; scaled/z-scored fp margin; hard-neg top-k + full-bank CE.

**259** — load 256 glue, `TapeView.with_value`, bit-identical keys/params, second edit wins.
