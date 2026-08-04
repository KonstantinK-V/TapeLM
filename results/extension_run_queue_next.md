# Queue

1. ~~**231–239**~~ closed
2. ~~**240–245** unexpected comparisons~~ — all measured

| Stage | Verdict | One-line |
|-------|---------|----------|
| **240** | **CF_VS_RAG_SURPRISE** | Frozen GPT emb index breaks after code-B (1.0→0.68); TapeLM keeps 0.95 |
| **241** | **WRONG_W_HURTS_OK** | Wrong-family W < no-W (0.77 vs 0.88); matched 1.0 |
| **242** | **`REHEARSAL_DOSE_ANTICF_OK`** | 100% A-replay → GPT **0.938**, tape **1.0** — not PARTIAL |
| **243** | **CARRIER_DRIFT_OK** | Same code-B: slots+W 0.98 vs weights 0.45 (gap 0.53) |
| **244** | **FORGET_CLEAN_OK** | Slot delete: tgt→0, ret/next_tok untouched; GPT unlearn collateral |
| **245** | **MIXED_NO_W_TIES_P1W** | Mixed-scratch raw 0.96 ≈ P1+W 0.92 (Δ+0.04, tie band) |

4. ~~**246** domain curriculum~~ — **`DOMAIN_CURRICULUM_DUP254`** (same as **254 joint**)
   - Tape mem wiki держит **0.88** до конца; GPT wiki PPL **16→347** (после stories пик ~29k).
   - Heads на wiki gen слабые (~0.30 nt) — mem OK, gen gate не прошёл.
   - Optional next: `--steps 30000` или сильнее head budget.

5. ~~**247** ingest forks probe~~ — **`INGEST_FORK_SLOTS_AND_HOP`**
   - P_slots: acq/cf **0.96**, edit collateral **0.04**, under **0.74**
   - P_hop: admit **0.83** / reject **0.08**, cf **0.83**, collateral **0**
   - P_ce: acq **0.75**, cf **0.71**, edit collateral **0.55**, under **0.61**
   - Hint: bindings → tape (+ hop gate); CE without bindings for understanding.

**Ingest understanding (251+):**
- **248–250** mem/hop OK; conclusion “masked CE cannot grow understanding” **withdrawn** (corpus was single-doc / degenerate mask; 251 confirms 1 doc vs 202029).
- ~~**251** CAL + CPC ladder~~ — **`CPC_UNDERSTAND_PARTIAL`** @4M tok/phase
  - CAL (plain CE, real corpus): exam **0.825 → 0.850**, holdout CE 4.28 → 4.13 — instrument moves, stack *can* learn.
  - CPC-only on upper: 179 gap **0.185 → 0.096** but exam **0.717**, holdout PPL **72 → 245**. Cause: CPC touches only `fast`; `slow`/`head` (which read `cat([fast, slow])`) never re-aligned.
- **252** **`JOINT_CPC_OK`** @4M/arm (~83 min): winners **λ=0.05, λ=0.2** (gates vs in-run λ=0 control, not frozen baseline).
  - λ=0 control: nt **0.808**, hold **4.184**
  - **λ=0.2**: nt **0.850** (=251 CAL peak), gap **0.185→0.137**, hold **4.20**, mem **0.95**
- **253** **`SCALE_JOINT_OK`** @16M λ=0.2 (~2h): nt **0.867**, hold **4.00**, gap **0.129**; beats 252 @4M on all gates; ckpt `checkpoints/stage253_joint_l02.pt`
- **254** **`CONTINUAL_UNDERSTAND_NO`** (~2h): mem **0.92+**; **G_no_forget_vs_P1 passes** (all domains better than frozen P1); **G_peak_hold_regress** fails (stories +0.47 vs end of stories phase); hop2 **0.44**; leak gate still borderline.
  - **Why 246 had to be redone:** it trained a fresh head per domain, so wiki `tape_next_tok` was the same 0.2998 in all four phases and every drop was exactly 0.0 — retention trivially true, growth structurally impossible.
  - 254 carries **one shared upper** through wiki→stories→med→news (joint CE+0.2·CPC, 25% replay, ≤3 epochs/domain), facts hop-gated into one growing canonical bank with bindings stripped from the CE text.
  - Gates: no language forgetting (hold CE drift ≤0.15), exam grows, slot recall ≥0.75 **against the full accumulated bank**, leak ≤0.40, cross-domain 2-hop ≥0.50, no collapse.
- **255 stream engine** (north star: chunked training + domain switch on small hardware), smoke done, full run waits for the GPU:
  - `python _stage255_stream_ingest.py --schedule wiki:8,med:6,news:6 --chunk-lines 25000`
  - Single pass, chunk dropped after use; bounded reservoir replay is the only past text kept; tape fp16 on CPU with blocked matmul; `--resume` + `results/stream255/STOP`.
  - Smoke **STREAM_INGEST_PARTIAL** (4 chunks, CPU): forget ≤0.02, recall 0.83 @146 slots, exam 0.825→0.800.
  - **Key lesson:** slot keys must all use one convention. Context-only entity keys (`ctx_fp`) are a generic direction that outscores subject-anchored probe keys — recall collapsed 0.50→0.00 as the bank grew. With `fp(anchor)+ctx_fp` the gate also stopped over-rejecting (+31/-9 vs +3/-37).
  - Gates: hold CE **vs P1**; recall **W_q-adapted** top1/MRR (frozen arc query is ablation only). `query_adapter.pt` per run.
  - `--no-query-train` ablation; `--lambda-admit-alpha` default **0.35** (meaningful λ drop at full entity cap).
- **254** needs **re-run** with local mask + **W_query** (`mem` canonical read); old JSON used shift-only mem + global mask.
- **Running:** wiki:12 **v2** → `--run-tag wiki12`. Log: `results/_stage255_wiki12_full.out`, decision `stage255_decision_wiki12.json`.
- **Queued after wiki:12:** `python _run_queue_after_wiki12.py` → `stream_wmn_v1`, then **`ablate_no_wq_wiki12`**, then **stage 257** (`_stage257_fp_compose.py` full). One-shot ablation+257: `--scale-ablation-only`. If ablation already running: `python _run_queue_after_wiki12.py --stage257-only` (poll → 257 → 258 smoke unless `--skip-stage258-smoke`). Skip glue stage: `--skip-stage257`. Logs: `_stage255_ablate_no_wq_wiki12_full.out`, `_stage257_full.out`, verdict `stage257_decision.json`.
- **259 hot swap (no wait):** `python _stage259_hot_swap.py --smoke` or `python _run_queue_after_wiki12.py --stage259-only`. Needs `stage256_slot_bias.pt`. Verdict `stage259_decision.json`, log `results/_stage259_smoke.out`.
- **Recommended parallel smokes:** `python _run_queue_after_wiki12.py --quick-smokes` → 259 → 257 `--smoke` → 258 `--smoke` while wiki12 ablation runs.
- **257 claim scope (fixed pre-run):** `results/stage257_claim_scope.json` → copied into decision; **0.6–0.7 EM on toy chains = mechanism win**, not general reasoning @ d256/6L.
- **258:** `_stage258_semantic_query.py` — `results/stage258_claim_scope.json`; smoke after 257 in queue.
- **259:** demo-grade edit-in-memory — `docs/stages_258_259_plan.md`.

**Night queue (plan: 255 → 260c smoke → full 257/258 → 261 → 242 → 212b):**

```powershell
cd C:\Users\Kostya\sote-letter-assembly
# 255 ablation already done:
python _run_queue_night_full.py --skip-255
# full chain from 260b:
python _run_queue_night_full.py --skip-255 --from-stage 260c
python _run_queue_night_full.py --force   # rerun all
```

| Step | Script | Notes |
|------|--------|--------|
| 255 | `_stage255_stream_ingest.py` … `ablate_no_wq_wiki12` | **done** 12/12 |
| **260b** | `_stage260b_open_gate.py --smoke` | off-tape in train + gate sup; `gate_reads_tape`, `prior_260` |
| 257/258 | full | curve vs GPT on 258 unseen_para |
| **261** | `_stage261.py` | **not in repo yet** — queue skips with log |
| **242** | `_stage242_rehearsal_dose.py` | grid to **1.0**, `--rates`, **`dose_wall_s`** |
| **212b** | `_stage212b_instance_sem.py` | W_sem collisions; **`G_fp_blind_by_construction`** |

Log: `results/_run_queue_night_full.log`. **Push only after full runs.**

**Still open (separate):** **209** meaning / scale frontier.

**Autonomous night (leave machine):**
- Phase1: `python _run_branch_ingest_50h.py --hours 45` → `results/branch_ingest_50h/`
- Day2 (+~24h): `python _run_branch_ingest_day2.py --hours 24` → `results/branch_ingest_50h/day2/`
  (waits for phase1 leftover; `--now` to skip wait). Stop day2: `results/branch_ingest_50h/day2/STOP`

