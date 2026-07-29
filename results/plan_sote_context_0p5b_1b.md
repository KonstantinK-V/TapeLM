# Plan — SOTE context dig @ 0.5B–1B (real scale)

**Status:** **FROZEN** (2026-07-28). Word-CE / battery-vs-majority path treated as dead-end for the original “leave text→text” goal.  
**Successor:** `results/plan_curve_dynamics.md` (Stage170 — train on curve dynamics; no letter/word CE gate).

**169 closing snapshot (Adam resume):** early-stop ~270k, bat≈32.6%, lift_maj≈−4.4 pp — still below majority. AdamW arm earlier: `S2_EARLY_OR_NULL` @90k.

**Why freeze:** objective stayed next-atom CE on word ids; letter-assembly only as frozen init. That cannot answer “LM on its own substrate.” Artifacts kept (`stage169_*`, ckpts) for reference; do not continue S3/S4 unless explicitly reopened.

---

## Where 168 left (closed chapter — small regime only)

| CP | Result |
|----|--------|
| CP0 | Rich wiki + battery OK |
| CP1 | Plain CE: **0L > 1L**; orthography alone ≠ context |
| CP2 | Ambig+contrast 1L & 4L: **LOSS_NULL**; battery ~7%, lift_maj ≈ −31pp |
| CP3 | Aborted ~40k/80k; same zone bat~5–7%, lift ≈ −32pp |
| CP4 | Not run |

**Read:** on tens of M tokens, next-atom CE stays last/majority. **Not** a verdict on SOTE at 0.5–1B.

---

## Goal (this plan)

Can **SOTE word atoms** at **0.5B–1B** tokens show **context use** — i.e. beat last-only and majority-from-last on hard probes — with a model sized for that data (not thin STORY lock).

**Not goals:** BPE race as primary; soft morph; “falsify on 50M first”; ALL% as success.

---

## Hardware + train ritual (RTX 3050 4GB) — locked from review

| Aspect | Lock |
|--------|------|
| Corpus on disk | Pretokenized codebook ids (`uint16` if V≤65k else `int32`) memmap |
| **Seq_len** | Prefer **512** for context; **S1 VRAM smoke:** 4L/d256, micro-batch **2–4**, seq **512** on 4GB — if OOM, **lock 256** for full run |
| **Vocab (V)** | **32k–64k** + `<unk>` for true rare/OOV only — log coverage in S0 |
| Micro-batch | 2–8 (whatever fits; expect **2–4** @ seq 512) |
| **Grad accumulation** | Default **effective 64** (micro=2 → 32 accum steps — slower but OK); if wall critical use **effective 32** |
| Primary arch | **4L / 4–8H / d256** (d512 if fits); thin 1L = control only |
| **LR / warmup** | **Adam, lr=1e-3, wd=0** (locked); warmup **2000–5000**; then constant or cosine |
| **Checkpoints** | Every **10–20k** steps (resume + curve analysis) |
| **BPE arm** | **Secondary** — only if time; main answer = vs 0L + majority |

---

## Data (the real lever)

1. **Source (primary):** WikiText-103 train — **~100M tokens** after SOTE rich filter (order of magnitude; log exact count in S0).  
2. **Fallback to reach 0.5B:** append **OpenWebText** and/or **extra wiki dumps** in stream order until budget **0.5B** atoms (then optional 1B). Document which sources contributed how many tokens in `stage169_s0_decision.json`.  
3. **Charset:** case + punct + digits + basic specials (168 token_re spirit).  
4. **Codebook:** top‑V word/punct atoms (V **32k–64k**) + `<unk>`.  
5. **Emit once:**  
   - `data/sote_ids_0p5b.bin` (or `_1b.bin`) — contiguous token ids  
   - `data/sote_codebook_*.json` — itos/stoi + meta + coverage  
6. **Target:** **0.5B first** (kill/go), stretch to **1B** if disk/time OK.  
7. **Battery:** **rebuild same-last from the large 0.5B stream** (mandatory; discard 168/50M battery).

**Counting:** 1 token = 1 codebook atom (word or punct piece), logged explicitly.

---

## S0 — codebook + pretok (explicit rules)

- Build vocab from **first pass** (or streaming count) on planned train mix; start V in **32k–64k** range.  
- **Coverage rule:** if train token coverage **&lt; 95%**, **increase V** (toward 64k) and rebuild — **do not** compensate with mass `<unk>` while keeping V small. Log `coverage`, `unk_rate`, `V` in decision.  
- Map stream → memmap ids; stop at 0.5B (or max reachable).  
- Emit battery JSONL from **same** large stream (same-last / ambiguous-last stats).  
- **Out:** bin, codebook, battery, `stage169_s0_decision.json`, mini report.  
- **Kill:** disk fail → report max reachable tokens; do not pretend 0.5B.

---

## Arms (matched ritual)

| Arm | Role |
|-----|------|
| `sote_4L_d256` (or best that fits 4GB) | **Primary** — capacity for context |
| `sote_0L_last` | Last-token ceiling |
| `sote_1L_d256` | Thin control (optional if time) |
| `bpe_matched` | Optional unit control — same steps/tokens budget; **secondary** |

Train from memmap; fat light or none (context dig, not fat STORY race).

---

## Gate = context (not ALL)

Primary metrics on HOLD + battery:
1. **Battery acc** and **lift vs majority-from-last**
2. **1L/4L − 0L** on battery and on order-drop  
3. ALL% logged only as secondary

**Success (SOTE made the step):**  
primary SOTE **≥ +5pp** vs 0L **and** **≥ +5pp** lift vs majority on battery (order-drop gap supportive).  

**Fail (honest):** after 0.5B (and optional 1B stretch) still 0L/majority win →  
`NO_CONTEXT_AT_0p5B_CE` — then next is **not** more 50M tricks; either more data/model, or non-CE objective / dual-channel task. No fake win on ALL.

---

## Pipeline steps (checkpoints)

### S0 — pretok (~2–6h wall, may grow with OpenWebText)
See **S0 — codebook + pretok** section above.

### S1 — smoke 20k steps on primary + 0L (~1–2h)
Sanity: loss falls, eval runs. **VRAM probe:** 4L/d256, seq **512**, micro-batch 2–4 — if OOM, set `seq_len=256` for S2/S3 and log in decision.  
**Kill:** unrecoverable OOM after seq/batch shrink → shrink d before abandoning scale.

### S2 — full train primary @ 0.5B budget (~1–3+ days depending steps)
Enough steps to see curve flatten on battery (order: **200k–500k steps** with accum; tune to wall).  
**Ckpt every 10–20k.**  

**Kill @ ~30% steps (honest early stop):**  
- **Stop** if `lift_vs_majority` **&lt; 0** *and* battery (or lift) curve is **flat / not climbing** (incl. slow climb counts as climbing — **do not stop**).  
- Also stop if loss clearly not falling.  
→ may label `NO_CONTEXT_AT_0p5B_CE` and save ckpt for analysis.  
**Do not stop** solely because lift is negative while curve still rises.  
Don’t abort on ALL alone.

### S3 — matched 0L full (~0.5–1× primary time)
Same data/steps ritual.

### S4 — probes + verdict
Battery, order, same-last; write `stage169_context_0p5b_decision.json` + mini report.

### S5 — optional stretch to 1B
If S4 is **MIXED** (some lift, not ≥5pp) or curves still climbing → extend corpus to 1B and continue train (resume), not restart from toy data.

---

## Explicit non-goals (anti-falsification)

- No “prove on 50M then scale” as gate for this question  
- No crowning thin because STORY liked it  
- No declaring understands from ALL% alone  
- No 500k soak on failed small-regime recipe as substitute for data  
- Soft morph / weird digs out of this queue  

---

## Files (to implement next)

| File | Role |
|------|------|
| `results/plan_sote_context_0p5b_1b.md` | this plan |
| `_stage169_context_scale_pipeline.py` | S0–S5 runner `--step N` |
| `data/sote_ids_*.bin` | memmap ids |
| `results/stage169_s*_*.json` / `*_mini.md` | checkpoints |

---

## One-line contract

**0.5B–1B codebook ids + capacity model + context probes vs 0L/majority — or honest fail. No small-data stand-in.**
