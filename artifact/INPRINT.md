# Inprint v0.1

**Product name:** Inprint  
**Code lineage:** TapeLM variant A (same repo & checkpoints)  
**Tagline:** facts as **fingerprints on character ink** — not token-id memory, not chunk RAG.

Machine-readable bill of materials: [`inprint/manifest.json`](inprint/manifest.json).

---

## What v0.1 is

One **frozen P1** curve encoder plus **three product layers** we actually demo and document:

| Layer | Stages | What you get |
|-------|--------|----------------|
| **Fp core** | 192–205 | Calibration, slots, hop, edit; **204/205** vs fair GPT+RAG |
| **Memory trunk** | **221→227→228c→230→226c** | Canonical bank, **W**@read, fp decode, resolve, cross-domain use |
| **Continual + stream** | 251–255 | Shared upper learns language; **facts stay on tape**; chunked ingest + **W_q** from real entities |
| **Glue (preview)** | 256 | Copy-mixture decode so free-form head can **use** retrieved slots without a 4-way menu |

**Not in the default demo:** internalization (210–212), variant B (207), post-trunk ops-only forks (231–250) — measured, listed in `show_map`, not the v0.1 walkthrough.

---

## Run v0.1 (5–15 min, GPU)

```bash
pip install -r artifact/requirements.txt
python artifact/scripts/download_checkpoints.py --with-w-registry
python artifact/scripts/run_inprint.py demo
```

No GPU: `python artifact/scripts/run_inprint.py map`

---

## Honest scope (Aug 2026)

- **Demo path** proves **memory trunk** end-to-end plus **step 4 glue** when `checkpoints/stage256_slot_bias.pt` (and ideally `stage253_joint_l02.pt`) are present.
- **Stream:** `python artifact/scripts/run_inprint.py ingest --schedule wiki:2,med:2 --run-tag my_run`
- Stream **recall** metrics use symmetric queries + bank-wide **W_q** training; saturated top1 on small banks is a known measurement pitfall — compare frozen vs adapted and empty/shuffled ablations.

---

## Read order

1. [`WHY_TAPELM.md`](WHY_TAPELM.md) — thesis (rename to Inprint in prose over time)
2. [`OVERVIEW.md`](OVERVIEW.md) — one page
3. [`../docs/MEMORY_ENGINEERING.md`](../docs/MEMORY_ENGINEERING.md) — API grain
4. [`../docs/STAGES.md`](../docs/STAGES.md) — full stage map
5. [`decisions/`](decisions/) — JSON verdicts

Legacy entrypoint `run_product.py` calls the same memory demo; prefer **`run_inprint.py`** for v0.1.
