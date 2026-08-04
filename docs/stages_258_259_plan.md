# Stages 258–259

## Stage 258 — semantic query (implemented)

**Script:** `_stage258_semantic_query.py`  
**Claim:** `results/stage258_claim_scope.json`

Retrieval-only: `q = normalize((1-a)*W_q(fp) + a*W_sem(h))`. No SlotBias, no decode. Keys canonical frozen fp; **SHA256 snapshot** on subject slots after train.

**Exam:** one subject × four relations (same anchor fp); paraphrase queries with **empty content-word overlap** vs write (audited at runtime).

**Verdicts:** `SEM_QUERY_OK` / `PARTIAL` / `NO` (GPT wins) / `NO_AT_SCALE` (GPT also fails) / `INVALID` (fp-only > chance or keys moved).

**Run order:** after 255 ablation → 257 full → **`python _stage258_semantic_query.py --smoke`** (fp@0.25 gate first).

---

## Stage 259 — sign / e2e (planned)

## Stage 259 — hot swap (implemented)

**Script:** `_stage259_hot_swap.py`  
**Glue:** `checkpoints/stage256_slot_bias.pt` (data rebuilt with `SEED_256=256`).  
**Requires:** `TapeView.with_value` in `_inprint_glue.py`.

Measured gates: old answer dies, neighbours OK, keys unchanged (`torch.equal`), glue param hash unchanged, second edit wins, empty tape leak floor.

**Run order:** **`python _stage259_hot_swap.py --smoke`** first (minutes, no ablation wait) → smokes 257/258; full 257 after wiki12 ablation via queue.

Heavier channel or 255→glue wiring — spec after stream ingest if needed.

---

## 257 frame (unchanged)

See `stage257_claim_scope.json` — mechanism @ toy scale, not reasoning.
