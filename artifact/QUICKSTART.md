# TapeLM — 5-minute quickstart

**Why this exists:** [`WHY_TAPELM.md`](WHY_TAPELM.md)

---

## Path 1 — ~2 minutes, no checkpoints (start here)

No GPU. No Hugging Face download. See curated stage verdicts:

```bash
pip install -r artifact/requirements.txt
python artifact/scripts/show_map.py
```

Then skim [`OVERVIEW.md`](OVERVIEW.md) or the stage table in [`../README.md`](../README.md).

---

## Path 2 — full product demo (~5–15 min, GPU + weights)

Downloads **P1** (+ optional **w_registry**) from Hugging Face — hundreds of MB, one-time.

```bash
pip install -r artifact/requirements.txt
python artifact/scripts/download_checkpoints.py --with-w-registry
python artifact/scripts/run_product.py
```

This runs the **product memory track**:

1. **Canonical slots** — write & 4-way recall (stage **227** idea)  
2. **Contradictions** — resolution policy ( **230** ; multi-hit from **229**)  
3. **Cross-domain** — **W** + **fp decode** ( **221 / 228c / 226c** )

Optional:

```bash
python artifact/scripts/export_w_registry.py --smoke
python artifact/scripts/run_product.py --all    # + Stage 196 scorecard (longer)
```

**Frozen P1:** steps 1–2 do not train the encoder on new facts. Details: [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md#frozen-p1--precise-contract)

---

## Product track (stage numbers explained)

| Order | Stages | What you get |
|-------|--------|----------------|
| **221 → 227 → 228c → 230 → 226c** | W · canonical bank · fp decode · resolve · cross-domain e2e | What `run_product.py` demonstrates |

API: [`../docs/MEMORY_ENGINEERING.md`](../docs/MEMORY_ENGINEERING.md) · paper §4.8: [`../results/preprint_tapelm_draft.md`](../results/preprint_tapelm_draft.md)

---

## Publish / cite

[`../docs/GITHUB.md`](../docs/GITHUB.md) · [`../docs/HUGGINGFACE.md`](../docs/HUGGINGFACE.md) · [`../CITATION.cff`](../CITATION.cff)
