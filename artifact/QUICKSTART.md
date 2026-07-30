# TapeLM — 5-minute quickstart

> **Shipping trunk:** **221 → 227 → 228c → 230 → 226c**

**Why this exists:** [`WHY_TAPELM.md`](WHY_TAPELM.md) — structured knowledge on one encoder, not RAG paragraphs.

You will run the **memory trunk**: write canonical slots (**227**), resolve a contradiction (**230**), then cross-domain read with **W + fp decode** (**221/228c/226c**). That is the strongest product line in the repo.

## No GPU

```bash
pip install -r artifact/requirements.txt
python artifact/scripts/show_map.py
```

Read [`OVERVIEW.md`](OVERVIEW.md) or [`WHY_TAPELM.md`](WHY_TAPELM.md).

## With GPU (recommended)

```bash
pip install -r artifact/requirements.txt
python artifact/scripts/download_checkpoints.py --with-w-registry
python artifact/scripts/run_product.py
```

Optional:

```bash
python artifact/scripts/export_w_registry.py --smoke
python artifact/scripts/run_product.py --all         # + scorecard (196)
```

## What the demo steps mean

| Demo step | Trunk stage | Point |
|-----------|-------------|--------|
| Write / 4-way recall | **227** | One canonical bank; keys stay in P1 geometry |
| Contradiction picks | **230** | Structure + policy, not “first stored wins” |
| Code query → answer | **221 + 228c + 226c** | Domain **W**; **fp decode** uses memory (~0.88) when the head does not (~0.45) |

**Frozen P1:** steps 1–2 do not train the encoder. Step 3 may simulate query-side drift; product path uses exported **W** — [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md#frozen-p1--precise-contract).

API: [`../docs/MEMORY_ENGINEERING.md`](../docs/MEMORY_ENGINEERING.md) · paper: [`../results/preprint_tapelm_draft.md`](../results/preprint_tapelm_draft.md) §4.8

## Publish / cite

[`../docs/GITHUB.md`](../docs/GITHUB.md) · [`../docs/HUGGINGFACE.md`](../docs/HUGGINGFACE.md) · [`../CITATION.cff`](../CITATION.cff)
