# TapeLM — 5-minute quickstart

**One product, two headline results:** core fp stack (191–205) + **memory trunk (221–230)** — the demo below is mostly **Pillar B** (canonical slots → conflicts → cross-domain **228c/226c**).

## No GPU

```bash
pip install -r artifact/requirements.txt
python artifact/scripts/show_map.py
```

Read [`OVERVIEW.md`](OVERVIEW.md).

## With GPU (recommended)

```bash
pip install -r artifact/requirements.txt
python artifact/scripts/download_checkpoints.py --with-w-registry
python artifact/scripts/run_product.py
```

This runs the **memory demo** (canonical slots → contradiction policy → cross-domain fp decode).

Optional:

```bash
python artifact/scripts/export_w_registry.py --smoke   # persist family W
python artifact/scripts/run_product.py --all         # + full scorecard (196)
```

## What you just saw

| Step | Mechanism | Encoder |
|------|-----------|---------|
| Write / recall | Canonical fp slot keys | **Frozen P1** (`eval`, no weight updates) |
| Conflicts | Multi-hit slots + `resolve_slot_contradiction` | Frozen P1 |
| Cross-domain | `W_code` qmap + 4-way retrieve + fp decode | **Canonical bank unchanged**; query may use a **code-domain copy** of the encoder in the demo to simulate drift — product path uses **exported W**, not slot reindex |

**Frozen P1 (product claim):** checkpoint `stage191_p1_curve.pt` is **not** trained when you add or read facts. Training in the repo is **offline pretrain (191)** or **stage exams** (e.g. fit **W** after a deliberate shift). See [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md#frozen-p1--precise-contract) · paper §3.1 [`../results/preprint_tapelm_draft.md`](../results/preprint_tapelm_draft.md).

**Read (no GPU):** [`OVERVIEW.md`](OVERVIEW.md) · [`../results/preprint_tapelm_draft.md`](../results/preprint_tapelm_draft.md) · `python artifact/scripts/show_map.py`

API reference: [`../docs/MEMORY_ENGINEERING.md`](../docs/MEMORY_ENGINEERING.md)

## Publish / cite

[`../docs/GITHUB.md`](../docs/GITHUB.md) · [`../docs/HUGGINGFACE.md`](../docs/HUGGINGFACE.md) · [`../CITATION.cff`](../CITATION.cff)
