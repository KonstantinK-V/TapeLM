# TapeLM — 5-minute quickstart

**One product:** character-curve LM + fingerprint memory on the same encoder.

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

| Step | Mechanism |
|------|-----------|
| Write / recall | Canonical fp slot keys |
| Conflicts | Multi-hit slots + `resolve_slot_contradiction` |
| Cross-domain | `W_code` qmap + 4-way retrieve + fp decode |

API reference: [`../docs/MEMORY_ENGINEERING.md`](../docs/MEMORY_ENGINEERING.md)

## Publish / cite

[`../docs/GITHUB.md`](../docs/GITHUB.md) · [`../docs/HUGGINGFACE.md`](../docs/HUGGINGFACE.md) · [`../CITATION.cff`](../CITATION.cff)
