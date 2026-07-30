# TapeLM — start here

**TapeLM variant A** — one **character-curve encoder** and **fingerprint memory** in the same space.

**Fast path:** [`QUICKSTART.md`](QUICKSTART.md)

```bash
pip install -r artifact/requirements.txt
python artifact/scripts/download_checkpoints.py
python artifact/scripts/run_product.py
```

---

## Path A — read (no GPU)

1. [`OVERVIEW.md`](OVERVIEW.md)
2. [`../docs/MEMORY_ENGINEERING.md`](../docs/MEMORY_ENGINEERING.md)
3. [`decisions/`](decisions/) — verdict JSON

```bash
python artifact/scripts/show_map.py
```

---

## Path B — reproduce

| Script | Purpose |
|--------|---------|
| `run_product.py` | **Recommended** — memory demo |
| `run_demo.py` | Full assemble scorecard (196) |
| `export_w_registry.py` | Persist family W weights |
| `download_checkpoints.py` | P1/P2 from Hugging Face |

[`../docs/CHECKPOINTS.md`](../docs/CHECKPOINTS.md) · [`../docs/GITHUB.md`](../docs/GITHUB.md)

```bash
python artifact/scripts/sync_decisions.py
```

---

MIT [`../LICENSE`](../LICENSE) · [`../CITATION.cff`](../CITATION.cff)
