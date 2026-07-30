# TapeLM — start here

Character **ink** → **fp** facts on one encoder (not token-id store, not chunk RAG). Text: **ink→arcBPE**. Story: [`WHY_TAPELM.md`](WHY_TAPELM.md) · trunk **221→227→228c→230→226c** · `run_product.py`

**TapeLM variant A** — one **character-curve encoder** (pretrained in 191, **frozen** for memory ingest) and **fingerprint memory** in the same space.

**Fast path:** [`QUICKSTART.md`](QUICKSTART.md)

```bash
pip install -r artifact/requirements.txt
python artifact/scripts/download_checkpoints.py --with-w-registry
python artifact/scripts/run_product.py
```

**Paper-shaped read:** [`../results/preprint_tapelm_draft.md`](../results/preprint_tapelm_draft.md) (§4.8 **221–230**) · **Diagram:** [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)

**Headline results:** **204–205** (noise/unlearn) + **221–230** (canonical, W, fp decode, resolve — `run_product.py`).

---

## Path A — read (no GPU)

1. [`OVERVIEW.md`](OVERVIEW.md)
2. [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — inference diagram + frozen-P1 table
3. [`../docs/MEMORY_ENGINEERING.md`](../docs/MEMORY_ENGINEERING.md)
4. [`../results/preprint_tapelm_draft.md`](../results/preprint_tapelm_draft.md) (§4.8: 221–230)
5. [`decisions/`](decisions/) — verdict JSON

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
