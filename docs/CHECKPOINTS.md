# Checkpoints & data

Git does **not** store weights (see root `.gitignore`). All files below belong to **one TapeLM variant A** install — core LM weights plus optional memory registry.

## Hugging Face (recommended)

After upload, weights live at:

**https://huggingface.co/Kostya03v/TapeLM-P1**

(Replace with your model id if different.)

```bash
pip install huggingface_hub
python artifact/scripts/download_checkpoints.py
python artifact/scripts/download_checkpoints.py --with-w-registry   # after HF upload
```

Step-by-step upload: [`HUGGINGFACE.md`](HUGGINGFACE.md).

---

## TapeLM product encoder (required for 192–212 demos)

| File | Role |
|------|------|
| `checkpoints/stage191_p1_curve.pt` | **P1** canonical curve encoder (d256, 6L) — **pretrained in 191, frozen for product** (no grad on memory ingest) |
| `checkpoints/stage191_p2_gpt.pt` | Matched GPT control |

Optional scaling ablations: `checkpoints/stage209_curve_d*_L*.pt`, `stage209_gpt_*`.

## Family W registry (extensions 221–227)

Not required for Stages 192–212 demos. Built after P1 download:

```bash
python artifact/scripts/export_w_registry.py
python artifact/scripts/upload_w_registry.py       # once, to Hugging Face
python artifact/scripts/download_checkpoints.py --with-w-registry
```

| File | Role |
|------|------|
| `checkpoints/w_registry/w_registry.json` | Manifest |
| `checkpoints/w_registry/W_prose_bwd.pt` | qmap prose-class → canonical |
| `checkpoints/w_registry/W_code_bwd.pt` | qmap code-class → canonical |

See [`MEMORY_ENGINEERING.md`](MEMORY_ENGINEERING.md). Upload: [`HUGGINGFACE.md`](HUGGINGFACE.md).

## BPE tokenizer

Built by Stage 177; cached at `results/stage177_curve_bpe_tokenizer.json` (created on first run if missing).

## Document cache (Stage 191)

`data/` holds wiki-derived caches (e.g. id-doc npz). Regenerate via `_stage191_night.py --phase p0` if absent.

## Legacy SOTE checkpoints

Hundreds of files under `checkpoints/` from pre-170 experiments — **not** needed for TapeLM. Safe to delete locally if disk is tight; keep P1/P2 above.
