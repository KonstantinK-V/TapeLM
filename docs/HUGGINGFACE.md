# Hugging Face — publish TapeLM weights

GitHub holds **code and narratives**; **model weights** (~51 MB for the main set) live on the [Hugging Face Hub](https://huggingface.co/Kostya03v/TapeLM-P1) with CDN download for visitors.

## Recommended files

| Local file | Role |
|------------|------|
| `checkpoints/stage191_p1_curve.pt` | P1 encoder — generation + fp/memory |
| `checkpoints/stage191_p2_gpt.pt` | Matched GPT control |
| `checkpoints/stage177_curve_bpe.pt` | Optional (~4 MB) |
| `checkpoints/w_registry/` | Optional family W (after `export_w_registry.py`) |

Do **not** upload hundreds of legacy checkpoints under `checkpoints/` (multi-GB).

Tokenizer: `results/stage177_curve_bpe_tokenizer.json` stays in GitHub.

---

## Upload steps

1. Create a **Model** at [huggingface.co/new](https://huggingface.co/new) — e.g. `TapeLM-P1`, license **MIT**.
2. Model card: paste [`huggingface/TapeLM-P1/README.md`](../huggingface/TapeLM-P1/README.md).
3. Upload `stage191_p1_curve.pt` and `stage191_p2_gpt.pt` (Files and versions or CLI).

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload Kostya03v/TapeLM-P1 checkpoints/stage191_p1_curve.pt stage191_p1_curve.pt
huggingface-cli upload Kostya03v/TapeLM-P1 checkpoints/stage191_p2_gpt.pt stage191_p2_gpt.pt
```

4. Ensure [`CHECKPOINTS.md`](CHECKPOINTS.md) links the Hub URL. Visitors run `python artifact/scripts/download_checkpoints.py`.

---

## Datasets & Spaces

Exam caches in `data/` can be rebuilt via `_stage191_night.py`; a HF **dataset** is optional. Stage 196 expects **CUDA** — a Model repo + clone instructions is the default path, not a GPU Space.
