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
2. Model card: paste [`huggingface/TapeLM-P1/README.md`](../huggingface/TapeLM-P1/README.md) (YAML `tags` + **Keywords** line at top — update on Hub when you change the product story).
3. Upload `stage191_p1_curve.pt` and `stage191_p2_gpt.pt` (Files and versions or CLI).

### Family `w_registry/` (optional)

```bash
python artifact/scripts/export_w_registry.py          # creates checkpoints/w_registry/
huggingface-cli login
python artifact/scripts/upload_w_registry.py          # -> HF .../w_registry/*
```

Visitors: `python artifact/scripts/download_checkpoints.py --with-w-registry`

Hub paths: `w_registry/w_registry.json`, `w_registry/W_prose_bwd.pt`, … (folder at model repo root).

### HF tags vs Keywords line

| Where | Purpose | Change when |
|-------|---------|-------------|
| YAML `tags:` (5–10 slugs) | Hub filters & related models | You add a **new axis** (e.g. tool-use memory) |
| `pipeline_tag: text-generation` | Default task badge | Only if you ship a non-LM artifact |
| **Keywords** paragraph (README) | Search / skim; free text | Any release; keep in sync with GitHub [`PUBLISHING.md`](PUBLISHING.md) search list |

**Current YAML tags (recommended):** `language-model`, `nlp`, `character-level`, `episodic-memory`, `retrieval-augmented-generation`, `knowledge-editing`, `representation-learning`, `pytorch`.

**Optional add later** (if Hub accepts): `machine-unlearning`, `memory-augmented-neural-networks` — not required for v1.

**GitHub Topics** and **HF tags** overlap but are edited separately; keep the same *ideas*, not necessarily identical strings.

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
