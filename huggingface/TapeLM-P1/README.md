---
license: mit
tags:
  - language-model
  - nlp
  - character-level
  - episodic-memory
  - retrieval-augmented-generation
  - knowledge-editing
  - representation-learning
  - pytorch
library_name: tapelm
pipeline_tag: text-generation
---

**Keywords:** character-level language model, curve encoder, dual-channel memory, word fingerprint, episodic slot memory, zero-train retrieval, canonical memory, knowledge editing, machine unlearning, RAG alternative, PyTorch, reproducible research.

# TapeLM weights (variant A)

Checkpoints for **TapeLM** — one system: **pretrained (191), frozen for memory ingest** dual-channel curve encoder (P1), matched GPT control, and optional **family W** for canonical fingerprint memory (227 qmap, 228c decode).

**Weights:** [huggingface.co/Kostya03v/TapeLM-P1](https://huggingface.co/Kostya03v/TapeLM-P1)  
**Code:** [github.com/KonstantinK-V/TapeLM](https://github.com/KonstantinK-V/TapeLM)  
**Quickstart:** [artifact/QUICKSTART.md](https://github.com/KonstantinK-V/TapeLM/blob/main/artifact/QUICKSTART.md)  
**Preprint draft:** [results/preprint_tapelm_draft.md](https://github.com/KonstantinK-V/TapeLM/blob/main/results/preprint_tapelm_draft.md) · **Architecture:** [docs/ARCHITECTURE.md](https://github.com/KonstantinK-V/TapeLM/blob/main/docs/ARCHITECTURE.md)

## Files

| File | Role |
|------|------|
| `stage191_p1_curve.pt` | P1 encoder — generation + fp; **frozen on memory path** |
| `stage191_p2_gpt.pt` | Matched GPT control |
| `w_registry/*` | Optional family W (qmap read) |

Tokenizer: `results/stage177_curve_bpe_tokenizer.json` on GitHub.

## Usage

```bash
git clone https://github.com/KonstantinK-V/TapeLM.git
cd TapeLM
pip install -r artifact/requirements.txt
python artifact/scripts/download_checkpoints.py
python artifact/scripts/download_checkpoints.py --with-w-registry
python artifact/scripts/run_product.py
```

## License

MIT — [TapeLM GitHub](https://github.com/KonstantinK-V/TapeLM/blob/main/LICENSE).

## Citation

[CITATION.cff](https://github.com/KonstantinK-V/TapeLM/blob/main/CITATION.cff)
