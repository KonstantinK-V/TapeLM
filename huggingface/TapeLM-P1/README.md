---
license: mit
tags:
  - language-model
  - nlp
  - character-level
  - episodic-memory
  - retrieval-augmented-generation
  - knowledge-editing
  - machine-unlearning
  - continual-learning
  - information-retrieval
  - representation-learning
  - pytorch
library_name: tapelm
pipeline_tag: text-generation
---

**Keywords:** character ink, word fingerprint, curve encoder, episodic slot memory, noisy recall, lexical calibration, one-shot knowledge edit, machine unlearning, multi-hop retrieval, cross-domain memory, conflict resolution, zero-train memory, fair GPT+RAG baseline, RAG alternative, PyTorch, reproducible research.

# TapeLM weights (variant A)

> **Facts on character ink (fp), not token-id store or chunk RAG** · one encoder · noisy recall, calibration, edits, unlearning · **ink→arcBPE** · optional `w_registry/`

Checkpoints for **TapeLM** — one system: **Pillar A** (P1 + core fp 191–205) and **Pillar B** (memory trunk, optional `w_registry/`).

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
