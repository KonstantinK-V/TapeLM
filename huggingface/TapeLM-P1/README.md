---
license: mit
tags:
  - language-model
  - character-level
  - episodic-memory
  - pytorch
library_name: tapelm
---

# TapeLM P1 + matched GPT (Stage 191)

Weights for the **TapeLM variant A** artifact: frozen **dual-channel curve encoder** (P1) and **matched GPT-2** control used in Stages 192–212.

**This model repo:** [huggingface.co/Kostya03v/TapeLM-P1](https://huggingface.co/Kostya03v/TapeLM-P1)  
**Code & verdicts:** [github.com/KonstantinK-V/TapeLM](https://github.com/KonstantinK-V/TapeLM)  **Architecture:** [docs/ARCHITECTURE.md](https://github.com/KonstantinK-V/TapeLM/blob/main/docs/ARCHITECTURE.md)

## Files

| File | Role | ~Size |
|------|------|-------|
| `stage191_p1_curve.pt` | **P1** curve encoder (SelfModelXL d256, 6L, ~150M char pretrain) | 29 MB |
| `stage191_p2_gpt.pt` | Matched GPT control for fair comparisons | 22 MB |
| `stage177_curve_bpe.pt` | Optional BPE-on-curve checkpoint (177 path) | 4 MB |

BPE tokenizer JSON lives in the GitHub repo: `results/stage177_curve_bpe_tokenizer.json`.

## Usage

```bash
git clone https://github.com/KonstantinK-V/TapeLM.git
cd TapeLM
pip install torch tokenizers transformers huggingface_hub
python artifact/scripts/download_checkpoints.py
python artifact/scripts/run_demo.py
```

Or download manually from this repo into `checkpoints/`:

```
checkpoints/stage191_p1_curve.pt
checkpoints/stage191_p2_gpt.pt
```

## License

MIT — same as [TapeLM GitHub](https://github.com/KonstantinK-V/TapeLM/blob/main/LICENSE).

## Citation

See [CITATION.cff](https://github.com/KonstantinK-V/TapeLM/blob/main/CITATION.cff) on GitHub.
