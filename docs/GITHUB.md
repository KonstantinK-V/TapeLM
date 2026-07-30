# GitHub & Hugging Face (quick start)

**Repository:** [github.com/KonstantinK-V/TapeLM](https://github.com/KonstantinK-V/TapeLM) — set the same URL in [`CITATION.cff`](../CITATION.cff) (`repository-code`).

**Weights are not in git.** Download P1/P2 into `checkpoints/`:

```bash
pip install huggingface_hub
python artifact/scripts/download_checkpoints.py
```

See [`CHECKPOINTS.md`](CHECKPOINTS.md) · upload: [`HUGGINGFACE.md`](HUGGINGFACE.md)

```bash
python artifact/scripts/download_checkpoints.py
python artifact/scripts/run_product.py
```
