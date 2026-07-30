# TapeLM

**One character-curve encoder. One fingerprint space for generation, memory, calibration, and edit.**

TapeLM (variant A) is a **non-standard LM stack** that works as **one system**: dual-channel **character-curve** encoder **pretrained once (Stage 191)**, then **frozen** for product use, with **operable fingerprint memory** on the same `arc_enc` geometry — unified fp-space instead of weights-plus-chunk-RAG.

**What you get**

- **Generation** at matched-GPT parity on the P1 protocol (191).
- **Zero-train fp layer** — calibration, episodic recall, multi-hop binding, one-shot edit, streaming under budget (192–198, 203–205).
- **Canonical memory** — one slot bank, **family W** for domain shift, **fp decode** when the CE head underuses memory (227, 228c, 226c).
- **Contradiction handling** — slots surface candidates; **resolution policy** chooses among them (229–230).

Every claim is tied to **staged, reproducible exams** (JSON verdicts in [`artifact/decisions/`](artifact/decisions/)). Comparisons use **matched GPT** and **fair GPT+RAG** where the stage defines them.

**Quickstart:** [`artifact/QUICKSTART.md`](artifact/QUICKSTART.md) · `python artifact/scripts/run_product.py`  
**Docs:** [`artifact/OVERVIEW.md`](artifact/OVERVIEW.md) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (diagram) · [`docs/MEMORY_ENGINEERING.md`](docs/MEMORY_ENGINEERING.md) · [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md)

---

## Why this architecture is different

```text
characters → arc_enc (P1) → fp(w) = normalize(encoder(word))
                │
                ├─ CE head → text
                └─ canonical slots + W@read + fp decode + resolve
```

| Typical stack | TapeLM |
|---------------|--------|
| BPE tokens → transformer | **Character ink → curve encoder → BPE targets** |
| Facts in weights or retrieved **strings** | Facts as **fp keys/values** in encoder space |
| Second embedder / reranker for memory | **Same** `arc_enc` for gen and memory |
| Domain shift → rebuild the index | **Canonical bank + W_family** at read |
| Retrieved text ignored by the LM | **228c fp decode** on explicit candidates |

---

## Quick orientation

| What | Where |
|------|--------|
| Try it (5 min) | [`artifact/QUICKSTART.md`](artifact/QUICKSTART.md) · `run_product.py` |
| Product story | [`artifact/OVERVIEW.md`](artifact/OVERVIEW.md) |
| Verdict JSON | [`artifact/decisions/`](artifact/decisions/) |
| Full program narrative | [`results/plan_curve_dynamics.md`](results/plan_curve_dynamics.md) |
| Memory contract | [`results/extension_memory_contract.md`](results/extension_memory_contract.md) |
| Stage index | [`docs/STAGES.md`](docs/STAGES.md) |
| Preprint draft | [`results/preprint_tapelm_draft.md`](results/preprint_tapelm_draft.md) (§4.8: 221–230) |

---

## Measured results (variant A)

**Core fp stack:** lexical calibration; fact recall; hop2/binding; subject-anchored edit; stream policies; structured external hops; strong **noise/OOV** and **slot unlearn** vs fair baselines on documented slices (204–205).

**Integrated memory (same product):**

| Capability | Stage |
|------------|-------|
| W-remap after encoder shift | 221 |
| Canonical storage + qmap read | 227 |
| Cross-domain recall + fp decode | 228c, 226c |
| Multi-hit slots + resolution policy | 229, 230 |

**Scope (what v1 does not claim):** variant B next-fingerprint generation (207); fp rerank on BPE head (208); PAWS semantic line at 3050 scale (209); latent hops / internal slow tape / instance channel inside forward (210–212). Verdicts remain in the repo for readers who want the full map; they are not the product headline.

Ongoing research (compositional W, temporal W, tool binding) is tracked in [`results/extension_closed_branches.md`](results/extension_closed_branches.md) until promoted into the contract.

---

## Reproduce

Python 3.10+, PyTorch, `tokenizers`, `transformers`; GPU recommended.

```bash
pip install -r artifact/requirements.txt
python artifact/scripts/download_checkpoints.py
python artifact/scripts/run_product.py
python artifact/scripts/run_product.py --all   # + full scorecard (196)
```

Stage scripts and pipeline: [`docs/EXTENSION_PIPELINE.md`](docs/EXTENSION_PIPELINE.md).

---

## Citation & license

[`CITATION.cff`](CITATION.cff) · MIT [`LICENSE`](LICENSE).
