# TapeLM

> **Shipping trunk (product memory):** **221 → 227 → 228c → 230 → 226c**  
> W remap · canonical bank + qmap · fp decode · conflict resolution · cross-domain e2e — `python artifact/scripts/run_product.py`

**One character-curve encoder. One fingerprint space — structured knowledge as operable vectors, not RAG text chunks.**

TapeLM (variant A) is **not “GPT + index with a different embedder.”** It is a **non-standard LM stack** where facts and relations live as **fp keys/values, binding, hops, edits, and resolution policies** in the **same** geometry as generation—pretrained once (Stage 191), then **frozen** for product memory ingest.

**What you get** — **two evidence pillars** (same product, same encoder):

1. **Core fp stack (191–205)** — calibration, recall, hop/bind, edit, stream; **noise/unlearn vs fair RAG** (204–205).
2. **Production memory trunk (221–230, 226c)** — after 210–212 closed internalization: **canonical bank + W + fp decode + resolve**; cross-domain **~0.88 utilization via fp decode vs ~0.45 CE head** (226c). Explored path 213–220 (215 adapter **NO** → **221 W**).

- **Generation** at matched-GPT parity on the P1 protocol (191).
- **Zero-train fp layer** — calibration, episodic recall, multi-hop binding, one-shot edit, streaming under budget (192–198, 203–205).
- **Canonical memory trunk** — **`221 → 227 → 228c → 230 → 226c`**; **`run_product.py`** walks this path.

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
| RAG: embed chunks, retrieve **text**, re-prompt LM | **Structured fp store**: subject keys, bind, hop **without** serial decode |
| Facts in weights or retrieved **strings** | Facts as **operable vectors** in encoder space |
| Second embedder / reranker for memory | **Same** `arc_enc` for gen and memory |
| Domain shift → rebuild the index | **Canonical bank + W_family** at read |
| Retrieved text ignored by the LM | **228c fp decode** on explicit candidates |
| Conflicting sources → prompt soup | **230** resolution over multi-hit slots |

---

## Quick orientation

| What | Where |
|------|--------|
| Try it (5 min) | [`artifact/QUICKSTART.md`](artifact/QUICKSTART.md) · `run_product.py` |
| Product story | [`artifact/OVERVIEW.md`](artifact/OVERVIEW.md) |
| Verdict JSON | [`artifact/decisions/`](artifact/decisions/) |
| Full program narrative | [`results/plan_curve_dynamics.md`](results/plan_curve_dynamics.md) |
| Memory trunk (221–230) | [`extension_memory_contract.md`](results/extension_memory_contract.md) · demo `run_product.py` |
| Stage index | [`docs/STAGES.md`](docs/STAGES.md) |
| Preprint draft | [`results/preprint_tapelm_draft.md`](results/preprint_tapelm_draft.md) (§3.1 frozen contract; §4.8: 221–230) |
| Frozen P1 (what trains) | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#frozen-p1--precise-contract) |

---

## Measured results (variant A)

### Pillar A — core fp (191–205)

Lexical calibration; fact recall; hop2/binding; subject-anchored edit; stream; external hops; **204–205 capability wins** vs fair RAG (noisy recall **0.913 vs 0.627**; slot unlearn collateral-free).

### Pillar B — memory trunk (221–230) ★ product demo

| Result | Stage | Headline |
|--------|-------|----------|
| W after encoder shift | 221 | ~**0.78** recall vs ~**0.87** oracle reindex |
| One canonical bank + qmap | 227 | cross-code ~**0.95** |
| Utilization (not just retrieve) | **228c** | **1.0** fp decode vs ~**0.48** head |
| Cross-domain e2e | **226c** | ~**0.88** fp vs ~**0.45** head |
| Conflicting slots | **230** | composite ~**1.0** vs argmax ~**0.47** |

Program **213–220** maps closed branches (e.g. **215 NO**); **shipping order** is **221 → 227 → 228c → 230 → 226c**. Narrative: [`plan_curve_dynamics.md`](results/plan_curve_dynamics.md) (*Memory extension program*) · preprint **§4.8**.

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
