# TapeLM

> **Shipping trunk:** **221 → 227 → 228c → 230 → 226c** — `python artifact/scripts/run_product.py`

**Knowledge in weights** is hard to edit. **RAG** retrieves text and hopes the LM listens. **TapeLM** keeps facts and relations as **operable vectors** in the **same fingerprint space** as generation — bind, hop, write, migrate across domains, resolve conflicts — on **one** frozen character-curve encoder, with staged proofs and fair baselines.

**Start here if you are human:** [`artifact/WHY_TAPELM.md`](artifact/WHY_TAPELM.md) · **5 min run:** [`artifact/QUICKSTART.md`](artifact/QUICKSTART.md)

---

## What is strongest (conceptually)

| Idea | What we show | Where |
|------|----------------|-------|
| **Not another embedder** | One `arc_enc` for generate + key + calibrate + compose; vector-native APIs vs text chunks | Preprint §5.1 · [`OVERVIEW`](artifact/OVERVIEW.md) |
| **Structured memory** | Subject slots, binding, external hops, edit, **230** resolution | 192–205, 203 |
| **Frozen P1, living slots** | Facts without backbone finetune; delete without collateral | §3.1 · 205 |
| **Capability vs fair RAG** | Noisy recall **0.913 vs 0.627**; unlearn without wrecking the model | 204–205 |
| **Product trunk** | Retrieve ≠ utilize — **228c** ~**1.0** fp vs ~**0.48** head; cross-domain **226c** ~**0.88** | **221→227→228c→230→226c** |

We report **parity** on clean static recall against a **fair GPT+RAG** index — on purpose. The headline is **architecture + structure + trunk + noise/unlearn**, not SOTA on tidy retrieval.

Every claim: JSON in [`artifact/decisions/`](artifact/decisions/) · `python artifact/scripts/show_map.py`

---

## Architecture in one glance

```text
characters → arc_enc (P1, frozen for memory) → fp(w)
                ├─ CE head → text
                └─ 221 W → 227 slots → 228c decode → 230 resolve → (226c cross-domain)
```

| Typical stack | TapeLM |
|---------------|--------|
| RAG: chunks + second embedder | **One geometry**; structured fp store |
| Domain shift → reindex everything | **Canonical bank + W** @ read (227) |
| LM ignores retrieved text | **228c** fp decode on candidates |
| Conflicting docs in the prompt | **230** policy on multi-hit slots |

Diagram: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Evidence pillars (same product)

**Pillar A — core fp (191–205):** GPT parity on generation (191); calibration, recall, hop/bind, edit, stream; **204–205** wins vs fair RAG under noise and unlearn.

**Pillar B — memory trunk:** **221 → 227 → 228c → 230 → 226c** (see table above). Exploration **213–220** documents dead ends (e.g. **215 NO** → **221 W**).

Deep dive: [`results/plan_curve_dynamics.md`](results/plan_curve_dynamics.md) · [`results/extension_memory_contract.md`](results/extension_memory_contract.md) · preprint [`results/preprint_tapelm_draft.md`](results/preprint_tapelm_draft.md)

**Out of product headline:** variant-B fp generation (207), fp rerank (208), PAWS @ 3050 (209), internalization (210–212) — verdicts kept for a complete map.

---

## Quick orientation

| What | Where |
|------|--------|
| **Why read this** | [`artifact/WHY_TAPELM.md`](artifact/WHY_TAPELM.md) |
| Try it | [`artifact/QUICKSTART.md`](artifact/QUICKSTART.md) |
| One-page product | [`artifact/OVERVIEW.md`](artifact/OVERVIEW.md) |
| Frozen P1 contract | [`docs/ARCHITECTURE.md#frozen-p1--precise-contract`](docs/ARCHITECTURE.md#frozen-p1--precise-contract) |
| Memory API | [`docs/MEMORY_ENGINEERING.md`](docs/MEMORY_ENGINEERING.md) |
| Stages | [`docs/STAGES.md`](docs/STAGES.md) |

---

## Reproduce

Python 3.10+, PyTorch, `tokenizers`, `transformers`; GPU recommended.

```bash
pip install -r artifact/requirements.txt
python artifact/scripts/download_checkpoints.py --with-w-registry
python artifact/scripts/run_product.py
python artifact/scripts/run_product.py --all   # + scorecard (196)
```

---

## Citation & license

[`CITATION.cff`](CITATION.cff) · MIT [`LICENSE`](LICENSE)
