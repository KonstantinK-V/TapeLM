# Publishing on GitHub (copy-paste)

Use this when creating the remote repo or editing **About** settings.

---

## Repository name

**`TapeLM`** — as on GitHub: [KonstantinK-V/TapeLM](https://github.com/KonstantinK-V/TapeLM).

---

## Short description (GitHub “About”, ≤ 350 characters)

```text
TapeLM: character-curve LM + unified fp memory — structured knowledge as operable vectors (not RAG+embedder swap). One encoder; staged evidence vs matched GPT / fair RAG.
```

*(~160 characters.)*

---

## Extended “About” / social preview (optional README excerpt)

```text
Non-standard stack: P1 curve encoder and operable fingerprint memory in one space. Generation parity, recall, edit, cross-domain memory, contradiction resolution — reproducible stage program.
```

---

## GitHub Topics (up to 20)

Paste as repo topics (hyphenated slugs GitHub accepts):

| Topic | Why |
|-------|-----|
| `language-model` | Broad LM audience |
| `character-level` | Ink / curve substrate |
| `episodic-memory` | Slot memory line |
| `retrieval-augmented-generation` | RAG comparison readers |
| `knowledge-editing` | Subject-anchored writes |
| `machine-unlearning` | Slot delete vs gradient unlearn |
| `representation-learning` | Unified fp-space |
| `reproducible-research` | JSON stage decisions |
| `pytorch` | Stack |
| `research-code` | Expectation setting |

Additional tags people might search (add if you have topic slots left):

- `memory-augmented-neural-networks`
- `information-retrieval`
- `nlp`

---

## Search keywords (README, papers, CITATION — not all are GitHub topics)

**Architecture:** character-level language model, curve encoder, arc dynamics, dual-channel memory, slow fast writer, surprise gating, self-model calibration.

**Memory / retrieval:** word fingerprint, fingerprint lexicon, episodic slot memory, vector binding, multi-hop retrieval, zero-train memory, external memory loop, compositional retrieval.

**Comparisons:** RAG baseline, matched GPT, knowledge in weights vs index, architectural unification.

**Capabilities:** OOD lexical calibration, noisy OCR text, out-of-vocabulary, one-shot knowledge edit, streaming under memory budget, machine unlearning collateral.

**Meta:** reproducible benchmarks, staged research program, axis-specific evaluation.

---

## Before first push

1. `repository-code` in [`CITATION.cff`](../CITATION.cff) → `https://github.com/KonstantinK-V/TapeLM`.
2. Confirm `.gitignore` excludes `checkpoints/`, large logs, secrets.
3. Point visitors to [`artifact/README.md`](../artifact/README.md) in root README (already linked).

---

## Reader paths (consistent story)

| Audience | Start here | Then |
|----------|------------|------|
| **5 min run** | [`artifact/QUICKSTART.md`](../artifact/QUICKSTART.md) | [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) |
| **Product** | [`artifact/OVERVIEW.md`](../artifact/OVERVIEW.md) | [`docs/MEMORY_ENGINEERING.md`](../docs/MEMORY_ENGINEERING.md) |
| **Paper** | [`results/preprint_tapelm_draft.md`](../results/preprint_tapelm_draft.md) (§3.1 frozen, §4.8 memory) | [`results/plan_curve_dynamics.md`](../results/plan_curve_dynamics.md) |
| **Stages / JSON** | [`docs/STAGES.md`](../docs/STAGES.md) | `artifact/decisions/` |

All paths should agree: **two pillars** — **191–205** (core fp + 204/205) and **221–230** (memory trunk; `run_product.py`); **P1** pretrained once (191), frozen on memory ingest; facts in **slots**; drift via **W @ read**.

---

## Optional badges (add to README if desired)

- License: `![MIT](https://img.shields.io/badge/license-MIT-blue.svg)`
- Python: `![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)`

No PyPI package — badges are informational only.

---

## Hugging Face model card (`Kostya03v/TapeLM-P1`)

Edit **two places** on the Hub (copy from [`huggingface/TapeLM-P1/README.md`](../huggingface/TapeLM-P1/README.md)):

1. **YAML `tags`** — fixed slugs for discovery (see [`HUGGINGFACE.md`](HUGGINGFACE.md)).
2. **`Keywords:` line** — first line of the README body; comma-separated phrases for humans and search.

You do **not** need new tags for every stage run — only when the **public product claim** changes (e.g. after promoting a research branch).
