# Publishing on GitHub (copy-paste)

Use this when creating the remote repo or editing **About** settings.

---

## Repository name

**`TapeLM`** — as on GitHub: [KonstantinK-V/TapeLM](https://github.com/KonstantinK-V/TapeLM).

---

## Short description (GitHub “About”, ≤ 350 characters)

```text
TapeLM: facts as fingerprints on character ink — not token-id memory, not chunk RAG. One frozen curve encoder for generation and structured slot memory (write, bind, hop, resolve). Noisy recall, lexicon calibration, one-shot edits, clean unlearning — vs fair GPT/RAG; reproducible JSON benchmarks.
```

*~248 characters — room for a personal tweak if you add a URL or author.*

---

## GitHub Topics (up to 20)

Paste as repo topics (hyphenated slugs GitHub accepts):

| Topic | Why |
|-------|-----|
| `language-model` | Broad LM audience |
| `character-level` | Ink / curve substrate |
| `episodic-memory` | Slot memory line |
| `retrieval-augmented-generation` | RAG comparison readers |
| `knowledge-editing` | One-shot subject writes without backbone finetune |
| `machine-unlearning` | Slot delete vs gradient unlearn (collateral-free) |
| `continual-learning` | Frozen encoder + living slot bank |
| `representation-learning` | Unified fp-space (generation + memory) |
| `memory-augmented-neural-networks` | External fp memory on one encoder |
| `information-retrieval` | Recall, hops, cross-domain read |
| `reproducible-research` | JSON stage decisions |
| `pytorch` | Stack |
| `research-code` | Expectation setting |

Additional tags if you have topic slots left:

- `nlp`
- `natural-language-processing`

---

## Search keywords (README, papers, HF `Keywords:` — not all are GitHub topics)

**Pitch:** facts as fingerprints on character ink; not token-id memory; not chunk RAG; one curve encoder; ink→arcBPE readout.

**Strong sides (search phrases):** noisy / typo-robust recall, spelling noise, lexical OOD calibration, lexicon surprise, one-shot knowledge edit, machine unlearning without collateral, fp decode when the LM head skips memory, cross-domain memory read, conflict resolution in slots, vector binding, multi-hop in fp-space, zero-train slot memory, frozen backbone.

**Architecture:** character-level language model, curve encoder, arc dynamics, dual-channel memory, slow fast writer, surprise gating, self-model calibration, word fingerprint, episodic slot memory, canonical memory bank, domain remap at read.

**Comparisons:** fair GPT+RAG baseline, matched GPT control, RAG alternative, knowledge in weights vs index.

**Meta:** reproducible benchmarks, staged research program, axis-specific evaluation, JSON verdicts.

---

## Before first push

1. `repository-code` in [`CITATION.cff`](../CITATION.cff) → `https://github.com/KonstantinK-V/TapeLM`.
2. Confirm `.gitignore` excludes `checkpoints/`, large logs, secrets.
3. Point visitors to [`artifact/README.md`](../artifact/README.md) in root README (already linked).

---

## Reader paths (consistent story)

| Audience | Start here | Then |
|----------|------------|------|
| **Why should I care?** | [`artifact/WHY_TAPELM.md`](../artifact/WHY_TAPELM.md) | [`artifact/OVERVIEW.md`](../artifact/OVERVIEW.md) |
| **5 min run** | [`artifact/QUICKSTART.md`](../artifact/QUICKSTART.md) | [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) |
| **Product** | [`artifact/OVERVIEW.md`](../artifact/OVERVIEW.md) | [`docs/MEMORY_ENGINEERING.md`](../docs/MEMORY_ENGINEERING.md) |
| **Paper** | [`results/preprint_tapelm_draft.md`](../results/preprint_tapelm_draft.md) (§3.1 frozen, §4.8 memory) | [`results/plan_curve_dynamics.md`](../results/plan_curve_dynamics.md) |
| **Stages / JSON** | [`docs/STAGES.md`](../docs/STAGES.md) | `artifact/decisions/` |

All paths should agree: **shipping trunk 221 → 227 → 228c → 230 → 226c**; **two pillars** — **191–205** + trunk; **P1** frozen on memory ingest; facts in **slots**; drift via **W @ read**.

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
