# Publishing on GitHub (copy-paste)

Use this when creating the remote repo or editing **About** settings.

---

## Repository name

**`TapeLM`** — as on GitHub: [KonstantinK-V/TapeLM](https://github.com/KonstantinK-V/TapeLM).

---

## Short description (GitHub “About”, ≤ 350 characters)

```text
TapeLM: frozen dual-channel curve encoder + zero-train fingerprint memory (calibration, slots, edit, hops) in one fp-space. Honest GPT/RAG parity + noise/unlearn wins. Full negative-result map (Stages 170–212).
```

*(Character count ~195 — room for your handle or “MIT”.)*

---

## Extended “About” / social preview (optional README excerpt)

```text
Alternative LM contract: character-curve substrate, episodic fp slots, vector binding — not “beat RAG on everything.” Reproducible stage decisions, falsified generative fingerprints, closed internalization frontier.
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

**Meta:** falsification, negative results, reproducible benchmarks, staged research program.

---

## Before first push

1. `repository-code` in [`CITATION.cff`](../CITATION.cff) → `https://github.com/KonstantinK-V/TapeLM`.
2. Confirm `.gitignore` excludes `checkpoints/`, large logs, secrets.
3. Point visitors to [`artifact/README.md`](../artifact/README.md) in root README (already linked).

---

## Optional badges (add to README if desired)

- License: `![MIT](https://img.shields.io/badge/license-MIT-blue.svg)`
- Python: `![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)`

No PyPI package — badges are informational only.
