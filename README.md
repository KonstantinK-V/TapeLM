# TapeLM

**One frozen curve encoder. Memory, calibration, and edits as operable fingerprints — in the same geometry as generation.**

TapeLM is a research artifact for people who care about **how** knowledge is represented, not only benchmark leaderboard position. The backbone is a **dual-channel character-curve encoder** (fast ink + surprise-gated slow writer). On top of it sits a **zero-train fingerprint stack**: lexicon calibration, episodic slots, one-shot subject writes, and vector hops — without training a separate retriever or re-embedding pipeline.

This is **not** marketed as “we beat RAG on everything.” The honest pitch is a **different contract**: unified fp-space on one encoder, **matched-GPT parity** where we measured, **capability wins** where the substrate matters (noisy/OOV text, slot unlearn), and a **published map of negatives** (generative fingerprints, hybrid heads, internalization 210–212).

**Browsing without running code?** Start at [`artifact/README.md`](artifact/README.md) → [`artifact/OVERVIEW.md`](artifact/OVERVIEW.md), then `python artifact/scripts/show_map.py`.

**First time on GitHub?** Read [`docs/GITHUB_FIRST_STEPS.md`](docs/GITHUB_FIRST_STEPS.md) (RU): what uploads, checkpoints, URL in `CITATION.cff`.

---

## Why the architecture is non-standard

| Usual stack | TapeLM (variant A) |
|-------------|-------------------|
| BPE tokens → transformer → logits | **Characters → curve states → BPE targets** |
| Facts in weights *or* retrieved **text chunks** | Facts as **fp slot keys/values** in encoder space |
| RAG: embed query, fetch strings, stuff context | **Cosine / bind / chain in fp-space** on frozen `arc_enc` |
| Calibration = softmax temperature tuning only | **Lexicon surprise** from entity fingerprints (193) |
| Edit = fine-tune or adapter | **Subject-anchored slot overwrite** (197), O(1) unlearn (205) |

Generation stays pretrained once (Stage 191). Everything else is composition on the **same** normalized word fingerprints — the “tape” is explicit slots and policies, not hidden weights.

---

## Quick orientation

| What | Where |
|------|--------|
| **Visitor tour (5 min)** | [`artifact/README.md`](artifact/README.md), [`artifact/OVERVIEW.md`](artifact/OVERVIEW.md) |
| Curated verdict JSON | [`artifact/decisions/`](artifact/decisions/) |
| Full program & scorecard | [`results/plan_curve_dynamics.md`](results/plan_curve_dynamics.md) |
| Closed frontier 210–212 | [`results/pre_publish_frontier.md`](results/pre_publish_frontier.md) |
| Draft preprint prose | [`results/preprint_tapelm_draft.md`](results/preprint_tapelm_draft.md) |
| Architecture (implementers) | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Stage index 170–212 | [`docs/STAGES.md`](docs/STAGES.md) |
| Checkpoints & data | [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md) |
| GitHub blurb & topics | [`docs/PUBLISHING.md`](docs/PUBLISHING.md) |
| Legacy SOTE / hop batteries | [`legacy/sote/`](legacy/sote/) (historical) |

---

## Confirmed on variant A (short)

- **Generation:** parity with matched GPT on P1 scale (191).
- **FP stack:** lexicon calibration, episodic recall, hop2/binding, edit, stream+budget (192–198).
- **vs vanilla GPT:** calibration, edit, streaming; **vs fair GPT+RAG:** mostly architectural unification — **capability** wins on **noise/OOV** (204) and **unlearn** (205).
- **Variant B** (predict next fingerprint): **falsified** (207, 207-MAX).
- **Semantic B @ PAWS on 3050:** not confirmed; not a structural block vs GPT (209).
- **Internalization** (hops inside forward, slow tape, instance channel): **THESIS_NO** (210–212).

Details and numbers: preprint draft §4–5 and Stage 196 scorecard.

---

## Reproduce

Python 3.10+, PyTorch, `tokenizers`, `transformers`; GPU recommended. Checkpoints are **not** in git — see [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md).

```bash
pip install -r artifact/requirements.txt
python artifact/scripts/check_env.py
python artifact/scripts/run_demo.py          # Stage 196 scorecard
python artifact/scripts/run_stage.py 204     # noise vs fair RAG
```

Full P1 training: `_stage191_night.py` (long run; see `results/plan_stage191_night9h.md`).

---

## Find this repo (keywords)

If you search for **character-level LM**, **curve / arc encoder**, **dual-channel memory**, **word fingerprint**, **episodic slot memory**, **zero-train retrieval**, **RAG comparison**, **knowledge editing**, **machine unlearning**, **multi-hop binding**, **OOD lexical calibration**, or **reproducible negative results** — this tree is the staged evidence trail.

Suggested GitHub **Topics** (copy from [`docs/PUBLISHING.md`](docs/PUBLISHING.md)):  
`language-model` · `character-level` · `episodic-memory` · `retrieval-augmented-generation` · `knowledge-editing` · `machine-unlearning` · `reproducible-research`

---

## Citation & license

[`CITATION.cff`](CITATION.cff) — [github.com/KonstantinK-V/TapeLM](https://github.com/KonstantinK-V/TapeLM).

Code: [MIT](LICENSE). Prose in `results/` and `docs/`: cite the repo; CC BY 4.0 if you republish verbatim.
