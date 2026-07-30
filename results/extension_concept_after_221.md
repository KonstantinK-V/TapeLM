# TapeLM roadmap after 221

**Product contract (224+):** [`extension_memory_contract.md`](extension_memory_contract.md) — **Freeze → W-remap (per family) → Stream decay**.

---

## Memory layer — what is established

1. **L1 Freeze:** with **`arc_enc` frozen**, episodic/lexicon slots stay on stable geometry (213).

2. **L2 W-remap:** encoder shift is recoverable; linear W on core vocab restores metrics (**0.997** align, **0.78** recall vs **0.87** oracle; OOV **~0.99**). W is **change-of-basis / restorer**. Under strong code shift (cos **~0.59**) matched W still **~0.88** (224).

3. **Family registry:** prose domains are largely interchangeable; **code** is a separate family (cross-drop **~0.12**). `{prose, code, …}` plus **fork** when reuse drop ≥ 0.05.

4. **Cos trigger:** `mean_cos` on core after shift → Identity / `W_family` / learn (`WFamilyPolicy`).

5. **L3 Stream:** decay / refresh (219); slot `W` version id — open.

6. **227–230, 226c:** canonical bank, qmap read, 228c decode, contradiction resolution, cross-domain utilization.

Internalization attempts (207–212) remain **out of scope** for the v1 product; 221+ is the **operational** migrate/recall/decode layer.

---

## Near-term engineering

- Persist `w_registry/` on Hugging Face.
- Stream policies + W version metadata on slots.
- Research branches: compositional W, temporal W, tool binding — promote only on clear gates.

---

## Paper / product framing

- Claim: *vector memory with family-equivariant remap* — write, hop, unlearn, **migrate**, decode, resolve.
- Deployment: classify corpus family → reuse or one-shot `learn_W(~800)` → never silent `arc_enc` fine-tune without a version bump.

---

## Related files

- [`extension_memory_contract.md`](extension_memory_contract.md)
- [`extension_thesis_W_remap.md`](extension_thesis_W_remap.md)
- [`docs/MEMORY_ENGINEERING.md`](../docs/MEMORY_ENGINEERING.md)
- Code: `_tapelm_ext.py`
