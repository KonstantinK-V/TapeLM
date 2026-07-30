# Why TapeLM?

> **Shipping trunk:** **221 → 227 → 228c → 230 → 226c** · try it: [`QUICKSTART.md`](QUICKSTART.md)

## Root: characters, not tokens

**Facts as fingerprints on character ink** — not a token-id store, not retrieved paragraphs.

Most stacks — including typical RAG — key retrieval on **token embeddings**. TapeLM’s substrate is a **character stream**: symbols move a **curve**; **fingerprints** for words are built from that ink. Memory modules (slots, calibration, bind/hop, resolution) use **fp**; text goes out as **arcBPE** — BPE readout from the curve after ink, not **GPT BPE** (token ids in and as memory keys).

That design choice explains **204**: under noise, **GPT BPE** keys **re-fragment**; character fps change more **smoothly** — measured as **0.913 vs 0.627** vs fair GPT+RAG on our exam. One encoder serves generation and memory without a separate index embedder.

---

Most systems force a choice: **knowledge in weights** (hard to edit, catastrophic forgetting) or **knowledge in a RAG index** (retrieve text, re-prompt, hope the LM uses it). TapeLM is built around a third idea:

**Knowledge as structured vectors in the same space as generation** — write a fact, bind two entities, hop across relations, resolve a contradiction, or read across domains **without** treating memory as pasted-in paragraphs and **without** training the backbone on every update.

That is not marketing. Every step below has a stage script, a JSON verdict, and (where it matters) a **fair GPT+RAG** control trained to the same rules.

---

## The five strongest ideas (conceptual)

### 1. One geometry, not “RAG with a different embedder”

One character-curve encoder (**P1**) defines both **how you speak** and **how you fingerprint** words. Memory keys, calibration (“is this word in *my* lexicon?”), and multi-hop steps live in **one fp-space**. A full RAG stack can tie you on **clean** cosine retrieval — we say that openly — but it does not give you **bind / hop / subject-write / resolve** as first-class vector operations on the **same** map.

### 2. Structure of knowledge, not opaque chunks

Facts are **subject-anchored slots**. Relations can use **binding** (`fp_A ⊙ fp_B`). Multi-hop queries are **chains in vector space** (195, 203) — not necessarily “retrieve paragraph → ask again.” When two writes disagree, **230** picks among structured candidates; that is a different problem from merging conflicting prompts.

### 3. Frozen encoder, living memory (anti–catastrophic forgetting)

**P1 is pretrained once, then frozen** for product use. New facts = **slot writes**, not gradient updates. Delete a fact → collateral-free (205). This is the continual-learning story **without** wrecking generation — shared with external memory in spirit, but unified with the curve substrate and explicit policies (surprise gating, budget, resolution).

### 4. Where we **beat** a strong rival (not just “different packaging”)

On **fair GPT+RAG**, two axes break parity in our measurements:

- **Noisy / OOV text (204):** hardened recall **0.913 vs 0.627** — typos shatter **GPT BPE** keys; character fps degrade smoothly.
- **Naive parametric unlearn (205):** slot delete leaves the rest intact; gradient unlearning destroys most retained facts.

These are **capability** results on defined exams, not architecture slides.

### 5. The product trunk — memory you can **use**, not just retrieve

The line we ship and demo:

**221 → 227 → 228c → 230 → 226c**

| Step | Idea | Why it matters |
|------|------|----------------|
| **221** | Tiny **W** after encoder drift | Migrate coordinates, don’t rebuild the whole bank |
| **227** | **Canonical** slot bank + **qmap** | One store, many domains |
| **228c** | **Fp decode** on retrieved value | CE head often **ignores** memory (~0.45); fp scorer **~1.0** on the same exam |
| **230** | **Resolution** policy | Conflicts are policy, not “pick highest cosine” (~0.47 → ~1.0) |
| **226c** | Cross-domain **e2e** | **~0.88** utilization via fp decode vs **~0.45** head alone |

`python artifact/scripts/run_product.py` is written to walk this trunk.

---

## Honest boundaries (why you should trust the positives)

We **closed** several seductive directions so they cannot be confused with the product:

- **210–212:** hops / slow tape / instance identity **inside** the transformer forward — **THESIS_NO**; external fp loops remain the hop API.
- **207:** predict next **fingerprint** instead of token — **falsified**.
- **208:** fp rerank on the **arcBPE** head — **no gain** on clean text.
- **Clean static recall:** fair GPT+RAG can **match** us — parity is not the headline; **structure + noise/unlearn + trunk utilization** are.
- **231–235:** post-trunk **ops** (temporal W, stream+`w_version`, tool bind, compositional W, mixed L1 probe) — measured, useful for deploy hygiene, **not** the product demo path.

Full map: [`docs/STAGES.md`](../docs/STAGES.md) · [`results/plan_curve_dynamics.md`](../results/plan_curve_dynamics.md).

---

## Read order for humans

1. **Run** — [`QUICKSTART.md`](QUICKSTART.md) (5 minutes, GPU optional for map-only).
2. **Picture** — [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) (diagram + frozen-P1 table).
3. **Paper tone** — [`../results/preprint_tapelm_draft.md`](../results/preprint_tapelm_draft.md) (§5.1 vs RAG, §4.8 trunk).
4. **Numbers** — `python artifact/scripts/show_map.py` · [`decisions/`](decisions/).

---

## One paragraph for citations

TapeLM couples a frozen dual-channel curve encoder with zero-train-at-inference fingerprint memory: structured slots, binding, hops, calibration, and a shipping trunk (**221→227→228c→230→226c**) for canonical storage, domain **W**, fp decode utilization, and conflict resolution — one geometry for generation and memory, with staged wins vs vanilla GPT and vs fair GPT+RAG on noise and unlearning, and honest parity on clean retrieval.
