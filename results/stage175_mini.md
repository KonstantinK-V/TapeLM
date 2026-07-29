# Stage175 — attn pen context gate

**Overall:** `ATTN_PEN_CONTEXT_NULL`

- Early stop @~6k: A_same stayed **0.99–1.0** (same wipe as GRU)
- Causal Transformer pen can attend to prefix, but **Δ-only training** makes it local
- Architecture alone ≠ context
- Next: explicit **context-retention** objective on pen (push apart same-suffix / different-prefix endpoints), then re-gate A/B — not deeper attn
