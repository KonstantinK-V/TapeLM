# Stage204 — W1 noise/OOV robustness

**Overall:** `NOISE_ROBUST_WIN`

| noise p | A id seen curve/gpt | A id OOV curve/gpt | B recall curve/rag | BPE pieces/word |
|---------|---------------------|--------------------|--------------------|-----------------|
| 0.0 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 3.93 |
| 0.1 | 0.883 / 0.549 | 0.930 / 0.747 | 1.000 / 0.933 | 4.49 |
| 0.2 | 0.841 / 0.507 | 0.890 / 0.707 | 0.993 / 0.887 | 4.59 |
| 0.3 | 0.801 / 0.403 | 0.860 / 0.567 | 1.000 / 0.880 | 4.76 |

- relative drop p0→p0.3 (fact recall): curve 0.000 vs rag 0.120
- **B2 hardened (noise in stored corpus AND query, 8-way, chance 0.125):** p=0.0 curve 1.000 / rag 0.993 · p=0.2 curve 0.947 / rag 0.733 · p=0.3 curve 0.913 / rag 0.627
- gates: id_seen=True id_oov=True fact=True degrade=True hard_B2=True
- slots=547, chance B=0.25