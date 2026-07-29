# Stage208 — hybrid rare-word fp reranker over frozen A

**Overall:** `HYBRID_NO_GAIN`

| scorer | all | common band | rare band |
|--------|-----|-------------|-----------|
| A only (BPE CE head) | 0.551 | 0.669 | 0.431 |
| fp reranker only | 0.396 | 0.435 | 0.355 |
| combined (w=0.25) | 0.541 | 0.667 | 0.413 |
| gated by fp-surprise (0.0, 0.5) | **0.544** | 0.669 | **0.416** |

- 4-way, frequency-matched candidates within band, chance 0.25; test n=700
- gates: no_degrade=True rare_win=False read_only=True