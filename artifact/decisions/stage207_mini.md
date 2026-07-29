# Stage207 — curve as thinking (fp-generative vs token CE)

**Overall:** `CURVE_THINKING_NO` (1/4 gates)

| gate | curve | CE / baseline | pass |
|------|-------|---------------|------|
| G1 quality (in-vocab, 4-way) | 0.536 | 0.861 | False |
| G3 open-vocab OOV (4-way, chance 0.25) | **0.315** | 0.245 | False |
| G2 drift raw first→last10 | 0.889→0.834 | snap rep 0.857 | True |
| G4 unified trunk-hidden memory | 0.175 | — | False |

- shared frozen arc-encoder input; both trunks 3.3M params, 3500 steps.
- G3 is the essence gate: CE's closed softmax gives every OOV word the same UNK logit → chance;
  the curve ranks them by fingerprint in an open metric vocabulary.