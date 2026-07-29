# Stage188 — surprise-conditioned head

**Overall:** `SURPRISE_HEAD_PARTIAL_3`

- G1: next_tok=0.700 vs 187 0.727 → True
- G2: surprise seen=0.0688 unseen=0.0713 → True
- G3: entropy real=4.964 fake=4.552 → False
- diag: surprise@span real=0.0655 fake=0.0545 (fake>real=False)
- temp w=3.82 b=-2.65
