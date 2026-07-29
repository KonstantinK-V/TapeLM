# Stage203 — internal hops (free-form vs structured)

**Overall:** `INTERNAL_HOPS_YES_IF_STRUCTURED`

- **soft-follow (structured) test:** k1=1.00 k2=1.00 k3=1.00 (chance 0.25)
- free-form test: k1=0.28 k2=0.15 k3=0.20 (train 1.00 → overfits=True)
- external hand-loop test: k1=1.00 k2=1.00 k3=1.00
- anti-CF (encoder untouched): True, tape slots=3719, T=3