# Stage170 — curve dynamics smoke

**Verdict:** `CURVE_DYN_SMOKE_YES`

- verdict `CURVE_DYN_SMOKE_YES` wall=0.07h steps=30000
- final cos_delta=0.858 lift_vs_mean=+0.882 lift_vs_copy=+1.301
- baselines mean=-0.025 copy=-0.443 zero=0.000
- loss = latent next-z + delta only (no text CE)
- 169 frozen; do not resume word-battery path unless reopened
