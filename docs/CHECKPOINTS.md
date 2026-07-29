# Checkpoints & data

Git does **not** store weights (see root `.gitignore`).

## TapeLM product encoder (required for 192–212 demos)

| File | Role |
|------|------|
| `checkpoints/stage191_p1_curve.pt` | **P1** frozen curve encoder (d256, 6L) |
| `checkpoints/stage191_p2_gpt.pt` | Matched GPT control |

Optional scaling ablations: `checkpoints/stage209_curve_d*_L*.pt`, `stage209_gpt_*`.

## BPE tokenizer

Built by Stage 177; cached at `results/stage177_curve_bpe_tokenizer.json` (created on first run if missing).

## Document cache (Stage 191)

`data/` holds wiki-derived caches (e.g. id-doc npz). Regenerate via `_stage191_night.py --phase p0` if absent.

## Legacy SOTE checkpoints

Hundreds of files under `checkpoints/` from pre-170 experiments — **not** needed for TapeLM. Safe to delete locally if disk is tight; keep P1/P2 above.
