# Stage 191 — NIGHT-9h: scale test on RTX 3050 (4.3 GB)

Question of the night: **does the curve keep GPT parity at scale, and does scale move meaning?**

Base facts (day of 184–190): measurer trusted (Exam v2); curve clean-CE self-model = GPT parity
(0.727) at 20M chars / 3k steps; hand losses poison; surprise must be read-only; ink lacks rarity
signal; doc-contrast learns topical, not semantic invariant.

Budget: ~9h wall-clock, sequential phases, each with hard time guard + checkpoint every eval —
a crashed/overrun night still yields data. VRAM target ≤3 GB (fits 4.3 GB card with margin).

| Phase | Time | What | Success gate |
|-------|------|------|--------------|
| **P0** data+exam | ~0.5h | Corpus 20M→**150M** chars; cache id-docs to disk (np). **Exam v3**: same freq-matched protocol, bigger (300 next_tok / 150 entity / 100 ood), built from held-out slice; unigram+random baselines auto-checked | unigram ≤ 0.35 |
| **P1** curve-XL | ~3h | Self-model dual-channel scaled: **D 128→256, fast 4L→6L**, T=64, micro 16, clean CE + read-only surprise (W_SELF=0.1). **15k steps**, cosine LR, eval every 2.5k, early-stop if next_tok flat 2 evals | beats own 0.727; tracks P2 |
| **P2** GPT-XL control | ~1.5h | GPT-2 matched (~same params d256/6L/T64 eq.), same data, same 15k steps, same exam | honest ceiling number |
| **P3** rarity ink | ~1.5h | P1 recipe + **char-trigram log-freq feature** into arc encoder (rarity channel BPE gives GPT for free). 10k steps. Gates: surprise@fake > @real; entropy after fake > real (S3-G3 finally?) | G3 calibration fixed |
| **P4** meaning sweep | ~1h | Gate B (para vs hard, extended pairs) + doclink + entity for P1/P2/P3 **and** old 187 — the "does scale move meaning" measurement; GPT gets gate B too (never measured) | direction, not pass/fail |
| **P5** report | ~0.5h | stage191_night_report.md + decisions json + plan update | — |

Buffer ≈ 1h. Runner: `_stage191_night.py --phase all` (or per-phase), each phase writes its own
decision json; runner skips completed phases on restart (idempotent).

Precomputed step-cost calibration before committing: 200-step probe of P1 config; if step > 0.75s,
auto-reduce to 12k steps (keeps night inside 9h).

Verdicts:
- `NIGHT_PARITY_HELD` — P1 within 0.03 of P2 on exam v3 next_tok
- `NIGHT_CURVE_AHEAD` / `NIGHT_GPT_AHEAD` — gap > 0.03 either way
- `NIGHT_G3_FIXED` — P3 passes OOD calibration
- `NIGHT_MEANING_MOVES` — gate B gap (hard−para) shrinks with scale on ANY system
