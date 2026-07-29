# Curated verdict files

Snapshots from `results/stage*_decision.json` for **GitHub browsing** and `show_map.py` — the public face of the staged program (parity wins, RAG comparisons, falsifications).

| Stage | Theme |
|-------|--------|
| 196 | TapeLM assemble (main scorecard) |
| 192 | FP lexicon calibration |
| 203–205 | Hops, noise vs RAG, unlearn |
| 207–209 | Variant B falsified, hybrid no gain, semantic scaling |
| 210–212 | Internalization frontier (all THESIS_NO) |

Refresh after new runs:

```bash
python artifact/scripts/sync_decisions.py
```

Narrative: [`../OVERVIEW.md`](../OVERVIEW.md) · Full JSON in [`../../results/`](../../results/).
