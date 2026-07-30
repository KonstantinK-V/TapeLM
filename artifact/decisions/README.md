# Curated verdict files

Snapshots from `results/stage*_decision.json` for GitHub browsing and `show_map.py` — the public scorecard for TapeLM variant A (generation parity, fp stack, memory trunk, baselines).

| Stage | Theme |
|-------|--------|
| 196 | Main assemble scorecard |
| 192–198 | Calibration, recall, hops, edit, stream |
| 203–205 | Structured hops, noise vs RAG, unlearn |
| 221–230, 226c | Canonical memory, W, decode, resolution |
| 207–212 | Explored directions (scope notes in OVERVIEW) |

```bash
python artifact/scripts/sync_decisions.py
```

Narrative: [`../OVERVIEW.md`](../OVERVIEW.md) · Full JSON: [`../../results/`](../../results/).
