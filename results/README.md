# Results

Staged evidence for TapeLM: narratives, preprint draft, and per-stage verdict files.

**New readers:** prefer the curated copy in [`../artifact/decisions/`](../artifact/decisions/) and the one-page story in [`../artifact/OVERVIEW.md`](../artifact/OVERVIEW.md).

---

## Core documents (in git)

| File | Content |
|------|---------|
| [`plan_curve_dynamics.md`](plan_curve_dynamics.md) | Full program narrative, scorecard, stage commentary |
| [`preprint_tapelm_draft.md`](preprint_tapelm_draft.md) | Preprint-shaped prose (abstract, protocol, negatives) |
| [`pre_publish_frontier.md`](pre_publish_frontier.md) | Closed frontier 210–212 |
| `stage*_decision.json` | Machine-readable overall verdict per stage |
| `stage*_mini.md` | One-page stage summaries where present |
| `stage177_curve_bpe_tokenizer.json` | Small tokenizer artifact needed to run stages |

---

## Regenerated locally (usually gitignored)

- `_stage*_log.txt`, `_stage*_console.txt` — verbose run logs
- Large intermediate JSON from long trains (see root [`.gitignore`](../.gitignore))

Refresh the visitor-facing subset after important runs:

```bash
python artifact/scripts/sync_decisions.py
```

---

## Root pointer

Repository entry: [`../README.md`](../README.md) · Architecture: [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)
