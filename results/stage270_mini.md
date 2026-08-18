# Stage 270 lying tape

**LOOKUP_SUFFICES** · 4 witnesses, 2 lying · 12 subjects · 260 slots · trained params **0** · SMOKE

| arm | contradicted | clean |
|---|---:|---:|
| A lookup (top-1 slot) | **0.750** | 1.000 |
| B majority over witnesses | **0.750** | — |
| C similarity-weighted | 0.583 | — |
| D glue span-lock | 0.000 | 0.000 |

- liar is top-1: **0.250**, witness recall 1.000
- liars removed → lookup 1.000 (was 0.750)

## Gates (read G_lookup_fails first — it is validity, not result)

- G_clean_ok: **True**
- G_witnesses_reachable: **True**
- G_liar_causal: **True**
- G_lookup_fails: **False**
- G_majority_works: **False**
- G_aggregation_beats_lookup: **False**
- G_glue_aggregates: **False**
