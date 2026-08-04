# 257 / 258 / 259 / 260f — full runs closed

Smoke notes below are **superseded** for verdicts. **Full-run narrative:** [`../results/stages_255_260_close.md`](../results/stages_255_260_close.md).

| Stage | Verdict (full) | Primary artifact |
|-------|----------------|------------------|
| **255** | `STREAM_INGEST_OK` | `results/stage255_decision.json` |
| **257** | `FP_COMPOSE_OK` | `results/stage257_decision.json` |
| **258** | `SEM_QUERY_OK` | `results/stage258_decision.json` |
| **259** | `HOT_SWAP_OK` | `results/stage259_decision.json` |
| **260f** | `OPEN_GATE6_OK` | `results/stage260f_decision.json` |

**258 headline (full):** unseen paraphrase **0.646** (curve+sem) vs **0.276** (matched GPT-2+sem); fp-only **0.000** on unseen.

---

## Archive — smoke log excerpts (historical)

### 257 — smoke

```
retrieval@cue 2hop:     chain_complete=1.0  hop0=1.0 hop1=1.0  halt_correct=1.0
overall=FP_COMPOSE_OK
em_2hop_glue=1.0  em_unseen_pair=1.0  em_shuffled=0.0
```

### 258 — smoke (smaller n)

```
curve+sem: seen_rel=1.000  unseen_para=0.583
gpt2+sem:  unseen_para=0.250
overall=SEM_QUERY_OK
```

### 259 — smoke

```
before=0.750 → new=1.000 old=0.000
gradient_steps=0 | overall=HOT_SWAP_OK
```

## How (unchanged design)

See [`stages_255_260_close.md`](../results/stages_255_260_close.md) § How the five fit together and reproduce commands.
