# Stage 256 — decode audit (per-step)

Held-out **24** · first-word EM miss **6** · mechanism miss **5** · em_window3 **19** · copy_restart **19** · retrieve `auto`

Retrieval @ cue was rank **1.0**. Gate opens on step 0; copy has **no end-of-value** (restarts at first token). Fix = **257 span-lock**.

## Diagnosis counts

- **copy_no_span_lock**: 5
- **metric_first_word**: 1

| S | gold | got | g0 | g_mean | n_tok | restart | diagnosis |
|---|------|-----|---:|-------:|------:|:-------:|----------|
| Vegeh | Diavolo | Diavli D | 1.000 | 0.665 | 5 | True | copy_no_span_lock |
| Demas | Shara | Sharaif Shara , | 1.000 | 0.597 | 2 | True | metric_first_word |
| Voczazsac | Cheese | Chef Chef | 1.000 | 0.667 | 3 | True | copy_no_span_lock |
| Bemikelve | Sphinx | Sphrine Sph | 0.978 | 0.621 | 4 | True | copy_no_span_lock |
| Docrefa | Densetsu | Densetsh Dens | 1.000 | 0.765 | 4 | True | copy_no_span_lock |
| Pehibedza | Markbreit | Markbrech Markb | 1.000 | 0.802 | 4 | True | copy_no_span_lock |

## OK but copy restart (same disease)

- **Bapozo** `Adding` → `Adding Sen Adams` (g_mean=0.821)
- **Ricafna** `Sinnott` → `Sinnott C S` (g_mean=0.994)
- **Kettiramla** `Hollinwood` → `Hollinwood C H` (g_mean=1.000)
- **Rigacmi** `Congo` → `Congo C Congo '` (g_mean=0.895)
