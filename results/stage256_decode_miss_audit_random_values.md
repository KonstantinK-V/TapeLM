# Stage 256 — decode audit (per-step)

Held-out **24** · first-word EM miss **3** · mechanism miss **3** · em_window3 **21** · copy_restart **10** · retrieve `auto`

Retrieval @ cue was rank **1.0**. Gate opens on step 0; copy has **no end-of-value** (restarts at first token). Fix = **257 span-lock**.

## Diagnosis counts

- **copy_no_span_lock**: 3

| S | gold | got | g0 | g_mean | n_tok | restart | diagnosis |
|---|------|-----|---:|-------:|------:|:-------:|----------|
| Voczazsac | Zevifumepo | Zevifer Zev | 1.000 | 0.723 | 6 | False | copy_no_span_lock |
| Bemikelve | Xifaqedoger | Xifaqedog | 1.000 | 1.000 | 7 | False | copy_no_span_lock |
| Docrefa | Naqasuzeref | Naqasuz | 1.000 | 0.895 | 8 | False | copy_no_span_lock |

## OK but copy restart (same disease)

- **Vegeh** `Fecelu` → `Fecelu Figh` (g_mean=0.831)
- **Demas** `Fufixu` → `Fufixu T F` (g_mean=0.800)
- **Nutso** `Dusoki` → `Dusoki T D` (g_mean=1.000)
- **Pesal** `Xilehej` → `Xilehej T X` (g_mean=1.000)
