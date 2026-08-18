# Stages 261–262 — research log (metrics)

Краткая карта: **261** = natural NL query, open wiki bank; **262** = тот же экзамен, что **258**, но trunk заменён на внешний LM. Подробнее: [`stage261_close.md`](stage261_close.md), [`stage261_mixer_ablation.md`](stage261_mixer_ablation.md), [`stages_255_260_close.md`](stages_255_260_close.md) (261 вне bundle 255–260f).

---

## Stage 261 — вопрос и шкала

**Цель (substance):** реальный вопрос из wikitext (без cue-template) находит слот, куда записан факт; банк с **4000+ noise**; разделение по **lexical overlap** write/ask.

**Масштаб (headline):** **top1** по ~4353 ключам. Порог «работает на масштабе» в коде: **`NL_QUERY_OK`** (top1 tape ≥ **0.30**, +0.10 vs fp, causal shuffle, low-overlap). Иначе **`NL_QUERY_NO_AT_SCALE`**.

**Банк (full):** 353 exam + 4000 wiki noise · 177 eval · trunk `stage253_joint_l02.pt` · chance top1 ≈ **0.00023** · 20-way chance **0.05**.

### Baseline (blend, full)

| Channel | top1 | acc_20way | shuf 20-way | α |
|---------|-----:|----------:|------------:|--:|
| fp-only | 0.034 | **0.220** | 0.062 | 0 |
| fp+sem | 0.000 | 0.090 | 0.062 | 0.72 |
| GPT-2+sem | 0.000 | — | — | — |

**Overall:** `NL_QUERY_NO_AT_SCALE` · **finding:** fp **~4.4×** 20-way chance; blend **ломает** fp (0.22→0.09).

### Ablation `--no-anchor` (ctx-only keys + queries)

Write/ask anchors are **different entities**; 256-style `anchor+ctx` may add uncorrelated noise on both sides.

| | top1 fp | fp 20-way | fp+sem 20-way | shuf 20-way |
|--|--------:|----------:|--------------:|------------:|
| **baseline** (anchor) | 0.034 | **0.220** | 0.090 | 0.062 |
| **`--no-anchor`** | 0.023 | **0.226** | 0.119 | (see JSON) |

**Read:** **20-way ≈ unchanged** (0.220→0.226); top1 не вырос. Гипотеза 1 **слабая**.

### Variant 2 — entity in question

**Diagnostic (natural query, baseline bank):** **23/177** eval items have entity string in prefix (`ent_in_query` leak). Split fp 20-way: **absent 0.058** (n=154) vs **leak 0.130** (n=23) — leak чуть лучше, но большинство **без** имени.

**Ceiling test `--query-names-entity`** (append gold entity to query fp+trunk): fp 20-way **0.062** (хуже baseline **0.22**), top1 **0**, α→**0.99**. Именование сущности в запросе **не** поднимает масштаб — на этом рецепте даже ломает.

Артефакт: [`stage261_decision_query_names_entity.json`](stage261_decision_query_names_entity.json)

Артефакт baseline (curve): [`stage261_decision.json`](stage261_decision.json)

### External trunk on 261 (`--model Qwen/Qwen2.5-0.5B`)

Query **`h_t`** from 262 `ExternalTrunk`; **fp keys / W_q fp arm unchanged** (comparable fp-only).

| | top1 fp | fp 20-way | fp+sem 20-way | α sem |
|--|--------:|----------:|--------------:|------:|
| **curve** (baseline) | 0.034 | **0.220** | **0.090** | 0.72 |
| **Qwen 0.5B** | **0.085** | **0.209** | **0.090** | 0.93 |

**Read:** **sem still harms** (0.09 ≈ curve); 261 **не оживает** по sem 20-way. **Fp top1** вырос (~0.08 vs 0.03) при ~ том же fp 20-way — stronger reasoner helps **open rank** slightly, not blend calibration.

Артефакт: [`stage261_decision_Qwen_Qwen2.5-0.5B.json`](stage261_decision_Qwen_Qwen2.5-0.5B.json)

---

## Stage 261f — word votes (zero train)

Скрипт: `_stage261f_word_votes.py` · тот же `collect`/wiki exam, **0** обучаемых параметров.

| run | 20-way | top1 | low-ov top1 | popularity 20-way | pop top1 | overall |
|-----|-------:|-----:|------------:|------------------:|---------:|---------|
| **exact** | **0.601** | **0.246** | **0.024** | 0.290 | 0.000 | `WORD_VOTES_BEATS_MEAN` |
| **soft + typo 0.15** | **0.509** | **0.192** | 0.012 | 0.290 | 0.000 | `WORD_VOTES_BEATS_MEAN` |

vs 261 ctx_fp mean **0.226** / top1 **0.034** / low-ov **0.000**. Gates: `G_causal_top1` (top1 vs popularity floor), `G_beats_popularity_20way` (+0.15), `G_beats_mean_fp`, `G_low_overlap_works`; `G_open_top1` false (0.246 < 0.30). Eval **n=338** (все пары, без fit-split; 261 eval **n=177**).

Артефакты: `stage261f_decision.json`, `stage261f_decision_soft_typo015.json`, `stage261f_mini*.md`.

---

### Шаги исследования 261

1. **Wiki noise bank** — закрыли shortcut «указать один из 26 fit-слотов» (smoke → full).
2. **Mixer ablations** (`--recipe fix*`) — fp floor, RRF, 260f gate, α cap, max-fusion; честный **`read`** (trained W_q, не init snapshot).
3. **Tape (без α)** — rerank / dualkey / symkey / read-**value** / **qkey** (write→question key).
4. **Честный rerank** — убрали inject gold в top-k и max-fp hack; перемерили (`tape_rerank` v2).

**Скрипт:** `_stage261_nl_query.py` · решения: `stage261_decision*.json` · логи: `_stage261_full_*.out`

### Mixer ablations (fp 20-way = trained)

| recipe | fp 20-way | sem/tape 20-way | shuf 20-way | overall |
|--------|----------:|----------------:|------------:|---------|
| baseline | 0.220 | 0.090 | 0.062 | `NO_AT_SCALE` |
| fix1 | 0.209 | 0.107 | 0.040 | `NWAY_FP_ONLY` |
| fix1m | 0.113 | 0.130 | 0.051 | `NWAY_FP_ONLY` |
| fix1p | 0.102 | 0.113 | 0.051 | `NWAY_FP_ONLY` |

Ни один fix не даёт **sem ≥ fp** при fp **~0.22**.

### Tape / conceptual (full, honest where noted)

| recipe | fp 20-way | tape 20-way | tape top1 | shuf top1 | shuf 20-way |
|--------|----------:|------------:|----------:|----------:|------------:|
| tape_rerank **(honest)** | 0.096 | 0.090 | **0.006** | **0.000** | 0.056 |
| tape_rerank_val | 0.096 | 0.096 | 0.000 | 0.000 | 0.056 |
| tape_dualkey | 0.107 | 0.158 | ~0.017 | ~0.017 | 0.181 |
| tape_symkey | 0.096 | 0.141 | ~0.006 | ~0.000 | 0.034 |
| tape_qkey | 0.096 | **0.130** | 0.000 | 0.000 | **0.045** |

**Read:** старый rerank **top1 0.27** — артефакт (gold в pool + low-fp outlier). Честный rerank **= fp** на top1; лучший честный 20-way среди tape — **qkey ~0.13**, **dualkey ~0.16** (fp-pretrain слабее baseline 0.22).

Ловушки гейтов: [`stage261_mixer_ablation.md`](stage261_mixer_ablation.md)

### 261 — что закрыто / нет

| Слой | Статус |
|------|--------|
| **Substance** (20-way fp, causal) | **Частично:** fp **0.22**, shuffle ~chance |
| **Sem / mixer** | **Нет:** blend harm; fixes не восстановили fp+sem |
| **Масштаб** (open top1) | **Нет:** top1 ≈ 0 (GPT parity) |
| **NL_QUERY_OK** | **Нет** |

---

## Stage 262 — trunk swap (258 exam)

**Вопрос:** переносится ли **258**-канал (P1 fp keys + `W_sem(h_t)`) на **внешний** frozen trunk, без смены ленты?

**Экзамен:** как **258** (`para_hold`, seed 258) · 64 subjects · 1712 tape slots · chance **0.125** (8-way subject bank).

**Full run:** `Qwen/Qwen2.5-0.5B` (h=896) vs curve `stage253_joint_l02.pt` · smoke: `sshleifer/tiny-gpt2` → `TRUNK_SWAP_NO`.

### Метрики (bank top1 / sel_acc, unseen paraphrase главное)

| Trunk | unseen para | seen rel | anchored | sem loss |
|-------|------------:|---------:|---------:|---------:|
| fp-only | 0.000 | 0.057 | — | — |
| **curve** | **0.625** | 0.995 | 0.750 | 0.411 |
| **Qwen 0.5B** | **0.703** | 1.000 | 0.746 | 0.410 |

**Overall:** **`TRUNK_SWAP_OK`** — external **≥** curve на unseen para (**0.703 vs 0.625**); fp-only **0**; интерфейс `h_t → W_sem` переносится.

**Contrast 261 vs 262:** 262 — **closed** subject bank + authored relations; 261 — **open** entity bank + natural contexts. OK на 262 **не** снимает `NO_AT_SCALE` на 261.

Артефакты: [`stage262_decision.json`](stage262_decision.json) · [`stage262_mini.md`](stage262_mini.md) · `_stage262_qwen_full.out` · код: `_stage262_trunk_swap.py`

---

## Ссылки

| Stage | Decision | Код |
|-------|----------|-----|
| 261 baseline | `stage261_decision.json` | `_stage261_nl_query.py` |
| 261 ablations | `stage261_decision_fix*.json`, `stage261_decision_tape_*.json` | `--recipe …` |
| 262 | `stage262_decision.json` | `_stage262_trunk_swap.py` |

**Отложено (261):** joint fp **0.22** → freeze → qkey / rerank-on-value; read по тексту value; «ключ = ожидаемый вопрос» с сильнее write-side training.
