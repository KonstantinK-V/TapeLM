# Full-bank vs closed-pool — метрики и главный вывод

Снимок после `context_words`, аудита **256**, **`full_bank_*`**, **пошагового decode-аудита** и контроля **`--random-values`**.  
Decision: [`stage261f_decision.json`](stage261f_decision.json), [`stage256_decision.json`](stage256_decision.json), [`stage256_decision_random_values.json`](stage256_decision_random_values.json), [`stage263_decision.json`](stage263_decision.json), [`stage258_decision.json`](stage258_decision.json).  
Decode: [`stage256_decode_miss_audit.md`](stage256_decode_miss_audit.md) · random: [`stage256_decode_miss_audit_random_values.md`](stage256_decode_miss_audit_random_values.md).

---

## Главное

**Голоса — узкий claim:** доказаны на **открытом банке 261f**, не как универсальный индекс. **263** ничья и на full bank (planted exam, retrieval saturated).

**256 — ретривал и декод разведены:** `full_bank_top1 @ cue = 1.0` при **EM = 0.75**. Промахи не gate и не readout.

**Механизм сильнее ярлыка «BPE spelling».** У копи-канала **нет состояния «значение кончилось»**: после исчерпания value span `p_copy` снова ставит **первый** токен значения top-1 (**copy_restart 19/24**, в том числе на «ok»: `Hollinwood C H`, `Boats C Boat Bo`, …). Гейт на cue ≈ **1.0**, дальше отпускает (`gate_mean` 0.36–0.99). Исход решает не сила гейта, а конкурирует ли LM за продолжение префикса.

**Контроль `--random-values` опроверг предсказание «EM обвалится к доле однотокенных».** Nonsense values, **0/24** single-token → EM **0.875** (выше wiki **0.75**). Значит 0.75 **не** просто «заимствовано у весов, умеющих писать слова». Скорее: **LM-прайор двусторонний** — на реальных словах иногда дописывает (Demography), иногда **уводит** (Cheese→Chef, Diavolo→Diavli). Без словаря конкуренции меньше, soft-copy по префиксу держит span дольше. Запрещённый источник всё равно участвует, но не как единственная опора EM.

**Починка: span-lock — stage 265** (`SPAN_LOCK_OK`). B em_span **0.975** без переобучения; C хуже. Длина: soft распад / locked плоско (`p^N`→`p`). Headline = **em_span**; рядом **em_text** + `glue_bpe_rate` — дефект «LM приклеивает подслово» остаётся видимым. **Граница конца значения не решена**, вынесена за метрику одноответным экзаменом (следующая линия).

**258:** bank_mrr **0.778** (+sem) vs fp-only **0.235** — сильнейший sem-результат в наборе.

**Следующий бенч:** exam 256 + банк 261f (подмена ленты).

---

## Метрики

| Стадия | Банк | Headline | Full bank / контроль |
|--------|-----:|----------|----------------------|
| **261f** | 4338 | 20-way **0.601** vs mean **0.226** | top1 **0.246** vs mean **0.034** |
| **263** | 1248 | EM **1.0** = **1.0** | full top1 **1.0** = **1.0** |
| **256** wiki | 1248 | EM **0.75** | @ cue **1.0**; mechanism miss **5/24**; restart **19/24** |
| **256** random-values | 1248 | EM **0.875** | single-tok frac **0**; mechanism miss **3/24**; restart **10/24** |
| **258** | subject×4 | unseen_para **0.646** | bank_mrr **0.778** (+sem) |

Замер **8:** auto **144/144** votes.

---

## Decode-аудит (wiki values)

Пошагово: gate / `p_copy_gold` / `copy_rank_first_val` на каждом t.

| | Count |
|--|------:|
| first-word EM miss | 6/24 |
| mechanism (`copy_no_span_lock`) | **5** |
| metric only (`Sharaif Shara` → em_window3) | **1** (Demas) |
| em_window3 pass | 19/24 |
| copy_restart after value | **19/24** |

**Не** gate_low, **не** readout_copy. Step 0: g≈1, copy_rank_gold=1; отказ на шагах **2+** или увод LM с общего префикса.

OK с той же болезнью: Hollinwood C H, Boats C Boat Bo, Campus C Campus, Filth C Fil, Fabray C Fab, Craft Craft Craft.

---

## Контроль random-values

Предсказание: EM → ~single-token frac (**0**).  
Факт: EM **0.875**, single-tok **0/24**.

| | wiki entities | nonsense |
|--|-------------:|---------:|
| EM | 0.75 | **0.875** |
| mechanism miss | 5 | 3 |
| copy_restart | 19 | 10 |

**Вывод для вердикта 256:** болезнь = **нет span-lock** (restart + хрупкий greedy copy). Гипотеза «0.75 целиком из spelling prior словаря» — **отклонена** этим контролем; прайор **мешает и помогает**. Честная формулировка: лента отдаёт слот и roughly первый токен; **полное значение не заперто** — часть пути достраивает LM, часть ломает.

---

## Verdict cheat sheet

| Можно | Нельзя |
|-------|--------|
| Votes на 261f-scale open bank | Votes > mean на exam 256/263 |
| 256: retrieval OK; decode без span-lock; restart виден на ok | «EM 0.75 = только веса пишут слова» (random-values выше) |
| Починка = 257 span-lock | Лечить гейтом / postings на этом exam |
| 258 bank_mrr как strongest sem | Смешивать 258 с wiki-open без оговорки |
