# TapeLM — простыми словами (RU)

> **Shipping trunk:** **221 → 227 → 228c → 230 → 226c**  
> **Публичный вход:** [`results/extension_plain_ru.md`](../../results/extension_plain_ru.md)

**Один продукт (variant A):** **факты = fp на символьном ink** — не store по id токенов, не «вставили абзац из RAG». Один curve encoder P1; текст наружу **ink→arcBPE** (не GPT BPE). Блок **192–205** — lexicon, слоты, hop, edit. Блок **213–230** — drift, домены, decode, конфликты в слотах.

---

## Главное

**Память = отпечатки слов (fp)** с одного encoder.  
**По умолчанию `arc_enc` frozen** — тогда fp и слоты стабильны (213).  
Если encoder всё же сдвинули — **не пересобираем весь банк**, учим маленькую **W** и/или пишем слоты в **каноническом** fp (227).  
Чтобы **использовать** найденное значение — **228c fp-decode**, не надеяться на CE-голову (226).  
Если в слоте два противоречивых значения — **230 policy**, не «кто первый записался» (229).

**Попробовать:** `python artifact/scripts/run_product.py` · [`artifact/QUICKSTART.md`](../../artifact/QUICKSTART.md).

---

## Закрытые ветки (213–220) — кратко

| Стадия | Идея | Итог |
|--------|------|------|
| 213 | freeze `arc_enc`, учить верх | fp ~0, wiki CE может упасть — **freeze = контракт** |
| 214 | recency вокруг сущности | **NO** — хуже baseline |
| 215 | adapter без remap | **NO** → см. 221 |
| 216 | FF-only, буквы frozen | **NO** — fp плывёт (~0.18–0.67 cos) |
| 217–218 | slow tape / snap hop | **NO** |
| 219 | decay старых слотов | **WIN** на stream-синтетике |
| 220 | sem sidecar PAWS | **NO** (мало gain) |

Подробнее (EN): [`results/extension_closed_branches.md`](../../results/extension_closed_branches.md).

---

## Stage 221 — матрица W (**да**)

**Сценарий:** encoder доучили на новом домене. Старые ключи слотов в **старом** fp.

**Решение:** W на ~800 core-слов: `fp_old ≈ W @ fp_new`. Ключи слотов: **W @ key_old**; запрос тоже через W в протоколе 221.

| Метрика | Полная W | Узкая W (256→32→256) |
|---------|----------|----------------------|
| Параметров | 65 536 | **16 384** |
| Align core | **~0.997** | **~0.967** |
| Recall фактов | **~0.78** | **~0.70** |
| Oracle reindex | **~0.87** | тот же |

**Probe 221:** W ≈ learnable warp; OOV/hold-out после W **~0.99**; **W_B / W_C** различаются; 100→400 core-слов для обучения W сходится.

**222:** лучше **W на ключах** + старый query (~0.73); new query без W на exam **~0.18**.  
**223:** «чужая W» не всегда хуже matched — переключатель домена **partial**.  
**224:** code shift жёсткий (cos **~0.59**); matched code W **~0.88**, чужая prose W **~0.68** (drop **~0.12**) → нужен **реестр семейств**, не одна W.  
**225:** `DOMAIN_BUNDLE_OK` — можно **reuse W_prose**; multi-head при frozen `arc_enc`; fp drift **0**.

Контракт L1–L3: [`results/extension_memory_contract.md`](../../results/extension_memory_contract.md).

---

## Stage 227 — канонический банк (**да**)

- **Пишем** ключи в **frozen canonical** fp (P1).
- **Читаем:** **qmap** — `W_bwd` переводит domain-query в пространство ключей.
- Same-domain **~1.0**; cross-code **~0.95**; drop vs matched **~0.05**.

**Смысл:** один банк слотов, «линзы» W disposable — не N копий памяти.

---

## Stages 226 / 226b — recall ≠ utilization (голова)

- **226** (первый прогон): выглядело как провал recall — шум протокола.
- **226b:** при протоколе **227** qmap recall **1.0**; inject в голову **~0.47** vs без **~0.43** — **utilization ≈ ноль**.
- **228a:** counterfactual inject — **partial**, не product path.

**Вывод:** CE-голова **не** надёжно использует retrieved gold → нужен **228c**.

---

## Stage 228b vs 228c — как доставать и выбирать

| Путь | Итог |
|------|------|
| **228b** global argmax по всем ключам + cos(fp, query) | **NO** (~0.33–0.40) |
| **228c** **4-way retrieve** (на каждое значение — лучший ключ) + **fp-scorer** `cos(fp(c), fp(retrieved))` | **YES** **1.0** vs head **~0.48** |

API: `_tapelm_ext.slot_retrieve_4way`, `fp_decode_pick_retrieved_4way`.  
W на диск / HF: `export_w_registry.py`, `w_registry/`.

---

## Stage 226c — сквозной cross-domain exam

`JOINT_FP_DECODE_OK`: canonical bank + code qmap + **228c** на return-токене.

- **recall_4way ~0.88**
- **fp decode ~0.88** vs **head ~0.45** (code-shift cos **~0.45**)

Это число для «продукт реально использует память», не старый 226 ~0.60.

---

## Stages 229–230 — противоречия в слотах

**229:** оба конфликтующих значения часто в top-2 (**~60%**); зазоры score маленькие — geometry не выбирает.

**230:** `RESOLUTION_POLICY_OK`

- raw **argmax** macro **~0.47** (bias к «official», первой записи)
- **query_cue** / **composite** macro **1.0** на neutral + cued + revision

Политика: `resolve_slot_contradiction` (provenance, recency, cue). Дополняет **detection** из 205.

---

## Stage 219 — decay (stream)

На синтетике с устаревшими слотами decay по возрасту **WIN**. Для длинного stream, не для одноразового exam. L3 в контракте — ещё wiring с version W на слотах.

---

## Что не в v1 (честно)

- Multi-domain **pretrain** encoder (нужен scale) — **open track**, не опровергнуто.
- Compositional W, temporal W, tool binding — ветки, см. EN pipeline.
- Internalization **210–212** — **THESIS_NO**; hops только **external fp loop** (203).

---

## Где смотреть цифры

| Что | Где |
|-----|-----|
| JSON verdicts | `results/stage213_decision.json` … `stage230_decision.json`, `stage226c_*`, `stage228c_*` |
| Копия для artifact | `artifact/decisions/` |
| Таблица стадий | `python artifact/scripts/show_map.py` |
| Полный narrative (EN) | [`results/plan_curve_dynamics.md`](../../results/plan_curve_dynamics.md) — секция *Memory extension program* |
| Preprint | [`results/preprint_tapelm_draft.md`](../../results/preprint_tapelm_draft.md) §4.8 |
| Концепт после 221 | [`results/extension_concept_after_221.md`](../../results/extension_concept_after_221.md) |
| Очередь прогонов | [`extension_run_queue_2h.md`](extension_run_queue_2h.md) |

Обновлено: **2026-07-30** (221–230 trunk).
