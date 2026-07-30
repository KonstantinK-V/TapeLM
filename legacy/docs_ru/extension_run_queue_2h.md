# Очередь прогонов ~2 часа (extensions 221+)

Цель: не «ещё один YES», а **развести сценарии деплоя** и **границы архитектуры** (один fp-bank + domain-W vs reindex vs freeze).

**Перед стартом:** `checkpoints/stage191_p1_curve.pt`, данные `data/` как в 221.

```bash
python artifact/scripts/sync_decisions.py
```

---

## Блок A — ~15 мин (база + артефакт)

| # | Команда | Зачем |
|---|---------|--------|
| A1 | `python artifact/scripts/sync_decisions.py` | Probe и 213–221 в `artifact/decisions/` для GitHub-читателя |
| A2 | `python artifact/scripts/show_map.py` | Быстро сверить curated core vs extensions |
| A3 | Просмотр `results/stage221_probe_decision.json` | Уже есть полный probe; зафиксировать цифры в заметках |

---

## Блок B — ~35–45 мин (главный: как **реально** читать память)

**222 — режимы деплoya fp** (один arc-shift Stories, те же 60 фактов):

```bash
python _stage222_fp_deploy_modes.py
python _stage222_fp_deploy_modes.py --smoke   # ~8 мин sanity
```

Сравнивает recall при:

- старые ключи / старый encoder на query  
- старые ключи / **новый** encoder на query (**без W**)  
- `W @ K_old` + разные комбинации query (old vs new, W или нет)  
- oracle (новые ключи + новый query)

**Интерес:** если «new query + old keys» уже ~0.97, а 221-style `W(old query)` ~0.78 — ты понимаешь, **когда W обязательна** (legacy pipeline), а когда достаточно **только нового encoder** на query.

---

## Блок C — ~35–45 мин (граница «архитектура vs хак»)

| # | Команда | ~время | Гипотеза |
|---|---------|--------|----------|
| C1 | `python _stage221_probe.py` | ~6 мин | Уже прогоняли; повтор только если менял код |
| C2 | `python _stage223_cross_adapter.py` | ~25–35 мин | **W_B vs W_C на чужом shift**: matched vs wrong adapter — оправдан ли «один банк + переключатель W» |
| C3 | `python _stage221_fp_remap_adapter.py --smoke` | ~3 мин | Регрессия gates после правок |

**C2** — новый скрипт (см. ниже). Это самый «концептуальный» прогон после 222.

---

## Блок D — ~20–30 мин (если осталось время)

Выбери **одну** линию:

| Линия | Команда | Что можешь найти |
|-------|---------|------------------|
| **Freeze contract** | `python _stage213_arc_enc_freeze_finetune.py --smoke` | fp=0 при frozen arc_enc; цена CE на wiki |
| **Stream** | `python _stage219_stream_decay.py` | WIN на stale slots — стоит ли в stream-продукте |
| **Закрытые ветки** | `python artifact/scripts/run_pipeline.py --from 214 --to 216 --smoke` | Быстро переубедиться, что recency/partial FF мертвы |
| **Сильнее shift** | `ARC_STEPS=1600 python _stage221_fp_remap_adapter.py` (если добавишь env в скрипт) | Ломается ли OOV/hold-out при cos ≪ 0.68 |

---

## Блок E — ~5 мин (оформление)

```bash
python artifact/scripts/sync_decisions.py
```

Update a short note in `results/extension_memory_contract.md` after 222/223 if gates change.

---

## Приоритет если время жмёт

1. **222** (deploy modes) — must  
2. **223** (wrong W) — must для multi-domain story  
3. sync + RU summary  
4. Остальное по appetite  

---

## Чего **не** ждать за 2 часа

- Multi-domain **pretrain** arc_enc на Wiki+Stories+Code (open track, GPU-дни)  
- Stage 210–212 переоткрытие (THESIS_NO)  
- Полный `run_pipeline.py --from 213 --to 221` без smoke (~часы на CPU)
