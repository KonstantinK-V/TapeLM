# Stage 265 — C anomaly / end-of-value

## Что было

C: verbatim **1.0**, EM **0.85**. Шесть строк: спан верен, LM после него дописывает BPE вплотную (`Whammy`→`Whammyn`, `Rascal`→`Rascalibur`, …).

## Как считаем теперь

| метрика | что |
|---------|-----|
| **em_span** (headline) | first-word на выданном спане ленты |
| **em_text** | first-word на полной генерации до `max_new` |
| **glue_bpe_rate** | доля `em_span ∧ ¬em_text` |

Декод по-прежнему генерирует хвост (дефект видим в em_text); scoring headline — по спану. **Граница конца значения не решена** — только отложена одноответным экзаменом. Тот же класс, что restart: модель не знает, что значение кончилось. На большой ленте с продолжением после value стоп не спасёт.

См. `deferred.end_of_value_boundary` в [`stage265_decision.json`](stage265_decision.json).
