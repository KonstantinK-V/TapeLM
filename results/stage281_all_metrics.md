# Stage 281 — все прогоны, метрики

Только реальные запуски `_stage281_frames.py`. Порог «280 стоит снова»: after ≥ **0.375** и tie `teacher_abstain` ≥ 0.5.

## Сводка

| run | overall | smoke | ceiling before → after | kept frames | asserts kept/total |
|---|---|---|---:|---:|---:|
| **full (night, 400 addr)** | `NO_FRAME_SURVIVES` | False | -0.544 → — | 1 | 6/3011 |
| **smoke base** | `NO_FRAME_SURVIVES` | True | — → — | 0 | 0/392 |
| **smoke loose** | `FRAMES_DO_NOT_HELP` | True | -0.192 → NaN | 2 | 9/392 |
| **smoke loose + frame-in-address** | `FRAMES_DO_NOT_HELP` | True | -0.192 → NaN | 2 | 9/392 |
| **smoke and_ok thresholds** | `NO_FRAME_SURVIVES` | True | — → — | 0 | 0/392 |

## Детали

### full (night, 400 addr)

Источник: `_stage281_full.out` · **`NO_FRAME_SURVIVES`** · smoke=False · frame_in_address=False

| gate | value |
|---|---|
| `G_frames_survive` | True |
| `G_ceiling_clears_silence` | False |
| `note` | after held-out wiped (n_slots=0); decision JSON later overwritten by smoke |

**ceiling BEFORE**

- n_items=265, reward=-0.544, addresses=265, slots=1799
- families: `{'clean': 51, 'decidable': 126, 'tie': 88}`

| family | n | teacher_acc | teacher_abstain |
|---|---:|---:|---:|
| clean | 51 | 0.686 | 0.0 |
| decidable | 126 | 0.325 | 0.016 |
| tie | 88 | 0.0 | 0.011 |

**ceiling AFTER**

- n_items=0, reward=—, addresses=None, slots=None

| family | n | teacher_acc | teacher_abstain |
|---|---:|---:|---:|

### smoke base

Источник: `stage281_decision_281_base.json` · **`NO_FRAME_SURVIVES`** · smoke=True · frame_in_address=None

| gate | value |
|---|---|
| `G_frames_survive` | False |

**ceiling BEFORE:** нет данных

**ceiling AFTER:** нет данных

### smoke loose

Источник: `stage281_decision_281_loose.json` · **`FRAMES_DO_NOT_HELP`** · smoke=True · frame_in_address=False

Пороги: min_n=3, min_confirm=0.05, min_anchors=1, max_vpa=5.0, allow_empty=False

| gate | value |
|---|---|
| `G_frames_survive` | True |
| `G_tape_shrinks` | True |
| `G_kept_frames_functional` | True |
| `G_ceiling_clears_silence` | False |
| `G_ceiling_improves` | False |
| `G_teacher_abstains_on_tie` | False |

**ceiling BEFORE**

- n_items=36, reward=-0.192, addresses=36, slots=144
- families: `{'decidable': 15, 'clean': 12, 'tie': 9}`

| family | n | teacher_acc | teacher_abstain |
|---|---:|---:|---:|
| clean | 12 | 0.833 | 0.0 |
| decidable | 15 | 0.467 | 0.0 |
| tie | 9 | 0.0 | 0.0 |

**ceiling AFTER**

- n_items=1, reward=NaN, addresses=None, slots=None

| family | n | teacher_acc | teacher_abstain |
|---|---:|---:|---:|

**Top frames (by confirm):**

| frame | n | confirm | dispute | anchors | vpa | empty |
|---|---:|---:|---:|---:|---:|---|
| `the` | 6 | 0.83 | 0.00 | 1 | 1.00 | False |
| `the county seat` | 3 | 0.67 | 0.00 | 1 | 1.00 | False |
| `` | 373 | 0.57 | 0.29 | 55 | 2.95 | True |
| `title assigned bishop` | 3 | 0.00 | 0.67 | 1 | 3.00 | False |
| `title assigned archbishop` | 3 | 0.00 | 0.67 | 1 | 3.00 | False |
| `protector` | 4 | 0.00 | 0.75 | 1 | 4.00 | False |

### smoke loose + frame-in-address

Источник: `stage281_decision_fia.json` · **`FRAMES_DO_NOT_HELP`** · smoke=True · frame_in_address=True

Пороги: min_n=3, min_confirm=0.05, min_anchors=1, max_vpa=5.0, allow_empty=False

| gate | value |
|---|---|
| `G_frames_survive` | True |
| `G_tape_shrinks` | True |
| `G_kept_frames_functional` | True |
| `G_ceiling_clears_silence` | False |
| `G_ceiling_improves` | False |
| `G_teacher_abstains_on_tie` | False |

**ceiling BEFORE**

- n_items=36, reward=-0.192, addresses=36, slots=144
- families: `{'decidable': 15, 'clean': 12, 'tie': 9}`

| family | n | teacher_acc | teacher_abstain |
|---|---:|---:|---:|
| clean | 12 | 0.833 | 0.0 |
| decidable | 15 | 0.467 | 0.0 |
| tie | 9 | 0.0 | 0.0 |

**ceiling AFTER**

- n_items=1, reward=NaN, addresses=None, slots=None

| family | n | teacher_acc | teacher_abstain |
|---|---:|---:|---:|

**Top frames (by confirm):**

| frame | n | confirm | dispute | anchors | vpa | empty |
|---|---:|---:|---:|---:|---:|---|
| `the` | 6 | 0.83 | 0.00 | 1 | 1.00 | False |
| `the county seat` | 3 | 0.67 | 0.00 | 1 | 1.00 | False |
| `` | 373 | 0.57 | 0.29 | 55 | 2.95 | True |
| `title assigned bishop` | 3 | 0.00 | 0.67 | 1 | 3.00 | False |
| `title assigned archbishop` | 3 | 0.00 | 0.67 | 1 | 3.00 | False |
| `protector` | 4 | 0.00 | 0.75 | 1 | 4.00 | False |

### smoke and_ok thresholds

Источник: `stage281_decision_281_and_ok.json` · **`NO_FRAME_SURVIVES`** · smoke=True · frame_in_address=None

| gate | value |
|---|---|
| `G_frames_survive` | False |

**ceiling BEFORE:** нет данных

**ceiling AFTER:** нет данных

## Короткий вывод

- Базовые пороги: кадры не выживают (`NO_FRAME_SURVIVES`).
- Loose / FIA: 2 кадра на train, held-out после SKIP пустой → after **NaN** (`FRAMES_DO_NOT_HELP`).
- Night full: before **−0.544**, after **NaN** — тот же разрыв train→eval.
- Ни один прогон не открыл 280 (нужно after ≥ 0.375).
