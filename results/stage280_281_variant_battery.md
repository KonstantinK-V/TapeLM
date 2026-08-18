# 280/281 variant battery

Smoke-only knobs. Logs: `results/_var_*.out`. Gate for 280 remains `teacher_ceiling ≥ 0.375`.

## 281 — frames / SKIP

| tag | overall | ceiling before → after | kept |
|---|---|---|---|
| `281_base` | `NO_FRAME_SURVIVES` | — | 0 / 392 |
| `281_loose` (confirm≥0.05, vpa≤5, anchors≥1) | `FRAMES_DO_NOT_HELP` | −0.192 → **NaN** | 2 frames / 9 asserts |
| `281_loose_fia` | `FRAMES_DO_NOT_HELP` | −0.192 → **NaN** | 2 / 9 |
| `281_and_ok` (confirm≥0.18, vpa≤3) | `NO_FRAME_SURVIVES` | — | 0 |

**Read:** ослабление порогов оставляет крохи (`the`, `the county seat`), на held-out SKIP снова обнуляет ленту. Пустой кадр (apposition) держит ~95% корпуса и правильно отфильтрован — но без него почти нечего оставлять. `frame-in-address` не меняет картину.

## 280 — teacher ceiling knobs

Baseline (earlier): ceiling **−0.189**, prec 0.34, rec 0.67, cands 6.67, hops 0.

| tag | overall | ceiling | prec | rec | cands | hops |
|---|---|---:|---:|---:|---:|---:|
| `280_kgap07` | `TEACHER_UNUSABLE_ON_RAW` | **−0.328** | 0.315 | 0.508 | 5.28 | 0 |
| `280_topk3` | `TEACHER_UNUSABLE_ON_RAW` | **−0.503** | 0.398 | 0.360 | 2.94 | 0 |
| `280_hopalways` (`--hop-min 99`) | `TEACHER_UNUSABLE_ON_RAW` | **−0.236** | 0.342 | 0.611 | 6.22 | **0.61** |
| `280_hopnone_kgap07` | `TEACHER_UNUSABLE_ON_RAW` | **−0.328** | 0.315 | 0.508 | 5.28 | 0 |

**Read:** жёстче резать список (gap/topk) **ухудшает** потолок — режутся свидетели адреса вместе с чужими. Принудительный hop стреляет, но ceiling не поднимает. Стена не в «хоп не вызвали» и не в «k слишком большой».

## Что остаётся как варианты (не прогнано здесь)

1. **Счёт большинства только по слотам того же адреса** (не score-gap) — прямо «источники расходятся vs говорят о разном».
2. **Другая нарезка кадра** — не bag-of-words между якорем и value; шаблон relation / POS / зависимость.
3. **SKIP на train+eval одним проходом** — сейчас кадры с train не живут на held-out lines.
4. **Мягкий match value** при dispute — меньше ложных «разных» из-за форм.

Machine JSON (partially polluted by leftover copies): `results/stage280_281_variant_battery.json`. Trust the table above (from raw logs).
