# KEEP LEVERS — не выкидывать (гейт мог не пройти; сигнал был)

Standing rule (`_STATE_353.md` §0): *A LEVER THAT WORKS AND FALLS SHORT MAY BE CONTINUED.*

This list is for **long corpus / 7B+** scale. TinyStories-400 can starve context;
do not delete these interfaces because a Tiny gate failed.

Verified 2026-08-25 against `_STATE_353.md` and `results/_stage*.json`.

| рычаг | зачем на длинной | сверка |
|--|--|--|
| **614 CTX GO** | hop1 с другого extra той же рамки — это и есть контекст | **3/3 GO** после hide; CTX−RAND ≈ +.16…+.17 |
| **641 silent ROOM ≈ +.22** | места есть; picker нет — **наблюдение**, не лицензия Φ | silent room +.21…+.26; peak−rand ≈ 0 → STOP picker; room держать |
| **540 кумулятив** | сколько брать (budget/allow), не какой первый | cum растёт с rank depth; lever = allow, не reshuffle top1 |
| **534 extra ≈ .12 / 536rnd** | метка живая; заменить на роль, не убить teacher | 534 `extra` ≈ .12….14, gate false; 536rnd: teacher ≫ shuffle |
| **538 `key_seen=1`** | роль едет (band/peaked/width → rank), не слово | `key_seen=1.0` на 3/3; gate смешанный — держать **ключ роли** |
| **524 2/3 peaked** | мягкий pin: peaked hop ≠ tie | gate true 1337/2890, false 8642 — soft pin жив |
| **525 confirm** | тонкий на 400; **считать до n≥15** | peak_conf n=2/5/3 → VOID; идея confirm не закрыта |
| **517 / 628 W** | слот есть; exam был кривой | 517 = working window; 628 = VIA register (TRUE−SWAP меняет addr, reward нет) |
| **630 SEARCH/COMMIT/REFUSE** | алфавит действий; learner **только если** priced room > .05 на длинной | kill-switch на Tiny; алфавит не выкидывать |
| **440–450, 478, 481** | цепочка и online — не cloze | 440 compose GO; 478 online GO; 481 mark LIVE/DEAD live |
| **486 / 499 / 542** | hunt на сыром / free-swim / place curriculum | не cloze-i.i.d.; сырой поток и place-walk |

## Также держать (недавний надел 634–638)

| рычаг | зачем |
|--|--|
| **634 exact-place W** | DIRECT ∪ HOP1 без peak-сжатия — 3/3 GO |
| **638 row/context encoder** | веса учатся (`Δinit≈+.07…+.10`); алфавит = места; щель под VIA/атомы. Бар над PMI на Tiny не отменяет рычаг |
| **урок 639** | kind0 = PMI; не класть extract↔QUERY lift во вход residual |
| **hide_two / n_fr−2 / unique-max** | честность и refuse-on-tie |

## Потрачено as-is (не воскрешать формулу; урок оставить)

- 8 чисел `feat_place` (635–637)
- REFUSE-класс CE (635)
- peak-COMMIT сжатие (630 learner path)
- SHARE-VIA constraint (640 TRUE≈SWAP)
- 618-peak как picker на PMI-silent (641 silent gate)
- residual «догони PMI» без нового контраста (639)

## Экзамен на длинной (не Tiny-400)

1. Носитель: **634 W + роли 517/628 + 638 encoder** (без требования GO над PMI на stories).
2. Контекст: **TRUE−SWAP** при фиксированных QUERY/CURRENT, или 614-style other-extra hop.
3. Φ пишет SEARCH/COMMIT/REFUSE или constraint **только** если priced/TRUE−SWAP room > .05 на длинной.
4. GPT next-token / CE по словарю — другой экзамен; не замена этому списку.
