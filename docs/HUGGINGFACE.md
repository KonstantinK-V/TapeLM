# Hugging Face — выложить веса TapeLM

На GitHub лежит **код и тексты** (~28 MB). **Веса моделей** (~51 MB для главного набора) удобнее положить на **Hugging Face Hub** — бесплатно, с CDN, одной ссылкой в README.

## Что имеет смысл залить (минимум)

| Файл (локально) | Зачем |
|-----------------|--------|
| `checkpoints/stage191_p1_curve.pt` | Обязателен для 192–212, demo 196 |
| `checkpoints/stage191_p2_gpt.pt` | Контроль GPT для сравнений |
| `checkpoints/stage177_curve_bpe.pt` | *Опционально* (~4 MB), если кто-то гоняет 177 |

**Не заливай** сотни legacy `.pt` из `checkpoints/` (4+ GB) — они не нужны для TapeLM.

Токенizer `results/stage177_curve_bpe_tokenizer.json` уже в GitHub — на HF дублировать не обязательно.

---

## Шаг 1 — модель на HF

1. [huggingface.co/new](https://huggingface.co/new) → **Model**.
2. **Name:** например `TapeLM-P1` (полный id: `Kostya03v/TapeLM-P1`).
3. License: **MIT**.
4. Create.

## Шаг 2 — README (model card)

Скопируй содержимое из репозитория:

[`huggingface/TapeLM-P1/README.md`](../huggingface/TapeLM-P1/README.md)

→ в HF: вкладка **Model card** → Edit → вставить → Commit.

(Или залей этот файл через **Files** вместе с весами.)

## Шаг 3 — файлы

**Через сайт:** Model → **Files and versions** → **Upload files** → перетащи:

- `stage191_p1_curve.pt`
- `stage191_p2_gpt.pt`
- (опционально) `stage177_curve_bpe.pt`

**Через CLI** (один раз `pip install huggingface_hub`, логин `huggingface-cli login`):

```powershell
cd C:\Users\Kostya\sote-letter-assembly
huggingface-cli upload Kostya03v/TapeLM-P1 checkpoints/stage191_p1_curve.pt stage191_p1_curve.pt
huggingface-cli upload Kostya03v/TapeLM-P1 checkpoints/stage191_p2_gpt.pt stage191_p2_gpt.pt
```

Если репо на HF называешь иначе — замени `Kostya03v/TapeLM-P1` на свой `username/repo`.

## Шаг 4 — ссылка в GitHub

После загрузки добавь в [`CHECKPOINTS.md`](CHECKPOINTS.md) строку (и при желании commit + push):

```text
HF Hub: https://huggingface.co/Kostya03v/TapeLM-P1
```

Скрипт для посетителей: `python artifact/scripts/download_checkpoints.py` (качает в `checkpoints/`).

---

## Dataset? (обычно не нужно)

`data/stage191_docs.npz` и wiki-кэши **не** обязательны для scorecard 196; их можно пересобрать через `_stage191_night.py`. На HF dataset имеет смысл только если хочешь зафиксировать exam/jsonl — для старта **достаточно model repo с .pt**.

## Spaces (демо в браузере)

Stage 196 хочет **CUDA** — бесплатный HF Space с GPU ограничен. Для артефакта разумнее **Model + инструкция clone**, не Space.

---

## Чеклист

- [ ] Model `TapeLM-P1` создан
- [ ] README model card
- [ ] `stage191_p1_curve.pt` + `stage191_p2_gpt.pt` uploaded
- [ ] URL в `docs/CHECKPOINTS.md` + push GitHub
