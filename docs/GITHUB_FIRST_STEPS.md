# GitHub с нуля (для этого репозитория)

Кратко: **GitHub = облачная копия папки с кодом и текстами**, не вся твоя машина.  
Тяжёлые веса моделей и гигабайтные кэши **сюда не кладём** — они остаются локально в `checkpoints/` и части `data/`.

---

## 1. Что такое URL в `CITATION.cff`

Строка:

```yaml
repository-code: "https://github.com/KonstantinK-V/TapeLM"
```

Это **не** ссылка «на аккаунт вообще», а **на один репозиторий** (один проект).

| Часть | Пример | Значение |
|--------|--------|----------|
| `KonstantinK-V` | логин | Профиль: `https://github.com/KonstantinK-V` |
| `TapeLM` | имя репо | **Repository name** при создании |

Итоговая ссылка: **`https://github.com/KonstantinK-V/TapeLM`**

После создания репо на GitHub скопируй **зелёную** ссылку Clone → HTTPS и вставь в `CITATION.cff`.

---

## 2. Чекпоинты — грузятся ли на GitHub?

**Нет.** В `.gitignore` стоят:

- `checkpoints/` — вся папка
- `*.pt`, `*.pth` — веса где угодно

У тебя локально ~4 ГБ в `checkpoints/` — на GitHub **не попадут**, если не ломать `.gitignore`.

Посетитель клонирует код, потом **сам** кладёт веса по [`CHECKPOINTS.md`](CHECKPOINTS.md) (или обучает P1 через `_stage191_night.py`). Так делают почти все research-репо.

---

## 3. Какие папки **попадут** на GitHub (если сделать `git add .` и push)

Git загружает только то, что **не** в `.gitignore`. Для этого проекта ориентир такой:

| Папка / зона | На GitHub? | Зачем |
|--------------|------------|--------|
| **`artifact/`** | Да | Витрина для гостей, скрипты, JSON вердиктов |
| **`docs/`** | Да | ARCHITECTURE, CHECKPOINTS, PUBLISHING |
| **`results/`** | Да (без `_*_log.txt`, `_*_console.txt`, `*.mmap`) | План, препринт, `stage*_decision.json`, mini |
| **`legacy/sote/`** | Да | Старый SOTE (история) |
| **`README.md`, `LICENSE`, `CITATION.cff`, `.gitignore`** | Да | Лицо репо |
| **`_stage*.py` в корне** | Да | Скрипты стадий 170–212 |
| **`data/`** | **Частично** | Мелкие exam/jsonl — да; кэши OWT, wiki, tinystories — **нет** (ignore) |
| **`checkpoints/`** | **Нет** | Веса только локально |
| **`.venv/`, `__pycache__/`** | **Нет** | Окружение Python |

Перед первым push имеет смысл проверить размер (должны быть **сотни МБ**, не несколько ГБ):

```bash
git add -n .
```

Если в списке видишь `data/_owt_tokens_cache.txt` или `checkpoints/` — **не пушь**, допиши `.gitignore`.

---

## 4. Нумерация 170, 191, 212 — перенумеровывать?

**Не надо.** Это **ID экспериментов** в журнале, как номера коммитов в статье:

- **170–191** — curve / dual-channel / обучение P1  
- **192–198** — fingerprint stack (TapeLM)  
- **199–212** — семантика, RAG, falsify, frontier  

Все JSON уже называются `stage209_decision.json` и т.д. Смена номеров сломает воспроизводимость и текст в `plan_curve_dynamics.md`.

Для гостя:

- читать **`artifact/OVERVIEW.md`** и **`docs/STAGES.md`** — там смысл, не «шаг 1–2–3»;
- запускать **`196`** как главную сборку, **`204`/`205`** как яркие capability-оси.

Скрипты **172–175** в корне — ранние falsify; их можно со временем только **переложить** в `legacy/` (номера не менять).

---

## 5. Минимальные шаги на GitHub (один раз)

1. Зайти на [github.com](https://github.com), войти в аккаунт.  
2. **New repository** → имя **`TapeLM`** → Public → **без** README (он уже есть локально).  
3. На компе в папке проекта (PowerShell):

```powershell
cd C:\Users\Kostya\sote-letter-assembly
git add .
git status
git commit -m "Initial TapeLM artifact: docs, stages, results"
git branch -M main
git remote add origin https://github.com/KonstantinK-V/TapeLM.git
git push -u origin main
```

4. В `CITATION.cff` уже указан `https://github.com/KonstantinK-V/TapeLM` — сверь с URL репо после создания.  
5. На странице репо → **About** → вставить описание и topics из [`PUBLISHING.md`](PUBLISHING.md).

GitHub попросит логин; для push часто нужен **Personal Access Token** (Settings → Developer settings → Tokens), а не пароль от аккаунта.

---

## 6. Что увидит человек без твоих чекпоинтов

- Прочитает README и `artifact/` — **полная картина и цифры** из JSON/md.  
- Запустит `show_map.py` — таблица вердиктов.  
- `run_demo.py` скажет, что нет `stage191_p1_curve.pt` — это ожидаемо.

Если позже захочешь выложить веса отдельно — Hugging Face / Google Drive + ссылка в `CHECKPOINTS.md` (не обязательно для первой публикации).

---

## 7. Словарь в двух словах

| Термин | Смысл |
|--------|--------|
| **Репозиторий (repo)** | Одна папка-проект на GitHub |
| **Commit** | Снимок изменений локально |
| **Push** | Отправить commits на GitHub |
| **Clone** | Скачать репо к себе |
| **`.gitignore`** | Список «не заливать» |

Подробнее про описание и теги: [`PUBLISHING.md`](PUBLISHING.md).
