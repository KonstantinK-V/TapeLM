# SOTE as БДЛМ — product layer (dual-channel)

**Status:** agreed direction — **not** a live train gate in the 150–158 queue.  
**Depends on:** dual-channel contract — **CE trunk** (language) + **hops/memory API** (facts, continuity) out of LM train/gate.

### Placement in the program (locked wording)

1. **Product layer:** БДЛМ — следующий продуктовый слой на уже согласованном dual-channel.  
2. **Research branch:** отдельная ветвь изучения — **после потолка SOTE-as-LM** (когда язык/codebook/compose digs упрутся или дадут честный потолок vs BPE/GPT), не параллельно смазывать с hparam/morph очередью и не вместо неё.

**Does not replace:** Stages 150–158 (hparams / morph / compose) — those clean the *language indexing* / SOTE-as-LM lane first.


---

## One-line thesis

**БДЛМ = следующий продуктовый слой на уже согласованном dual-channel:**  
не «ещё один маленький GPT», а **управляемая структурная память фактов + говорящий языковой слой**.

---

## vs тренд (векторный RAG)

| Характеристика | Тренд (vector DB / RAG) | SOTE (потенциально / частично уже в стеке) |
|----------------|-------------------------|--------------------------------------------|
| Хранение | Текст в чанках + embedding | fp-последовательности + **структурные hops** (связи) |
| Поиск | Векторное сходство | Точный/ролевой поиск по цепочкам (hops) + опц. fp-близость |
| Память | Внешняя vector DB | Встроенная/управляемая (codebook, SoftPhraseMemory, hops) |
| Интерпретируемость | Чёрный ящик LLM | Прозрачнее: какие fp и какие связи использованы |
| Контроль | Низкий (галлюцинации) | Высокий: codebook, морфология, допустимые hops |
| Суть RAG-аналога | «Найди похожий кусок текста» | «Извлеки **факт** (hop) и встрой в **цепочку рассуждения**» |

Это ближе к **fact-RAG / hop-RAG / neuro-symbolic retrieval**, чем к pure similarity-RAG.

---

## Dual-channel contract (already agreed)

```
Train / LM gate:  single CE trunk (codebook = tokenizer / atoms)
Hops / memory:    separate API — facts, continuity, regulate — OUT of LM train/gate
```

Evidence against merging hops into next-token arbiter: Stages **133 / 134** (honest hop-rank / hops-as-assist on RIGHT — not a STORY unlock).  
CE-only policy: Stage **135** POLICY_OK.

Product BDLM uses the hop channel **as designed**, not as a soft bias inside CE.

---

## Pipeline sketch (product, after language digs)

1. **Retrieve** — hop / SoftPhraseMemory / bank fact (not chunk cosine alone).  
2. **Augment** — structured slots / chain (rel, right, polarity, multi-hop).  
3. **Generate** — CE trunk or external LLM conditioned on extracted structure; prefer **constrained** use of retrieved facts.  
4. **Audit** — log which fps and which hops were used (interpretability surface).

Language work in flight (**150–158**: matched rituals, morph-in-codebook, ComposeLayer) improves *how text is indexed into the codebook*.  
BDLM is *what you do with structural memory once language atoms are honest*.

---

## Chronology: where luck already was

**Yes — сильные результаты в «БДЛМ-области» были в основном до / вне ставки «SOTE = LM как GPT».**

| Era | What worked | Role |
|-----|-------------|------|
| Path / hops (V2 R7–R8, ~80–81) | hop2/3 floors; **HOPS_PRIMARY_FOR_RIGHT** | structural fact channel beats next-for-right |
| F85 + Stage97 | foundation encode + **clean hop2 joint ~95%** on frozen bank; dual-channel freeze | memory/hops as first-class |
| SoftPhraseMemory / binders | continuity & bind probes (when not asked to lift STORY LM) | managed memory |
| Stage88+ word-TF / TS scale / vs BPE | atom CE as mini-LM; STORY gap vs BPE; hparam mud | **SOTE-as-LM** competition — harder, different game |
| Stage120 mem→STORY | phrase-mem +0.5pp STORY | memory glued to LM gate ≈ null |
| Stage133–135 | hops out of LM; fact API separate | confirms split |

**Reading:** удача на **точных связях, encode, hops-primary, dual-channel freeze** — это ядро БДЛМ.  
Боль и «не догоняем GPT/BPE на STORY» — в основном от режима **SOTE как языковая модель next-token**, не от провала hops-as-DB.

**Program order (agreed):**
1. Дожать / честно измерить **SOTE-as-LM** (150–158 и родственные).  
2. При потолке LM-ветки — открыть **отдельную ветвь БДЛМ** (fact-RAG / hop retrieve → chain → controlled speak), не смешивая gates с CE.

Keep those lanes separate:  
- **150+** → честный язык / codebook / compose (LM lane).  
- **BDLM** → hop/fact retrieve → reason chain → controlled speak (**after LM ceiling**; product + research branch).

---

## Non-goals for this note

- Soft@5 as product claim.  
- Hops back into CE train gate.  
- Claiming vector-RAG parity on open-domain semantic search (fp ≠ SOTA text embeddings yet).  
- Replacing 150–158 queue with BDLM digs immediately.

---

## Pointers

- Dual-channel freeze: `results/SOTE_F85_DUAL_CHANNEL_BAND_FROZEN.txt`  
- Path replay: `results/sote_v2_path_replay.md`  
- Clean compare plan: `results/plan_150_plus_clean_compare.md`  
- Living fp contract: `results/fp_language_contract.md`
