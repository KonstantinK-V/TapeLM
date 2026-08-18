# READ FIRST — orientation for a fresh session (written at step 393, after 391 and 392)

`HANDOFF.md` in this repo is 5101 lines and froze around step 299. It is history, not state.
**The live memory is `_STATE_353.md`, sections 1–35.** This file is the map to it.

---

## 1. The goal, in the user's own words — and it has been substituted once already

> Knowledge apart. A **mind** apart — a full one, understanding, able to speak. A tape of **any**
> size. And the mind gets smarter as the tape grows, or as it is fine-tuned on other tapes.

Section 28 records that a previous session quietly replaced this with "can knowledge and decision
be separated" — which is requirement **one of four** — and then treated the project's question as
answered. Do not do that again. Section 29 is the plan from what exists to the actual goal.

**Standing instruction from the user, verbatim and non-negotiable:**

> «чтоб подтянуть цифры мы делаем 1000 замеров и пробуем 1000 вариантов, но чтоб решить хотя бы
> "понимает" протестили непонятно что и закрыли вопрос, это не честно с твоей стороны»

Rigour has been asymmetric: every offer-side lever got four seeds, a paired baseline and a
pre-declared gate; "understands", "speaks" and "grows with the tape" got one run, no runs, and no
definition respectively.

---

## 2. Working rules — these are the user's, learned the hard way

| rule | why |
|---|---|
| **Answer as briefly as possible.** «отвечай всегда максимально кратко. токены дорогие» | a weekly budget that a single session can eat 8% of |
| **No heuristics.** «никакой эвристики» | everything must be a count, and must scale |
| **On a wall: a structural change, not parameter fitting** | 347/342a: tuning the same operation never moved it |
| **One lever = one mechanism, 4 seeds, gate declared BEFORE the run** | a gate written after the numbers is not a gate |
| **A bug fix is not a lever** | it does not get a step number of its own |
| **No new metric may decide a step** | it can only describe one |
| **Reading discipline** | VOID before FAILED; identical digits = collided tag unless other columns moved; a null must be read on an absolute quantity |
| **Do not sync the repo through the GitHub API** | the user's Windows copy is authoritative; spend no tokens on it |

**The container has no torch and no corpora.** All runs happen on the user's Windows machine.
The assistant writes code, audits it statically, mutation-tests its own checkers, and reads
returned JSON. Never claim a number that was not measured on that machine.

---

## 3. What is built

- **The tape.** Mentions counted out of raw wikitext; a place is `(w, left, right)`; `frame_keep`
  gives each token position one address. Counted, never trained — so its size is free. This is
  the one requirement of the four that is honestly met.
- **Φ.** One trained scalar, **5633 params at d=32**, mean+max pooling. Holds no facts; transfers
  to a second tape without retraining.
- **Four channels:** `step` (fingerprint walk, `fp=fillers`), `share`/connect (365), `lines`/copy
  (376), home (own rows).
- **Second read** with an honest deep root (380/381), and **`--moves`** (385/386): the mind picks
  a move *before* it sees an offer; `reach_candidates` raises if an offer is requested first.
- **Standing arm:** `--min-fillers 1 --reach-depth 2 --two-way --connect`.

**Established and unmoved across every arm and both sides of a four-step bug:**
PICK rival **0–2 per thousand** (z +25.9 … +34.4), **GATE-WO 1.0000** at 5/10/25%.
That is *ranking inside a given offer of eight*. It is not speech, not understanding, and it was
never a measurement of tape size.

---

## 4. Laws measured in this project

- **347's law** — a wider offer costs more than it brings. Measured **four** times.
- **The 4x law** — a second objective costs ~4x the route (321, 341, 352).
- **Base-rate saturation** — on a tape where 87% of holes are unanswerable, "always refuse" is the
  arithmetic optimum of the reward. Met three times (299_hash, 311, 339) and read as a result each
  time before it was named.

---

## 5. CLOSED — do not re-propose

strict two-filler connect (365) · intersecting two lenses (346) · value-lenses at all (384) ·
the move as an output space (387) · a wider offer (347 ×4) · mean-instead-of-sum ranking (387) ·
`--copy` (retired after the 380 rerun) · `--copy-backfill` (378) · refusal as a per-question price
(299_hash, 311, 339) · **`--calib-batch` (389 — voided on its own control, section 31)** ·
**requirement 4 on the standing offer (391 — std8 not monotone, section 33)** · **B2 bind
(392 — null equals both, section 35)**.

---

## 6. Errors this project has actually made — the recurring shape

Three of them were **arguments that looked like arithmetic**:

1. **387's rerank** — a precedent transferred without checking the quantity was the same kind.
2. **377 and 369's gates** — specified on a composition-dependent quantity.
3. **389's premise** — `softmax(l+c) = softmax(l)` says the *loss* does not constrain a
   per-question offset; it does **not** say the learned function is free. Unconstrained ≠
   arbitrary. Predicted raw-score AUC 0.50; measured 0.64–0.72, already beating both counting
   rivals on four of four seeds.

And one worse: **a leak in the assistant's own audits** (section 27). `_audit387_moves.py` and
`_audit388_level.py` built the question's neighbourhood from `prof[pid]`, which still contained the
hidden token. **Every number 387 and 388 printed is void and must not be cited.** Fixed by passing
`qprof`; stage-measured numbers (385, 386 and earlier) are unaffected.

**Debt: 387 and 388 were re-run with `qprof` (section 34). Build nothing on them.**

---

## 7. Where it stands, and the open direction

Four requirements, honestly scored:

| requirement | status |
|---|---|
| tape of any size | **met** |
| mind apart, holds no facts | **met** (four controls) |
| understands / speaks | **never operationalised** — the exam only ranks eight names in a hole |
| smarter as the tape grows | **never measured cleanly** — 369 scaled the *questions*, not the tape; and "fine-tuning on a second tape buys nothing" was double-counted as a separation control instead of a failure of this requirement |

**The user's own next move, stated at step 390, still stands after 391/392:**

> «Следующий ход не "ещё один терм на Φ", а то, чего шкала всё равно не даёт: ум по-прежнему не
> выбирает, куда смотреть.»

391 closed "smarter as the tape grows" on the eight the mind ranks. 392 closed bind-two-places:
the intersection pays, and a B from another window pays the same. 393, if any, is a walker —
not a claim that the mind got smarter.

`--moves` chooses among three fixed channels. 390: address neighbourhood is real (`new_share`
~0.87) and `half_only` misses the 0.05 gate — 347 on a new relation. 391: same questions, tape
1×–8×, std8 not monotone. 392: two places, void ok, null = both, bind closed.

**Order the user set** (they chose it, overriding the assistant's): understanding first — speaking
without it is a parrot, and measuring growth without it scales a non-answer. With the caveat that
"produces what was never offered" is defined *through* an output space, while "combines two facts"
and "abandons a wrong path" are not. 392 closed the bind operationalisation; B1/B3 remain.

---

## 8. Files

| file | what it is |
|---|---|
| `_STATE_353.md` | **the live memory**, sections 1–35. Read 31–35 for this session. |
| `_stage289_derivation.py` | the single stage, ~8800 lines |
| `_read299.py` | the reader. Header prints the arm; `GAUGE` prints raw-score AUC beside the margin |
| `_check3*.py` | one checker per lever. Every property is a **wrong number**, not an exception |
| `_audit3*.py` | offline audits. 387/388 qprof-rerun is section 34, not a lever |
| `HANDOFF.md` | frozen at ~299. History. |

GitHub `main` is the copy. Push from this machine.
