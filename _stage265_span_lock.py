"""
Stage 265 — Span-lock: the gate decides WHERE a value starts, the tape decides HOW it is spelled.

The 256 decode audit isolated the failure. At every one of the five mechanism misses the copy
channel held the correct next token at rank 1 with p_copy ~= 0.997, and the gate closed anyway:

    Markbreit t=3  copy_rank_gold=1  p_copy=0.997  gate=0.00025  needed "it"  emitted "ch"
    Diavolo   t=3  copy_rank_gold=1  p_copy=0.997  gate=0.0017   needed "ol"  emitted "l"
    Cheese    t=2  copy_rank_gold=1  p_copy=0.997  gate=0.0027   needed "ese" emitted "f"
    Densetsu  t=3  copy_rank_gold=1  p_copy=0.997  gate=0.12     needed "u"   emitted "h"
    Sphinx    t=2  copy_rank_gold=1  p_copy=0.997  gate=0.13     needed "in"  emitted "r"

On each of those steps the emitted token is the LM's own top-1. The gate takes the entropy of the
base logits as an input feature, so it learned "the LM is confident -> hand over the wheel". Outside
a value that is reasonable. Inside one it is always wrong. The same mechanism explains the
--random-values control coming out HIGHER (0.875 vs 0.75): on nonsense strings the LM is never
confident, so the gate never closes.

Deleting the entropy feature would not fix it -- h_t is also an input and the same signal is
recoverable from it. The structural fix is to stop asking the gate that question at all:

    soft   (256):  the value survives only if the gate holds on EVERY token   P ~ p^N
    locked (265):  the gate opens once, then tape.tok_ids is emitted verbatim  P ~ p

That is the scaling argument, and the 256 data already shows the exponent: the five mechanism
misses sit on 5/4/4/4/3-token values while the successes cluster at 2-3. Longer values are exactly
what a real tape holds -- dates, identifiers, names outside English, anything not in BPE's comfort
zone -- so the compounding penalty is a ceiling that scale does not lift.

Span-lock also makes the contract checkable rather than statistical: the emitted span is bit-identical
to the slot, so the weights' contribution to a value is exactly zero, by construction and assertable.
It gets the restart defect (19/24 decodes in the audit re-emitted the value's first token after the
value ended) for free, because the span length comes from the tape instead of a learned stop.

Three arms, so the claim is attributable:

    A  soft train  + soft decode    reproduction of 256          (validity gate)
    B  soft train  + locked decode  does the lock alone fix it   (no retraining at all)
    C  open train  + locked decode  gate trained only on "open"  (full proposal)

Arm C trains the gate ONLY at the value-start step; inside the span g is pinned to 1 with no
gradient, so the entropy shortcut never gets a training signal. 257's lesson is respected --
training stays a soft mixture everywhere else, since hard-commit-only training collapses the
stop gate.

Controls: head-only, shuffled keys, empty tape, per-fact slot delete, prose CE on/off, prose gate,
and a paired nonsense-value exam in the SAME tape -- under the lock, EM must stop caring whether
the value is a dictionary word.

  python _stage265_span_lock.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage24x_lib as L
import _stage251_cpc_understand as s251
import _stage252_joint_cpc as s252
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank

from _inprint_glue import (
    ANCHOR_RE,
    DEFAULT_CUE,
    DEFAULT_FACT_TMPL,
    SlotBias,
    TapeView,
    copy_dist,
    ctx_query,
    hidden_and_logits,
    mix_logprob,
    raw_query,
)

RES = Path("results")
DECISION = RES / "stage265_decision.json"
MINI = RES / "stage265_mini.md"
LOG = RES / "_stage265_log.txt"
TRACE = RES / "stage265_span_trace.json"
DECISION_256 = RES / "stage256_decision.json"
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_JOINT = Path("checkpoints/stage253_joint_l02.pt")
CKPT_OUT = Path("checkpoints/stage265_span_lock.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 265

CUE = DEFAULT_CUE
FACT_TMPL = DEFAULT_FACT_TMPL
PLACEHOLDER = "The chronicle continues with routine administrative detail ."


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def fp_version() -> str:
    fn = getattr(L, "canonical_fp_version", None)
    if callable(fn):
        try:
            return str(fn())
        except Exception:
            pass
    return CKPT_P1.name


# --------------------------------------------------------------------------------------
# span bookkeeping
# --------------------------------------------------------------------------------------
def match_step(prefix_ids: list[int], ids: list[int]) -> int:
    """How many tokens of `ids` the prefix already ends with. Mirrors copy_dist exactly."""
    for cut in range(min(len(ids), len(prefix_ids)), 0, -1):
        if prefix_ids[-cut:] == ids[:cut]:
            return cut
    return 0


def span_candidates(glue, tape: TapeView, sims, idx, prefix_ids) -> list[tuple[int, float]]:
    """Retrieved slots whose value has not started yet -> (slot, copy weight).

    A span may only OPEN at step 0 of a value. Slots already under way are excluded because the
    lock, not the gate, is what carries a value that is mid-emission.
    """
    w = glue.weights(sims)
    out = []
    for pos, j in enumerate(idx.tolist()):
        ids = tape.tok_ids[j]
        if not ids:
            continue
        if match_step(prefix_ids, ids) == 0:
            out.append((j, float(w[pos])))
    return out


@torch.no_grad()
def decode(
    glue,
    model,
    char_table,
    tok,
    bank,
    tape: TapeView,
    fact,
    pad_id,
    V,
    device,
    k: int,
    max_new: int,
    *,
    locked: bool,
    use_glue: bool = True,
    open_thresh: float = 0.5,
    reopen_margin: float = 0.1,
    max_opens: int = 1,
    no_repeat: bool = True,
    trace: list | None = None,
):
    """Greedy free-form continuation of the cue. `locked=False` is 256's decode verbatim.

    Under the lock the gate is consulted only while no span is open. Once it opens on a slot the
    slot's tokens are emitted verbatim; the LM cannot overwrite them and no learned stop is needed,
    because the length is the slot's length.

    Hysteresis: after a span closes the lock is disarmed and cannot fire again until the gate has
    fallen below `open_thresh - reopen_margin` for at least one step. Without it the smoke run
    re-opened on a *different* slot as soon as the value ended and there was room left (restart
    0.375) -- 256's restart defect wearing a new coat: the gate is still high because the query has
    not moved, so `no_repeat` on the value string does not catch it.

    Planted-fact exam default `max_opens=1`: after the first open the lock stays disarmed for the
    rest of the decode. That is a property of this exam ("one answer per cue"), not of the
    architecture. Multi-slot answers need `max_opens=0` (unlimited) with hysteresis only.

    Two EM readings under the lock (do not collapse them):
      em_span  — first-word match on the emitted tape span alone (stop-at-boundary for scoring)
      em_text  — first-word match on the full max_new continuation (LM may BPE-glue onto the value)
    Headline ``em`` = em_span. em_text keeps the end-of-value defect visible; the boundary is NOT
    solved — only deferred by the one-answer exam. Same family as restart: the model does not know
    the value has ended.
    """
    cue_ids = [i for i in tok.encode(CUE.format(S=fact["S"])).ids if i != pad_id]
    seq = list(cue_ids)
    gen: list[int] = []
    gates: list[float] = []
    opens: list[dict] = []
    emitted_values: set[str] = set()
    armed = True
    span_end: int | None = None  # gen length right after a locked span emit

    while len(gen) < max_new:
        ids = torch.tensor([seq[-MAX_ARCS:]], dtype=torch.long, device=device)
        h, logits = hidden_and_logits(model, char_table, ids, pad_id)
        base = logits[0, -1]
        score = torch.log(F.softmax(base, -1) + 1e-9)

        if use_glue:
            q = ctx_query(glue, bank, tok, seq, anchor_ids=cue_ids)
            hit = tape.topk(q, k) if q is not None else None
            if hit is not None:
                sims, idx = hit
                ent = float(-(F.softmax(base, -1) * F.log_softmax(base, -1)).sum())
                p_copy, cov = copy_dist(glue, tape, sims, idx, seq, V, device)
                g_val = glue.g(h[0, -1], float(sims.max()), float(sims.mean()), ent, cov)
                gates.append(float(g_val))
                if max_opens <= 0 or len(opens) < max_opens:
                    if float(g_val) < open_thresh - reopen_margin:
                        armed = True
                if locked and armed and float(g_val) >= open_thresh and (
                    max_opens <= 0 or len(opens) < max_opens
                ):
                    cands = [
                        c
                        for c in span_candidates(glue, tape, sims, idx, seq)
                        if not (no_repeat and tape.values[c[0]] in emitted_values)
                    ]
                    if cands:
                        j = max(cands, key=lambda c: c[1])[0]
                        span = list(tape.tok_ids[j])
                        room = max_new - len(gen)
                        take = span[:room]
                        opens.append(
                            {
                                "at": len(gen),
                                "slot": j,
                                "value": tape.values[j],
                                "g": float(g_val),
                                "n_tok": len(span),
                                "truncated": len(take) < len(span),
                            }
                        )
                        emitted_values.add(tape.values[j])
                        gen.extend(take)  # the tape writes the whole span; nothing else is consulted
                        seq.extend(take)
                        armed = False
                        if span_end is None:
                            span_end = len(gen)
                        continue
                score = mix_logprob(base, g_val, p_copy, cov)

        nxt = int(score.argmax())
        gen.append(nxt)
        seq.append(nxt)

    text_full = tok.decode(gen).strip()
    text_span = tok.decode(gen[:span_end]).strip() if span_end is not None else text_full
    gold_ids = None
    for j, v in enumerate(tape.values):
        if v == fact["value"]:
            gold_ids = tape.tok_ids[j]
            break
    first = opens[0] if opens else None
    span_txt = tok.decode(gold_ids).strip() if gold_ids else None
    em_first_full = text_full.strip().split(" ")[0].strip(" .,;:") if text_full else ""
    em_first_span = text_span.strip().split(" ")[0].strip(" .,;:") if text_span else ""
    row = {
        "S": fact["S"],
        "gold": fact["value"],
        "got": text_span if (locked and span_end is not None) else text_full,
        "got_span": text_span,
        "got_text": text_full,
        "em_first_word": em_first_span if (locked and span_end is not None) else em_first_full,
        "em_first_word_text": em_first_full,
        "gold_decode": span_txt,
        "gate_mean": float(np.mean(gates)) if gates else float("nan"),
        "n_opens": len(opens),
        "opened_value": first["value"] if first else None,
        "opened_at": first["at"] if first else None,
        "opened_correct": bool(first and first["value"] == fact["value"] and first["at"] == 0),
        "truncated": bool(first and first["truncated"]),
        "verbatim": bool(
            gold_ids is not None
            and first is not None
            and first["value"] == fact["value"]
            and not first["truncated"]
            and gen[first["at"] : first["at"] + len(gold_ids)] == gold_ids
        ),
        "n_val_tokens": len(gold_ids) if gold_ids else None,
        "opens": opens,
        "span_end": span_end,
    }
    if trace is not None and len(trace) < 48:
        trace.append(row)
    return row


def exact_match(text: str, value: str) -> bool:
    return text.strip().split(" ")[0].strip(" .,;:") == value if text else False


def em_window3(text: str, value: str) -> bool:
    if not text:
        return False
    return value in [w.strip(" .,;:'") for w in text.strip().split(" ")[:3]]


@torch.no_grad()
def exam(
    glue, model, char_table, tok, bank, tape, facts, pad_id, V, device, k, max_new, *, locked, use_glue=True,
    open_thresh=0.5, reopen_margin=0.1, max_opens=1, trace=None,
):
    rows = [
        decode(
            glue, model, char_table, tok, bank, tape, f, pad_id, V, device, k, max_new,
            locked=locked, use_glue=use_glue, open_thresh=open_thresh,
            reopen_margin=reopen_margin, max_opens=max_opens, trace=trace,
        )
        for f in facts
    ]
    for r, f in zip(rows, facts):
        # em_span: score the tape span alone (headline under lock). em_text: full continuation —
        # keeps Whammy+n / Rascal+ibur visible when the LM glues onto the value after the span.
        r["em_span"] = exact_match(r.get("got_span") or r["got"], f["value"])
        r["em_text"] = exact_match(r.get("got_text") or r["got"], f["value"])
        r["em"] = r["em_span"] if locked else r["em_text"]
        r["em3"] = em_window3(r.get("got_span") or r["got"], f["value"]) if locked else em_window3(
            r["got"], f["value"]
        )
        r["glue_bpe"] = bool(r["em_span"] and not r["em_text"])  # span ok, text glued
    n = max(1, len(rows))
    gs = [r["gate_mean"] for r in rows if not math.isnan(r["gate_mean"])]
    out = {
        "em": sum(r["em"] for r in rows) / n,
        "em_span": sum(r["em_span"] for r in rows) / n,
        "em_text": sum(r["em_text"] for r in rows) / n,
        "em3": sum(r["em3"] for r in rows) / n,
        "glue_bpe_rate": sum(r["glue_bpe"] for r in rows) / n,
        "gate_mean": float(np.mean(gs)) if gs else float("nan"),
        "rows": rows,
    }
    # Span statistics exist only under the lock. Reporting 0.0 for the soft arm would read as
    # "soft never restarts", when in fact it never opens a span at all and the metric is undefined.
    span = {
        "verbatim": sum(r["verbatim"] for r in rows) / n,
        "open_recall": sum(r["opened_correct"] for r in rows) / n,
        "open_precision": (
            sum(r["opened_correct"] for r in rows) / max(1, sum(r["n_opens"] > 0 for r in rows))
        ),
        "restart_rate": sum(r["n_opens"] > 1 for r in rows) / n,
        "truncated_rate": sum(r["truncated"] for r in rows) / n,
    }
    out.update(span if locked else dict.fromkeys(span, None))
    return out


def em_by_length(rows: list[dict], cut: int = 4) -> dict:
    """The scaling claim, measured: does EM fall off as the value gets longer?

    Under a soft mixture a value survives only if the gate holds on every token, so EM should decay
    with length. Under the lock the gate is asked once, so length must stop mattering.
    """
    short = [r for r in rows if r["n_val_tokens"] and r["n_val_tokens"] < cut]
    long_ = [r for r in rows if r["n_val_tokens"] and r["n_val_tokens"] >= cut]
    per = Counter()
    tot = Counter()
    for r in rows:
        if r["n_val_tokens"]:
            tot[r["n_val_tokens"]] += 1
            per[r["n_val_tokens"]] += int(r["em"])
    return {
        "cut": cut,
        "n_short": len(short),
        "n_long": len(long_),
        "em_short": float(np.mean([r["em"] for r in short])) if short else float("nan"),
        "em_long": float(np.mean([r["em"] for r in long_])) if long_ else float("nan"),
        "by_n_tokens": {str(n): per[n] / tot[n] for n in sorted(tot)},
    }


# --------------------------------------------------------------------------------------
# training (256's recipe; --train-mode open removes the in-span gradient on the gate)
# --------------------------------------------------------------------------------------
def nce_loss(glue: SlotBias, raw_q: torch.Tensor, gold_mask: torch.Tensor, K: torch.Tensor, tau: float):
    q = F.normalize(glue.W_q(raw_q), dim=-1)
    sims = (q @ K.t()) / tau
    pos = torch.where(gold_mask, sims, torch.full_like(sims, -1e4)).logsumexp(dim=-1)
    return (sims.logsumexp(dim=-1) - pos).mean()


def fact_batch(glue, model, char_table, tok, bank, tape, facts, pad_id, V, device, k, *, open_only: bool):
    """Teacher-forced CE on the value tokens.

    With `open_only`, steps inside the value use g=1 as a constant: the copy path still gets
    gradient through tau and W_q, but the gate receives none. That is the exact signal that taught
    it to defer to a confident LM mid-word, and it is the only thing removed here.
    """
    losses, gates = [], []
    for f in facts:
        cue_ids = [i for i in tok.encode(CUE.format(S=f["S"])).ids if i != pad_id]
        val_ids = [i for i in tok.encode(" " + f["value"]).ids if i != pad_id]
        if not cue_ids or not val_ids:
            continue
        seq = (cue_ids + val_ids)[-MAX_ARCS:]
        n_ctx = len(seq) - len(val_ids)
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        h, logits = hidden_and_logits(model, char_table, ids, pad_id)
        for step, tid in enumerate(val_ids):
            t = n_ctx + step - 1
            if t < 0 or t >= logits.size(1):
                break
            prefix = seq[: t + 1]
            base = logits[0, t]
            q = ctx_query(glue, bank, tok, prefix, anchor_ids=cue_ids)
            hit = tape.topk(q, k) if q is not None else None
            if hit is None:
                losses.append(-torch.log(F.softmax(base, -1) + 1e-9)[tid])
                continue
            sims, idx = hit
            ent = float(-(F.softmax(base, -1) * F.log_softmax(base, -1)).sum())
            p_copy, cov = copy_dist(glue, tape, sims, idx, prefix, V, device)
            g_val = glue.g(h[0, t], float(sims.max()), float(sims.mean()), ent, cov)
            if open_only and step > 0:
                g_use = torch.ones((), device=device, dtype=g_val.dtype)
            else:
                g_use = g_val
                gates.append(float(g_val))
            losses.append(-mix_logprob(base, g_use, p_copy, cov)[tid])
    if not losses:
        return None, float("nan")
    return torch.stack(losses).mean(), (float(np.mean(gates)) if gates else float("nan"))


def prose_batch(glue, model, char_table, tok, bank, tape, ids, pad_id, V, device, k, gate_l1, use_glue=True):
    h, logits = hidden_and_logits(model, char_table, ids, pad_id)
    losses, gates = [], []
    seq = ids[0].tolist()
    valid = [t for t in range(len(seq) - 1) if seq[t] != pad_id and seq[t + 1] != pad_id]
    if not valid:
        return None, float("nan")
    for t in valid[:: max(1, len(valid) // 8)]:
        base = logits[0, t]
        prefix = seq[: t + 1]
        if not use_glue:
            losses.append(-torch.log(F.softmax(base, -1) + 1e-9)[seq[t + 1]])
            gates.append(0.0)
            continue
        q = ctx_query(glue, bank, tok, prefix)
        hit = tape.topk(q, k) if q is not None else None
        if hit is None:
            losses.append(-torch.log(F.softmax(base, -1) + 1e-9)[seq[t + 1]])
            gates.append(0.0)
            continue
        sims, idx = hit
        ent = float(-(F.softmax(base, -1) * F.log_softmax(base, -1)).sum())
        p_copy, cov = copy_dist(glue, tape, sims, idx, prefix, V, device)
        g_val = glue.g(h[0, t], float(sims.max()), float(sims.mean()), ent, cov)
        losses.append(-mix_logprob(base, g_val, p_copy, cov)[seq[t + 1]] + gate_l1 * g_val)
        gates.append(float(g_val))
    if not losses:
        return None, float("nan")
    return torch.stack(losses).mean(), float(np.mean(gates))


def train_glue(
    glue, model, char_table, tok, bank, tape, fit_facts, flat, off, train_docs, pad_id, V, device,
    *, steps, k, gate_l1, nce_q, nce_slot, nce_w, nce_tau, open_only, rng, t0, tag,
):
    opt = torch.optim.AdamW(glue.trainable(), lr=3e-3, weight_decay=0.01)
    K_all = tape.K.float()
    curve = []
    for step in range(1, steps + 1):
        batch = [fit_facts[rng.randrange(len(fit_facts))] for _ in range(min(4, len(fit_facts)))]
        l_fact, g_fact = fact_batch(
            glue, model, char_table, tok, bank, tape, batch, pad_id, V, device, k, open_only=open_only
        )
        ids = s251.sample_windows_docs(flat, off, 1, rng, pad_id, train_docs).to(device)
        l_prose, g_prose = prose_batch(
            glue, model, char_table, tok, bank, tape, ids, pad_id, V, device, k, gate_l1
        )
        l_nce = None
        if nce_q is not None and nce_w > 0:
            sel = torch.randint(0, nce_q.size(0), (min(64, nce_q.size(0)),), device=device)
            gold = F.one_hot(nce_slot[sel], K_all.size(0)).bool()
            l_nce = nce_w * nce_loss(glue, nce_q[sel], gold, K_all, nce_tau)
        parts = [x for x in (l_fact, l_prose, l_nce) if x is not None]
        if not parts:
            continue
        loss = parts[0]
        for p in parts[1:]:
            loss = loss + p
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(glue.trainable(), 1.0)
        opt.step()
        if step % max(1, steps // 5) == 0:
            curve.append(
                {
                    "step": step,
                    "loss_fact": float(l_fact) if l_fact is not None else None,
                    "loss_prose": float(l_prose) if l_prose is not None else None,
                    "loss_nce": float(l_nce) if l_nce is not None else None,
                    "gate_fact": g_fact,
                    "gate_prose": g_prose,
                }
            )
            log(
                f"  [{tag}] step {step}/{steps} fact={float(l_fact) if l_fact is not None else float('nan'):.3f} "
                f"prose={float(l_prose) if l_prose is not None else float('nan'):.3f} "
                f"g_fact={g_fact:.3f} g_prose={g_prose:.3f} ({time.time()-t0:.0f}s)"
            )
    glue.eval()
    return curve


@torch.no_grad()
def full_bank_read(glue, bank, tok, tape: TapeView, facts, pad_id) -> dict:
    """Rank of the gold slot over the WHOLE bank at the cue -- no candidate pool anywhere.

    The closed-pool headline and this number are different questions; 256's exam is saturated here
    (top1 = 1.0), which is precisely why its EM was measuring decode and nothing else.
    """
    ranks = []
    for f in facts:
        cue_ids = [i for i in tok.encode(CUE.format(S=f["S"])).ids if i != pad_id]
        q = ctx_query(glue, bank, tok, cue_ids, anchor_ids=cue_ids)
        if q is None:
            continue
        sims = tape.K @ q
        gold = [j for j, v in enumerate(tape.values) if v == f["value"]]
        if not gold:
            continue
        gsim = float(sims[gold].max())
        ranks.append(1 + int((sims > gsim).sum()))
    if not ranks:
        return {"bank_size": len(tape.values), "n": 0}
    return {
        "bank_size": len(tape.values),
        "n": len(ranks),
        "top1": float(np.mean([r == 1 for r in ranks])),
        "hit10": float(np.mean([r <= 10 for r in ranks])),
        "mrr": float(np.mean([1.0 / r for r in ranks])),
        "median_rank": float(np.median(ranks)),
    }


def published_em_256() -> float | None:
    if not DECISION_256.is_file():
        return None
    try:
        d = json.loads(DECISION_256.read_text(encoding="utf-8"))
        return float(d["summary"]["em_glue"])
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--gate-l1", type=float, default=0.02)
    ap.add_argument("--nce-w", type=float, default=1.0)
    ap.add_argument("--nce-tau", type=float, default=0.05)
    ap.add_argument("--facts", type=int, default=0)
    ap.add_argument("--nonsense-facts", type=int, default=0, help="held-out facts whose value is not a word")
    ap.add_argument("--distractor-slots", type=int, default=0)
    ap.add_argument("--open-thresh", type=float, default=0.5, help="gate level that opens a span")
    ap.add_argument(
        "--reopen-margin",
        type=float,
        default=0.1,
        help="after a span, the gate must fall below open_thresh - margin before it may fire again",
    )
    ap.add_argument(
        "--max-opens",
        type=int,
        default=1,
        help="max spans per decode; 1 = exam 'one answer per cue' (not architecture); 0 = unlimited",
    )
    ap.add_argument("--no-arm-c", action="store_true", help="skip the open-trained arm (half the wall time)")
    args = ap.parse_args()

    LOG.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    steps = args.steps or (200 if args.smoke else 800)
    n_facts = args.facts or (8 if args.smoke else 48)
    n_nonsense = args.nonsense_facts or (4 if args.smoke else 16)
    n_dist = args.distractor_slots or (150 if args.smoke else 1200)
    max_new = 6 if args.smoke else 12  # must exceed the longest value; a 9-token value truncated at 8
    n_hold = 4 if args.smoke else 12
    max_lines = 400 if args.smoke else 6000
    k = args.topk

    log(
        f"Stage265 span-lock start {datetime.now(timezone.utc).isoformat()} device={device} "
        f"steps={steps} facts={n_facts} nonsense={n_nonsense} distractors={n_dist} "
        f"topk={k} open_thresh={args.open_thresh}"
    )

    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)

    trunk_ckpt = CKPT_JOINT if CKPT_JOINT.exists() else CKPT_P1
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(trunk_ckpt, map_location=device, weights_only=False)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    model_can = SelfModelXL(n_char, V).to(device)
    model_can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model_can.eval()
    for p in model_can.parameters():
        p.requires_grad_(False)
    bank_can = FpBank(model_can, stoi, device)
    log(f"  trunk={trunk_ckpt.name} (frozen) fp_version={fp_version()}")

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(1_000_000 if args.smoke else 6_000_000)
    values_pool = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5))
    rng.shuffle(values_pool)
    lines = [l.strip() for l in wtext.split("\n") if len(l.strip()) >= 60][:max_lines]

    # ---- facts. Wiki-valued half fits the glue, half is held out (255: never score what you fit).
    # The nonsense-valued facts are ALWAYS held out: they are the paired control for the claim that
    # the lock makes EM independent of whether the LM can spell the value.
    subs = [w for w in gen_fakes(set(values_pool), rng, n_facts + n_nonsense + 60) if len(w) >= 5]
    subs = list(dict.fromkeys(subs))
    fake_vals = [w for w in gen_fakes(set(values_pool) | set(subs), rng, n_nonsense + 40) if len(w) >= 6]
    fake_vals = [w for w in dict.fromkeys(fake_vals) if w not in subs][:n_nonsense]
    if len(subs) < n_facts + len(fake_vals):
        raise SystemExit(f"not enough distinct subjects: {len(subs)} < {n_facts + len(fake_vals)}")

    facts = []
    for i in range(n_facts):
        facts.append(
            {
                "S": subs[i],
                "value": values_pool[i],
                "sent": FACT_TMPL.format(S=subs[i], V=values_pool[i]),
                "glue_train": i % 2 == 0,
                "kind": "wiki",
            }
        )
    for j, fv in enumerate(fake_vals):
        S = subs[n_facts + j]
        facts.append(
            {"S": S, "value": fv, "sent": FACT_TMPL.format(S=S, V=fv), "glue_train": False, "kind": "nonsense"}
        )

    fit_facts = [f for f in facts if f["glue_train"]]
    eval_wiki = [f for f in facts if not f["glue_train"] and f["kind"] == "wiki"]
    eval_non = [f for f in facts if f["kind"] == "nonsense"]
    eval_facts = eval_wiki + eval_non
    log(f"  facts: fit={len(fit_facts)} held_out_wiki={len(eval_wiki)} held_out_nonsense={len(eval_non)}")

    # ---- tape: canonical frozen keys, values never enter the CE text ----
    keys, vals = [], []
    pair_q, pair_slot = [], []
    for f in facts:
        kf = bank_can.fp([f["S"]])[0]
        c = bank_can.ctx_fp(f["sent"], exclude=f["value"])
        keys.append(F.normalize(kf + c, dim=-1) if c is not None else kf)
        vals.append(f["value"])
    used = set(vals)
    for ln in lines:
        if len(vals) >= len(facts) + n_dist:
            break
        for m in ENT_RE.finditer(ln):
            ent = m.group(1)
            if len(ent) < 5 or ent in used:
                continue
            lo, hi = max(0, m.start() - 120), min(len(ln), m.end() + 120)
            c = bank_can.ctx_fp(ln[lo:hi], exclude=ent)
            if c is None:
                continue
            anchors = [w for w in ANCHOR_RE.findall(ln[lo : m.start()]) if w != ent]
            if not anchors:
                continue
            keys.append(F.normalize(bank_can.fp([anchors[-1]])[0] + c, dim=-1))
            cq = bank_can.ctx_fp(ln[lo : m.start()])
            if cq is not None:
                pair_q.append(F.normalize(bank_can.fp([anchors[-1]])[0] + cq, dim=-1))
                pair_slot.append(len(vals))
            vals.append(ent)
            used.add(ent)
            if len(vals) >= len(facts) + n_dist:
                break
    tape = TapeView(torch.stack(keys, 0).to(device), vals, tok, pad_id)
    log(f"  tape slots={len(vals)} ({len(facts)} planted + {len(vals)-len(facts)} wiki noise)")

    tlen = Counter(len(tape.tok_ids[vals.index(f['value'])]) for f in eval_facts)
    log(f"  held-out value lengths (BPE tokens): {dict(sorted(tlen.items()))}")

    prose = "\n".join(lines + [PLACEHOLDER] * min(len(facts), len(lines) // 4))
    flat, off = s213.build_flat_from_text(prose, tok, pad_id, max_lines=max_lines + 64, min_line_len=20)
    n_docs = len(off) - 1
    hold_docs = list(range(max(1, n_docs - max(2, n_docs // 20)), n_docs))
    train_docs = list(range(0, hold_docs[0]))
    hold_batches = s252.make_hold_batches(flat, off, hold_docs, pad_id, n_hold, SEED + 5)

    nce_q = torch.stack(pair_q).to(device).float() if pair_q else None
    nce_slot = torch.tensor(pair_slot, device=device) if pair_slot else None
    log(f"  W_q training pairs={0 if nce_q is None else nce_q.size(0)}")

    d_hidden = 2 * (model.head.in_features // 2)
    base_hold = s252.fixed_hold_ce(model, hold_batches, char_table, pad_id, device)

    def run_exam(glue, tp, facts_, locked, trace=None):
        return exam(
            glue, model, char_table, tok, bank_can, tp, facts_, pad_id, V, device, k, max_new,
            locked=locked, open_thresh=args.open_thresh, reopen_margin=args.reopen_margin,
            max_opens=args.max_opens, trace=trace,
        )

    # ---- soft-trained glue: arms A and B share it, so B costs nothing but decode ----
    glue_soft = SlotBias(d_hidden, device)
    curve_soft = train_glue(
        glue_soft, model, char_table, tok, bank_can, tape, fit_facts, flat, off, train_docs,
        pad_id, V, device, steps=steps, k=k, gate_l1=args.gate_l1, nce_q=nce_q, nce_slot=nce_slot,
        nce_w=args.nce_w, nce_tau=args.nce_tau, open_only=False, rng=rng, t0=t0, tag="soft",
    )

    head_only = exam(
        glue_soft, model, char_table, tok, bank_can, tape, eval_facts, pad_id, V, device, k, max_new,
        locked=False, use_glue=False,
    )
    trace_a: list = []
    A = run_exam(glue_soft, tape, eval_facts, locked=False, trace=trace_a)
    trace_b: list = []
    B = run_exam(glue_soft, tape, eval_facts, locked=True, trace=trace_b)
    log(
        f"arm A (soft train, soft decode)  EM={A['em']:.3f} em3={A['em3']:.3f} gate={A['gate_mean']:.3f}\n"
        f"arm B (soft train, LOCKED decode) em_span={B['em_span']:.3f} em_text={B['em_text']:.3f} "
        f"glue_bpe={B['glue_bpe_rate']:.3f} verbatim={B['verbatim']:.3f} open_rec={B['open_recall']:.3f} "
        f"restart={B['restart_rate']:.3f}"
    )

    # ---- open-trained glue: the gate never sees an in-span step ----
    C = None
    curve_open = []
    glue_open = None
    trace_c: list = []
    if not args.no_arm_c:
        glue_open = SlotBias(d_hidden, device)
        curve_open = train_glue(
            glue_open, model, char_table, tok, bank_can, tape, fit_facts, flat, off, train_docs,
            pad_id, V, device, steps=steps, k=k, gate_l1=args.gate_l1, nce_q=nce_q, nce_slot=nce_slot,
            nce_w=args.nce_w, nce_tau=args.nce_tau, open_only=True, rng=rng, t0=t0, tag="open",
        )
        trace_c = []
        C = run_exam(glue_open, tape, eval_facts, locked=True, trace=trace_c)
        log(
            f"arm C (open train, LOCKED decode) em_span={C['em_span']:.3f} em_text={C['em_text']:.3f} "
            f"glue_bpe={C['glue_bpe_rate']:.3f} verbatim={C['verbatim']:.3f} open_rec={C['open_recall']:.3f} "
            f"restart={C['restart_rate']:.3f}"
        )

    best = C if (C is not None and C["em"] >= B["em"]) else B
    best_glue = glue_open if best is C else glue_soft
    best_name = "C_open_locked" if best is C else "B_soft_locked"

    # ---- controls, all on the winning locked arm ----
    shuf = run_exam(best_glue, tape.shuffled(SEED + 1), eval_facts, locked=True)
    empty = run_exam(best_glue, tape.emptied(), eval_facts, locked=True)
    per_fact_after, retained = [], []
    for f in eval_facts:
        tape_del = tape.copy()
        tape_del.drop_value(f["value"])
        per_fact_after.append(run_exam(best_glue, tape_del, [f], locked=True)["em"])
        others = [o for o in eval_facts if o is not f]
        if others:
            retained.append(run_exam(best_glue, tape_del, others, locked=True)["em"])
    em_tgt_after = float(np.mean(per_fact_after)) if per_fact_after else float("nan")
    em_ret_after = float(np.mean(retained)) if retained else float("nan")

    with torch.no_grad():
        gp, ce_on, ce_off = [], [], []
        erng = random.Random(SEED + 99)
        for _ in range(12):
            ids = s251.sample_windows_docs(flat, off, 1, erng, pad_id, hold_docs).to(device)
            l_on, g = prose_batch(
                best_glue, model, char_table, tok, bank_can, tape, ids, pad_id, V, device, k, 0.0, True
            )
            l_off, _ = prose_batch(
                best_glue, model, char_table, tok, bank_can, tape, ids, pad_id, V, device, k, 0.0, False
            )
            if l_on is not None and l_off is not None:
                ce_on.append(float(l_on))
                ce_off.append(float(l_off))
            if not math.isnan(g):
                gp.append(g)
    gate_prose = float(np.mean(gp)) if gp else float("nan")
    prose_ce_glue = float(np.mean(ce_on)) if ce_on else float("nan")
    prose_ce_plain = float(np.mean(ce_off)) if ce_off else float("nan")

    # ---- the two readings that matter beyond EM ----
    def split_em(res):
        w = [r for r in res["rows"] if r["gold"] in {f["value"] for f in eval_wiki}]
        nn_ = [r for r in res["rows"] if r["gold"] in {f["value"] for f in eval_non}]
        return (
            float(np.mean([r["em"] for r in w])) if w else float("nan"),
            float(np.mean([r["em"] for r in nn_])) if nn_ else float("nan"),
        )

    a_wiki, a_non = split_em(A)
    b_wiki, b_non = split_em(best)
    len_soft = em_by_length(A["rows"])
    len_lock = em_by_length(best["rows"])
    fb = full_bank_read(best_glue, bank_can, tok, tape, eval_facts, pad_id)
    log(
        f"length: soft short={len_soft['em_short']:.3f} long={len_soft['em_long']:.3f} | "
        f"locked short={len_lock['em_short']:.3f} long={len_lock['em_long']:.3f}\n"
        f"prior:  soft wiki={a_wiki:.3f} nonsense={a_non:.3f} | locked wiki={b_wiki:.3f} nonsense={b_non:.3f}\n"
        f"full bank: top1={fb.get('top1')} mrr={fb.get('mrr')} bank={fb.get('bank_size')}"
    )

    em256 = published_em_256()
    # The soft arm is the validity anchor, but only a full run is comparable to 256's published
    # number: at --smoke it trains 200 steps on 4 fit facts and lands near 0.375 by design.
    # Validity is that the soft arm reproduces 256's MECHANISM, not its number. The full run made
    # the difference plain: A landed at 0.525 against 256's published 0.75 on an exam whose values
    # run to 9 BPE tokens (256's clustered at 2-3), with 16 nonsense-valued facts folded into the
    # same average and max_new 8 instead of 6. Comparing those two numbers compares two exams. What
    # has to hold is that retrieval is saturated exactly as it was in 256, the gate opens on facts,
    # and the glue beats the bare head -- then any EM gap is the decode question this stage exists
    # to answer.
    g_soft_reproduces_256 = (
        A["em"] > head_only["em"] + 0.20
        and fb.get("top1", 0.0) >= 0.95
        and not math.isnan(A["gate_mean"])
        and A["gate_mean"] >= 0.50
    )
    g_verbatim = best["verbatim"] >= 0.95 * best["open_recall"] and best["open_recall"] > 0
    g_locked_beats_soft = best["em"] >= A["em"] + 0.10
    g_length_flat = (
        not math.isnan(len_lock["em_long"])
        and not math.isnan(len_lock["em_short"])
        and len_lock["em_long"] >= len_lock["em_short"] - 0.10
    )
    g_prior_invariant = (
        not math.isnan(b_wiki) and not math.isnan(b_non) and abs(b_wiki - b_non) <= 0.10
    )
    g_no_restart = best["restart_rate"] <= 0.05
    g_tape_causal = shuf["em"] <= max(0.10, best["em"] - 0.40)
    g_no_leak = empty["em"] <= 0.10
    g_slot_delete = best["em"] >= 0.4 and em_tgt_after <= 0.1 and em_ret_after >= 0.7 * best["em"]
    g_lang_intact = (
        not math.isnan(prose_ce_glue)
        and not math.isnan(prose_ce_plain)
        and prose_ce_glue <= prose_ce_plain + 0.05
    )
    g_gate_selective = (
        not math.isnan(best["gate_mean"]) and not math.isnan(gate_prose) and best["gate_mean"] >= gate_prose + 0.20
    )

    valid = g_soft_reproduces_256 and g_tape_causal and g_no_leak and g_lang_intact
    core = valid and g_verbatim and g_locked_beats_soft and g_no_restart
    if core and g_length_flat and g_prior_invariant and g_slot_delete and g_gate_selective:
        overall = "SPAN_LOCK_OK"
    elif core:
        overall = "SPAN_LOCK_PARTIAL"
    elif not valid:
        overall = "SPAN_LOCK_INVALID"
    else:
        overall = "SPAN_LOCK_NO"

    def strip(res):
        return {kk: vv for kk, vv in res.items() if kk != "rows"} if res else None

    out = {
        "stage": 265,
        "overall": overall,
        "trunk": trunk_ckpt.name,
        "fp_version": fp_version(),
        "topk": k,
        "steps": steps,
        "open_thresh": args.open_thresh,
        "max_new": max_new,
        "tape_slots": len(vals),
        "n_fit": len(fit_facts),
        "n_eval_wiki": len(eval_wiki),
        "n_eval_nonsense": len(eval_non),
        "best_arm": best_name,
        "gates": {
            "G_soft_reproduces_256": g_soft_reproduces_256,
            "G_span_verbatim": g_verbatim,
            "G_locked_beats_soft": g_locked_beats_soft,
            "G_length_flat": g_length_flat,
            "G_prior_invariant": g_prior_invariant,
            "G_no_restart": g_no_restart,
            "G_tape_causal": g_tape_causal,
            "G_no_param_leak": g_no_leak,
            "G_slot_delete_clean": g_slot_delete,
            "G_lang_intact": g_lang_intact,
            "G_gate_selective": g_gate_selective,
        },
        "arms": {
            "A_soft_soft": strip(A),
            "B_soft_locked": strip(B),
            "C_open_locked": strip(C),
            "head_only": strip(head_only),
        },
        "controls": {
            "em_shuffled_tape": shuf["em"],
            "em_empty_tape": empty["em"],
            "em_target_after_delete": em_tgt_after,
            "em_retained_after_delete": em_ret_after,
            "prose_ce_glue_on": prose_ce_glue,
            "prose_ce_glue_off": prose_ce_plain,
            "gate_mean_prose": gate_prose,
            "hold_ce_base": base_hold,
        },
        "length": {"soft": len_soft, "locked": len_lock},
        "prior_split": {
            "soft_wiki": a_wiki,
            "soft_nonsense": a_non,
            "locked_wiki": b_wiki,
            "locked_nonsense": b_non,
        },
        "full_bank_at_cue": fb,
        "em_256_published": em256,
        "curve": {"soft": curve_soft, "open": curve_open},
        "note": "Span-lock: the gate decides only WHERE a value starts; once open, tape.tok_ids is "
        "emitted verbatim and neither the gate nor the LM is consulted until the span ends. Training "
        "stays a soft mixture (257: hard-commit-only training collapses the stop gate); open-train "
        "additionally withholds gradient from the gate on in-span steps. G_span_verbatim is an "
        "assertion: an opened span is bit-identical to the slot. Headline EM is em_span (first word "
        "of the emitted span). em_text scores the full max_new continuation alongside — when the LM "
        "BPE-glues onto the value (Whammy+n -> Whammyn) em_span stays high and em_text drops; "
        "glue_bpe_rate counts that gap. The end-of-value boundary is NOT solved: stop-at-span and "
        "em_span only defer it via the one-answer exam. Same defect family as restart; on a tape "
        "where generation continues after the value, it returns at full strength. G_length_flat and "
        "G_prior_invariant are the scaling claims.",
        "deferred": {
            "end_of_value_boundary": (
                "Not solved. em_span / max_opens=1 hide LM subword glue and multi-value continuation. "
                "Next line: an explicit end-of-value signal (or refuse to hand the wheel back to LM "
                "without a boundary), not another exam-only stop."
            )
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    TRACE.write_text(
        json.dumps(
            {
                "soft": trace_a[:12],
                "locked": trace_b[:12],
                "open_locked": trace_c[:12],
                "C_verbatim_em_mismatch": [
                    {
                        "S": r["S"],
                        "gold": r["gold"],
                        "got": r["got"],
                        "em_first_word": r.get("em_first_word"),
                        "gold_decode": r.get("gold_decode"),
                        "em": r["em"],
                        "em3": r["em3"],
                        "verbatim": r["verbatim"],
                        "opened_correct": r["opened_correct"],
                        "opened_at": r["opened_at"],
                        "opened_value": r["opened_value"],
                        "truncated": r["truncated"],
                        "n_val_tokens": r["n_val_tokens"],
                        "opens": r.get("opens"),
                    }
                    for r in (C["rows"] if C is not None else [])
                    if bool(r.get("verbatim")) != bool(r.get("em"))
                ],
                "B_verbatim_em_mismatch": [
                    {
                        "S": r["S"],
                        "gold": r["gold"],
                        "got": r["got"],
                        "em_first_word": r.get("em_first_word"),
                        "gold_decode": r.get("gold_decode"),
                        "em": r["em"],
                        "verbatim": r["verbatim"],
                        "opened_correct": r["opened_correct"],
                        "opened_at": r["opened_at"],
                        "truncated": r["truncated"],
                        "n_val_tokens": r["n_val_tokens"],
                    }
                    for r in B["rows"]
                    if bool(r.get("verbatim")) != bool(r.get("em"))
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    MINI.write_text(
        f"# Stage 265 span-lock\n\n**{overall}** trunk={trunk_ckpt.name} slots={len(vals)} "
        f"eval={len(eval_facts)} best={best_name}\n\n"
        f"- EM headline (=em_span): head_only **{head_only['em']:.3f}** | soft **{A['em']:.3f}** | "
        f"locked **{best.get('em_span', best['em']):.3f}** "
        f"(em_text **{best.get('em_text', float('nan')):.3f}**, "
        f"glue_bpe **{best.get('glue_bpe_rate', float('nan')):.3f}**)\n"
        f"- verbatim spans **{best['verbatim']:.3f}**, open recall {best['open_recall']:.3f}, "
        f"restart {best['restart_rate']:.3f} (undefined for the soft arm: it opens no spans)\n"
        f"- length (short/long): soft {len_soft['em_short']:.2f}/{len_soft['em_long']:.2f}, "
        f"locked {len_lock['em_short']:.2f}/{len_lock['em_long']:.2f}\n"
        f"- prior (wiki/nonsense): soft {a_wiki:.2f}/{a_non:.2f}, locked {b_wiki:.2f}/{b_non:.2f}\n"
        f"- causal: shuffled {shuf['em']:.3f}, empty {empty['em']:.3f}, "
        f"delete {best['em']:.2f}->{em_tgt_after:.2f} (retained {em_ret_after:.2f})\n"
        f"- full bank @ cue: top1 {fb.get('top1')} mrr {fb.get('mrr')} over {fb.get('bank_size')} slots\n"
        f"- deferred: end-of-value boundary not solved — em_span only defers LM BPE-glue\n",
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"], "arms": out["arms"]}, indent=2))

    if not args.smoke and best_glue is not None:
        CKPT_OUT.parent.mkdir(exist_ok=True)
        torch.save(
            {
                "W_q_glue": best_glue.W_q.state_dict(),
                "gate": best_glue.gate.state_dict(),
                "log_tau": best_glue.log_tau.detach().cpu(),
                "stage": 265,
                "arm": best_name,
                "fp_version": fp_version(),
            },
            CKPT_OUT,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
