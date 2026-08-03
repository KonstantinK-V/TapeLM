"""
Stage 256 — Glue layer: slot-bias decoding (228c fp-decode x free-form head).

228c showed the fp path picks the right value 1.0 of the time, but only as a CONSTRAINED
choice over a 4-way candidate set. The head generates freely but never sees which slots were
retrieved (228a: HEAD_LEXICAL_PRIOR_ONLY, sensitivity ~0.036). This stage glues them:

  p'_t = (1 - g_t) * p_LM(t) + g_t * p_copy(t | tape, q_t)

  q_t      = W_q(ctx_fp(prefix))            queries move, tape KEYS stay frozen canonical
  p_copy   = span-aware distribution over the next token of top-k retrieved slot values
  g_t      = sigmoid(MLP([h_t, max_sim, mean_topk, entropy, coverage]))  "read the tape now?"

Mixing in PROBABILITY space, not as an additive logit bonus, is what makes the gate honest: an
additive bias has to out-shout logits of order ~10, and leaving the gate open costs nothing, so
it saturates at 1.0 and the tape stays decorative. Under a mixture, g_t=1 means "answer purely
from the tape", so on ordinary prose (where p_copy puts ~0 on the true next token) an open gate
is paid for directly in CE. A small L1 on g_t over prose keeps it from drifting back up.

Trunk is FROZEN. Only the glue trains: W_q + gate MLP + tau. Values live in the tape
only — the CE text has the fact sentence replaced by a placeholder, so the gradient toward the
right value can flow ONLY through the bias path. That keeps 244-style unlearning honest: delete
the slot and the answer dies.

Ablations that make the test strong (not just "number looks good"):
  head_only       glue off
  shuffle_tape    permute keys, breaking key<->value pairing
  slot_delete     drop the target slot, check target dies and retained survive
  empty_tape      no slots at all (parametric leak floor)
  prose gate      mean g_t on ordinary wiki windows must stay low

  python _stage256_slot_bias_decode.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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

RES = Path("results")
DECISION = RES / "stage256_decision.json"
MINI = RES / "stage256_mini.md"
LOG = RES / "_stage256_log.txt"
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_JOINT = Path("checkpoints/stage253_joint_l02.pt")
CKPT_OUT = Path("checkpoints/stage256_slot_bias.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 256

FACT_TMPL = "{S} was appointed director of {V} in the regional chronicle of 1987 ."
CUE = "{S} was appointed director of"
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


import _inprint_glue as glue_lib
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

CUE = DEFAULT_CUE
FACT_TMPL = DEFAULT_FACT_TMPL


def nce_loss(glue: SlotBias, raw_q: torch.Tensor, gold_mask: torch.Tensor, K: torch.Tensor, tau: float):
    """InfoNCE over the whole bank: pull the adapted cue query onto its slot, push off the rest.

    CE through the copy mixture only nudges retrieval second-hand (it can lower the loss by
    sharpening tau instead), so W_q needs a direct retrieval objective.
    """
    q = F.normalize(glue.W_q(raw_q), dim=-1)
    sims = (q @ K.t()) / tau
    pos = torch.where(gold_mask, sims, torch.full_like(sims, -1e4)).logsumexp(dim=-1)
    return (sims.logsumexp(dim=-1) - pos).mean()


def fact_batch(glue, model, char_table, tok, bank, tape, facts, pad_id, V, device, k: int):
    """Teacher-forced CE on the value tokens, logits corrected by the gated slot bias."""
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
                logp = torch.log(F.softmax(base, -1) + 1e-9)
                g_val = torch.zeros((), device=device)
            else:
                sims, idx = hit
                ent = float(-(F.softmax(base, -1) * F.log_softmax(base, -1)).sum())
                p_copy, cov = copy_dist(glue, tape, sims, idx, prefix, V, device)
                g_val = glue.g(h[0, t], float(sims.max()), float(sims.mean()), ent, cov)
                logp = mix_logprob(base, g_val, p_copy, cov)
            losses.append(-logp[tid])
            gates.append(float(g_val))
    if not losses:
        return None, float("nan")
    return torch.stack(losses).mean(), float(np.mean(gates))


def prose_batch(
    glue,
    model,
    char_table,
    tok,
    bank,
    tape,
    ids: torch.Tensor,
    pad_id,
    V,
    device,
    k: int,
    gate_l1: float,
    use_glue: bool = True,
):
    """Same glue on ordinary text. Under a mixture an open gate directly costs CE here; the L1 term
    only stops it from drifting up where the LM happens to be uncertain anyway."""
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
            # score it anyway, otherwise glue-on and glue-off average over different positions
            losses.append(-torch.log(F.softmax(base, -1) + 1e-9)[seq[t + 1]])
            gates.append(0.0)
            continue
        sims, idx = hit
        ent = float(-(F.softmax(base, -1) * F.log_softmax(base, -1)).sum())
        p_copy, cov = copy_dist(glue, tape, sims, idx, prefix, V, device)
        g_val = glue.g(h[0, t], float(sims.max()), float(sims.mean()), ent, cov)
        logp = mix_logprob(base, g_val, p_copy, cov)
        losses.append(-logp[seq[t + 1]] + gate_l1 * g_val)
        gates.append(float(g_val))
    if not losses:
        return None, float("nan")
    return torch.stack(losses).mean(), float(np.mean(gates))


@torch.no_grad()
def free_decode(
    glue, model, char_table, tok, bank, tape, fact, pad_id, V, device, k: int, max_new: int, use_glue: bool
) -> tuple[str, float]:
    """Greedy free-form continuation of the cue; no candidate set anywhere."""
    cue_ids = [i for i in tok.encode(CUE.format(S=fact["S"])).ids if i != pad_id]
    seq = list(cue_ids)
    gen, gates = [], []
    for _ in range(max_new):
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
                score = mix_logprob(base, g_val, p_copy, cov)
                gates.append(float(g_val))
        nxt = int(score.argmax())
        gen.append(nxt)
        seq.append(nxt)
    return tok.decode(gen).strip(), (float(np.mean(gates)) if gates else float("nan"))


@torch.no_grad()
def retrieval_report(glue, bank, tok, tape: TapeView, facts, pad_id, k: int) -> list[dict]:
    """At the cue (the exact state free decode starts from): where does the gold slot rank?"""
    rows = []
    for f in facts:
        cue_ids = [i for i in tok.encode(CUE.format(S=f["S"])).ids if i != pad_id]
        q = ctx_query(glue, bank, tok, cue_ids, anchor_ids=cue_ids)
        if q is None:
            rows.append({"S": f["S"], "rank": None})
            continue
        sims = tape.K @ q
        gold = [j for j, v in enumerate(tape.values) if v == f["value"]]
        gsim = float(sims[gold].max()) if gold else float("-inf")
        rank = 1 + int((sims > gsim).sum())
        top = tape.values[int(sims.argmax())]
        w = glue.weights(torch.topk(sims, min(k, sims.numel()))[0])
        rows.append(
            {
                "S": f["S"],
                "gold": f["value"],
                "rank": rank,
                "top1": top,
                "gold_sim": gsim,
                "w_max": float(w.max()),
            }
        )
    return rows


def exact_match(text: str, value: str) -> bool:
    return text.strip().split(" ")[0].strip(" .,;:") == value if text else False


@torch.no_grad()
def em_over(
    glue, model, char_table, tok, bank, tape, facts, pad_id, V, device, k, max_new, use_glue=True, samples=None
):
    ok, gs = 0, []
    for f in facts:
        got, g = free_decode(
            glue, model, char_table, tok, bank, tape, f, pad_id, V, device, k, max_new, use_glue
        )
        ok += int(exact_match(got, f["value"]))
        if not math.isnan(g):
            gs.append(g)
        if samples is not None and len(samples) < 6:
            samples.append({"cue_S": f["S"], "gold": f["value"], "got": got, "gate": g})
    return ok / max(1, len(facts)), (float(np.mean(gs)) if gs else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--gate-l1", type=float, default=0.02, help="L1 on g_t over prose steps")
    ap.add_argument("--nce-w", type=float, default=1.0, help="weight of the retrieval InfoNCE term")
    ap.add_argument("--nce-tau", type=float, default=0.05)
    ap.add_argument(
        "--nce-pool",
        choices=("wiki", "facts"),
        default="wiki",
        help="train W_q on bank-wide (prefix->slot) pairs, or overfit the fit facts (ablation)",
    )
    ap.add_argument("--facts", type=int, default=0)
    ap.add_argument("--distractor-slots", type=int, default=0, help="real wiki entities added as bank noise")
    args = ap.parse_args()

    LOG.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    steps = args.steps or (200 if args.smoke else 800)
    n_facts = args.facts or (8 if args.smoke else 48)
    n_dist = args.distractor_slots or (150 if args.smoke else 1200)
    max_new = 4 if args.smoke else 6
    n_hold = 4 if args.smoke else 12
    n_exam = 40 if args.smoke else 120
    max_lines = 400 if args.smoke else 6000
    k = args.topk

    log(
        f"Stage256 slot-bias glue start {datetime.now(timezone.utc).isoformat()} device={device} "
        f"steps={steps} facts={n_facts} distractors={n_dist} topk={k}"
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
    log(f"  trunk={trunk_ckpt.name} (frozen)")

    model_can = SelfModelXL(n_char, V).to(device)
    model_can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model_can.eval()
    for p in model_can.parameters():
        p.requires_grad_(False)
    bank_can = FpBank(model_can, stoi, device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(1_000_000 if args.smoke else 6_000_000)
    values_pool = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5))
    rng.shuffle(values_pool)
    lines = [l.strip() for l in wtext.split("\n") if len(l.strip()) >= 60][:max_lines]

    # ---- facts: half fit the glue, half are held out (255 lesson: never score what you fit) ----
    subs = [w for w in gen_fakes(set(values_pool), rng, n_facts + 30) if len(w) >= 5][:n_facts]
    facts = []
    for i, S in enumerate(subs):
        Vv = values_pool[i]
        facts.append(
            {
                "S": S,
                "value": Vv,
                "sent": FACT_TMPL.format(S=S, V=Vv),
                "fid": f"f{i}",
                "glue_train": i % 2 == 0,
            }
        )
    fit_facts = [f for f in facts if f["glue_train"]]
    eval_facts = [f for f in facts if not f["glue_train"]]
    log(f"  facts: fit={len(fit_facts)} held_out={len(eval_facts)}")

    # ---- tape: canonical frozen keys; fact sentences are NOT in the CE text ----
    keys, vals = [], []
    pair_q, pair_slot = [], []  # (prefix -> slot) pairs harvested from wiki noise, for W_q training
    for f in facts:
        kf = bank_can.fp([f["S"]])[0]
        c = bank_can.ctx_fp(f["sent"], exclude=f["value"])
        keys.append(F.normalize(kf + c, dim=-1) if c is not None else kf)
        vals.append(f["value"])
    used = set(vals)
    for ln in lines:
        if len(vals) >= n_facts + n_dist:
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
            # the query a decoder would form right before emitting this entity
            cq = bank_can.ctx_fp(ln[lo : m.start()])
            if cq is not None:
                pair_q.append(F.normalize(bank_can.fp([anchors[-1]])[0] + cq, dim=-1))
                pair_slot.append(len(vals))
            vals.append(ent)
            used.add(ent)
            if len(vals) >= n_facts + n_dist:
                break
    tape = TapeView(torch.stack(keys, 0).to(device), vals, tok, pad_id)
    log(f"  tape slots={len(vals)} ({len(facts)} planted + {len(vals)-len(facts)} wiki noise)")

    # prose corpus: fact sentences replaced by a placeholder, so values never enter CE text
    prose = "\n".join(lines + [PLACEHOLDER] * min(len(facts), len(lines) // 4))
    flat, off = s213.build_flat_from_text(prose, tok, pad_id, max_lines=max_lines + 64, min_line_len=20)
    n_docs = len(off) - 1
    hold_docs = list(range(max(1, n_docs - max(2, n_docs // 20)), n_docs))
    train_docs = list(range(0, hold_docs[0]))
    hold_batches = s252.make_hold_batches(flat, off, hold_docs, pad_id, n_hold, SEED + 5)
    items = s251.load_exam_next(n_exam)
    log(f"  prose docs={n_docs} train={len(train_docs)} hold={len(hold_docs)}")

    glue = SlotBias(2 * (model.head.in_features // 2), device)
    opt = torch.optim.AdamW(glue.trainable(), lr=3e-3, weight_decay=0.01)

    # W_q is trained on (prefix -> slot) pairs from the BANK, not on the planted facts. Fitting it on
    # a handful of facts only teaches it where those facts live: the smoke run showed held-out cues
    # collapsing onto a fit fact's slot while its own NCE sat at 0.006.
    if args.nce_pool == "facts":
        with torch.no_grad():
            pq, ps = [], []
            for f in fit_facts:
                cue_ids = [i for i in tok.encode(CUE.format(S=f["S"])).ids if i != pad_id]
                rq = raw_query(bank_can, tok, cue_ids, anchor_ids=cue_ids)
                if rq is None:
                    continue
                pq.append(rq)
                ps.append(vals.index(f["value"]))
        pair_q, pair_slot = pq, ps
    nce_q = torch.stack(pair_q).to(device).float() if pair_q else None
    nce_slot = torch.tensor(pair_slot, device=device) if pair_slot else None
    K_all = tape.K.float()
    log(f"  W_q training pairs={0 if nce_q is None else nce_q.size(0)} (pool={args.nce_pool})")

    base_hold = s252.fixed_hold_ce(model, hold_batches, char_table, pad_id, device)
    base_exam = s251.next_tok_acc(model, char_table, pad_id, items, device)
    em_head, _ = em_over(
        glue, model, char_table, tok, bank_can, tape, eval_facts, pad_id, V, device, k, max_new, use_glue=False
    )
    log(f"baseline hold_ce={base_hold:.3f} exam={base_exam:.3f} EM(head_only)={em_head:.3f}")

    # ---- train glue only ----
    curve = []
    for step in range(1, steps + 1):
        batch = [fit_facts[rng.randrange(len(fit_facts))] for _ in range(min(4, len(fit_facts)))]
        l_fact, g_fact = fact_batch(
            glue, model, char_table, tok, bank_can, tape, batch, pad_id, V, device, k
        )
        ids = s251.sample_windows_docs(flat, off, 1, rng, pad_id, train_docs).to(device)
        l_prose, g_prose = prose_batch(
            glue, model, char_table, tok, bank_can, tape, ids, pad_id, V, device, k, args.gate_l1
        )
        l_nce = None
        if nce_q is not None and args.nce_w > 0:
            sel = torch.randint(0, nce_q.size(0), (min(64, nce_q.size(0)),), device=device)
            gold = F.one_hot(nce_slot[sel], K_all.size(0)).bool()
            l_nce = args.nce_w * nce_loss(glue, nce_q[sel], gold, K_all, args.nce_tau)
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
        if step % max(1, steps // 6) == 0:
            curve.append(
                {
                    "step": step,
                    "loss_fact": float(l_fact) if l_fact is not None else None,
                    "loss_prose": float(l_prose) if l_prose is not None else None,
                    "loss_nce": float(l_nce) if l_nce is not None else None,
                    "gate_fact": g_fact,
                    "gate_prose": g_prose,
                    "tau": float(torch.exp(glue.log_tau)),
                }
            )
            log(
                f"  step {step}/{steps} fact={float(l_fact) if l_fact is not None else float('nan'):.3f} "
                f"prose={float(l_prose) if l_prose is not None else float('nan'):.3f} "
                f"nce={float(l_nce) if l_nce is not None else float('nan'):.3f} "
                f"g_fact={g_fact:.3f} g_prose={g_prose:.3f} "
                f"tau={float(torch.exp(glue.log_tau)):.3f} ({time.time()-t0:.0f}s)"
            )

    glue.eval()

    # ---- evaluation: free-form EM + causal ablations, all on held-out facts ----
    ret_eval = retrieval_report(glue, bank_can, tok, tape, eval_facts, pad_id, k)
    ret_fit = retrieval_report(glue, bank_can, tok, tape, fit_facts, pad_id, k)
    r_eval = [r["rank"] for r in ret_eval if r.get("rank")]
    r_fit = [r["rank"] for r in ret_fit if r.get("rank")]
    log(
        f"retrieval at cue: held-out top1={np.mean([r == 1 for r in r_eval]):.2f} "
        f"median_rank={np.median(r_eval):.0f} | fit top1={np.mean([r == 1 for r in r_fit]):.2f}"
    )
    for r in ret_eval[:4]:
        log(f"    {r}")

    decodes: list[dict] = []
    em_glue, g_glue = em_over(
        glue, model, char_table, tok, bank_can, tape, eval_facts, pad_id, V, device, k, max_new,
        samples=decodes,
    )
    for d in decodes[:4]:
        log(f"    decode {d}")
    em_shuf, _ = em_over(
        glue, model, char_table, tok, bank_can, tape.shuffled(SEED + 1), eval_facts, pad_id, V, device, k, max_new
    )
    em_empty, _ = em_over(
        glue, model, char_table, tok, bank_can, tape.emptied(), eval_facts, pad_id, V, device, k, max_new
    )

    # slot delete, averaged over every held-out fact: each one is decoded with ITS OWN slot removed.
    # Judging this on a single fact makes the gate hostage to whether that one fact decodes at all.
    per_fact_after, retained = [], []
    for f in eval_facts:
        tape_del = tape.copy()
        tape_del.drop_value(f["value"])
        em_f, _ = em_over(
            glue, model, char_table, tok, bank_can, tape_del, [f], pad_id, V, device, k, max_new
        )
        per_fact_after.append(em_f)
        others = [o for o in eval_facts if o is not f]
        if others:
            em_o, _ = em_over(
                glue, model, char_table, tok, bank_can, tape_del, others, pad_id, V, device, k, max_new
            )
            retained.append(em_o)
    em_tgt_before = em_glue
    em_tgt_after = float(np.mean(per_fact_after)) if per_fact_after else float("nan")
    em_ret_after = float(np.mean(retained)) if retained else float("nan")

    # language: the SAME held-out windows, glue on vs off — otherwise the gate is untested
    with torch.no_grad():
        gp, ce_on, ce_off = [], [], []
        erng = random.Random(SEED + 99)
        for _ in range(12):
            ids = s251.sample_windows_docs(flat, off, 1, erng, pad_id, hold_docs).to(device)
            l_on, g = prose_batch(
                glue, model, char_table, tok, bank_can, tape, ids, pad_id, V, device, k, 0.0, True
            )
            l_off, _ = prose_batch(
                glue, model, char_table, tok, bank_can, tape, ids, pad_id, V, device, k, 0.0, False
            )
            if l_on is not None and l_off is not None:
                ce_on.append(float(l_on))
                ce_off.append(float(l_off))
            if not math.isnan(g):
                gp.append(g)
    gate_prose = float(np.mean(gp)) if gp else float("nan")
    prose_ce_glue = float(np.mean(ce_on)) if ce_on else float("nan")
    prose_ce_plain = float(np.mean(ce_off)) if ce_off else float("nan")
    hold_after = s252.fixed_hold_ce(model, hold_batches, char_table, pad_id, device)

    g_freeform = em_glue >= 0.60
    g_beats_head = em_glue >= em_head + 0.20
    g_tape_causal = em_shuf <= max(0.10, em_glue - 0.40)
    g_slot_delete = em_tgt_before >= 0.4 and em_tgt_after <= 0.1 and em_ret_after >= 0.7 * em_glue
    g_no_leak = em_empty <= 0.10
    # trunk is frozen, so hold_ce cannot move by itself; the real question is whether the glue
    # damages ordinary text when it is switched on over the same windows
    g_lang_intact = (
        not math.isnan(prose_ce_glue)
        and not math.isnan(prose_ce_plain)
        and prose_ce_glue <= prose_ce_plain + 0.05
    )
    g_gate_selective = (not math.isnan(g_glue)) and (not math.isnan(gate_prose)) and g_glue >= gate_prose + 0.20

    core = g_freeform and g_beats_head and g_tape_causal and g_no_leak and g_lang_intact
    if core and g_slot_delete and g_gate_selective:
        overall = "SLOT_BIAS_GLUE_OK"
    elif g_beats_head and g_tape_causal and g_no_leak and g_lang_intact:
        overall = "SLOT_BIAS_GLUE_PARTIAL"
    else:
        overall = "SLOT_BIAS_GLUE_NO"

    out = {
        "stage": 256,
        "overall": overall,
        "trunk": trunk_ckpt.name,
        "topk": k,
        "steps": steps,
        "n_facts": len(facts),
        "n_fit": len(fit_facts),
        "n_eval": len(eval_facts),
        "tape_slots": len(vals),
        "gates": {
            "G_freeform_value": g_freeform,
            "G_beats_head_only": g_beats_head,
            "G_tape_causal": g_tape_causal,
            "G_slot_delete_clean": g_slot_delete,
            "G_no_param_leak": g_no_leak,
            "G_lang_intact": g_lang_intact,
            "G_gate_selective": g_gate_selective,
        },
        "summary": {
            "em_head_only": em_head,
            "em_glue": em_glue,
            "em_shuffled_tape": em_shuf,
            "em_empty_tape": em_empty,
            "em_target_before_delete": em_tgt_before,
            "em_target_after_delete": em_tgt_after,
            "em_retained_after_delete": em_ret_after,
            "gate_mean_fact": g_glue,
            "gate_mean_prose": gate_prose,
            "prose_ce_glue_on": prose_ce_glue,
            "prose_ce_glue_off": prose_ce_plain,
            "hold_ce_base": base_hold,
            "hold_ce_after": hold_after,
            "exam_base": base_exam,
            "tau": float(torch.exp(glue.log_tau)),
            "gate_l1": args.gate_l1,
        },
        "curve": curve,
        "retrieval_at_cue": {
            "held_out_top1": float(np.mean([r == 1 for r in r_eval])) if r_eval else None,
            "held_out_median_rank": float(np.median(r_eval)) if r_eval else None,
            "fit_top1": float(np.mean([r == 1 for r in r_fit])) if r_fit else None,
            "rows": ret_eval[:8],
        },
        "decode_samples": decodes,
        "note": "Glue only: trunk frozen, W_q + gate MLP + tau trained. Copy mixture "
        "p' = (1-g)p_LM + g*p_copy, so an open gate is paid for in CE. Values exist in the tape "
        "only, so CE toward the right value can flow only through the bias path. EM is free-form greedy "
        "decode (no candidate set); scored on facts the glue never fit.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.time() - t0,
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 256 slot-bias glue\n\n**{overall}** trunk={trunk_ckpt.name} slots={len(vals)} "
        f"eval_facts={len(eval_facts)}\n\n"
        f"- EM free-form: head_only **{em_head:.3f}** -> glue **{em_glue:.3f}**\n"
        f"- causal: shuffled **{em_shuf:.3f}**, empty **{em_empty:.3f}**\n"
        f"- slot delete: target {em_tgt_before:.2f} -> {em_tgt_after:.2f}, retained {em_ret_after:.2f}\n"
            f"- gate: fact **{g_glue:.3f}** vs prose **{gate_prose:.3f}**\n"
            f"- prose CE glue off {prose_ce_plain:.3f} -> on {prose_ce_glue:.3f} (hold CE {base_hold:.3f})\n",
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"], "summary": out["summary"]}, indent=2))

    if not args.smoke:
        CKPT_OUT.parent.mkdir(exist_ok=True)
        torch.save(
            {
                "W_q": glue.W_q.state_dict(),
                "gate": glue.gate.state_dict(),
                "log_tau": glue.log_tau.detach().cpu(),
                "stage": 256,
            },
            CKPT_OUT,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
