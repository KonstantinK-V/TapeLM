"""
Stage 261 — Can a natural question drive the retrieval, with no cue template anywhere?

256/257 used hand-written cues. 258 removed the cue's lexical overlap but still drew wording
from a fixed per-relation dictionary, so W_sem could learn "this template -> that relation".
Both sides of the exam were still authored. This stage authors neither.

The fact is written from one REAL wikitext sentence mentioning entity E. The question is a
DIFFERENT real wikitext sentence mentioning the same E, truncated just before it. Both
contexts are natural prose, written by different people about the same thing, and nothing in
between was designed by us:

    slot   key = norm( fp(anchor_A) + ctx_fp(sentence A, exclude=E) ),  value = E
    query  from sentence B, prefix up to where E begins        gold = that slot

The discriminator, and the reason this stage exists: LEXICAL OVERLAP between the two contexts.
Bag-of-spellings retrieval works when A and B happen to share words. Report accuracy split by
overlap quartile — if it only works in the high-overlap half, the query is still spelling
matching and NL_QUERY_LEXICAL_ONLY is the verdict, not a win.

Channels compared on identical queries:
    fp-only          W_q(anchor fp + ctx_fp)              = the 256 path
    fp + semantic    blend with W_sem(h_t), as in 258     = trunk understanding in the query

The bank also has to contain entities that are NOT on the exam. The first run built it from the
exam entities alone - 53 slots for 26 fit and 27 eval items - so InfoNCE could satisfy itself by
learning "point at one of these 26", drove the loss to 0.007 by step 40, and then sent eval
queries to the same places: fp+sem 0.037 against fp-only 0.148, below even the shuffled control.
Wiki noise slots make that shortcut worthless.

Keys canonical frozen fp; P1 and trunk frozen; only W_q, W_sem and the blend train. Entities
used for fitting and for evaluation are disjoint.

  python _stage261_nl_query.py [--smoke] [--no-gpt-control]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage24x_lib as L
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, hidden_and_logits

RES = Path("results")
DECISION = RES / "stage261_decision.json"
MINI = RES / "stage261_mini.md"
LOG = RES / "_stage261_log.txt"
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_JOINT = Path("checkpoints/stage253_joint_l02.pt")
CKPT_OUT = Path("checkpoints/stage261_nl_query.pt")
WIKI = Path("data/_wikitext103_train.txt")
WORD_RE = re.compile(r"[A-Za-z][a-z]{2,}")
SEED = 261


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


class SemQuery(nn.Module):
    """Trunk state -> key space, plus how much to trust it (fp top1 and margin as features)."""

    def __init__(self, in_dim: int, device):
        super().__init__()
        self.proj = nn.Linear(in_dim, 256).to(device)
        self.blend = nn.Sequential(nn.Linear(in_dim + 2, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
        nn.init.zeros_(self.blend[-1].weight)
        nn.init.constant_(self.blend[-1].bias, -2.0)

    def q(self, h):
        return F.normalize(self.proj(h), dim=-1)

    def a(self, h, conf):
        return torch.sigmoid(self.blend(torch.cat([h, conf], dim=-1))).squeeze(-1)


def fp_conf(q_fp, K):
    sims = q_fp @ K.t()
    two = torch.topk(sims, min(2, sims.size(-1)), dim=-1).values
    if two.size(-1) < 2:
        return torch.stack([two[..., 0], two[..., 0]], dim=-1)
    return torch.stack([two[..., 0], two[..., 0] - two[..., 1]], dim=-1)


def ctx_words(text: str, exclude: str | None = None) -> set:
    return {w.lower() for w in WORD_RE.findall(text) if w != exclude}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def fp_raw(bank: FpBank, text: str):
    c = bank.ctx_fp(text)
    if c is None:
        return None
    an = ANCHOR_RE.findall(text)
    return F.normalize(bank.fp([an[-1]])[0] + c, dim=-1) if an else c


@torch.no_grad()
def trunk_state(model, char_table, tok, pad_id, device, text):
    ids = [i for i in tok.encode(text).ids if i != pad_id][-MAX_ARCS:]
    if not ids:
        return None
    h, _ = hidden_and_logits(model, char_table, torch.tensor([ids], device=device), pad_id)
    return h[0, -1].detach().float()


def collect(lines, bank, min_occ=2):
    """Entities appearing in at least two different real sentences: one writes the slot, the
    other asks the question. Neither sentence was authored by us."""
    by_ent = defaultdict(list)
    for ln in lines:
        for m in ENT_RE.finditer(ln):
            e = m.group(1)
            if len(e) < 5:
                continue
            lo = max(0, m.start() - 140)
            an = [w for w in ANCHOR_RE.findall(ln[lo : m.start()]) if w != e]
            if an and len(by_ent[e]) < 4:
                by_ent[e].append({"line": ln, "start": m.start(), "end": m.end(), "anchor": an[-1]})
    return {e: v for e, v in by_ent.items() if len(v) >= min_occ}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--entities", type=int, default=0)
    ap.add_argument("--distractor-slots", type=int, default=0,
                    help="wiki entities added to the bank that no query ever asks for")
    ap.add_argument("--no-gpt-control", action="store_true")
    args = ap.parse_args()

    LOG.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    steps = args.steps or (200 if args.smoke else 800)
    n_ent = args.entities or (60 if args.smoke else 400)
    n_dist = args.distractor_slots or (400 if args.smoke else 4000)
    max_lines = 3000 if args.smoke else 25000

    log(f"Stage261 nl query start {datetime.now(timezone.utc).isoformat()} device={device} steps={steps}")

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
    bank = FpBank(model_can, stoi, device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(3_000_000 if args.smoke else 20_000_000)
    lines = [l.strip() for l in wtext.split("\n") if 80 <= len(l.strip()) <= 400][:max_lines]
    cands = collect(lines, bank)
    ents = sorted(cands)[:n_ent]
    rng.shuffle(ents)
    log(f"  entities with >=2 natural mentions: {len(cands)} (using {len(ents)})")
    if len(ents) < 16:
        log("  not enough multi-mention entities")
        return 1

    keys, vals, items = [], [], []
    for e in ents:
        occ = cands[e]
        a, b = occ[0], occ[1]  # A writes the slot, B asks
        wctx = a["line"][max(0, a["start"] - 140) : min(len(a["line"]), a["end"] + 140)]
        k = bank.ctx_fp(wctx, exclude=e)
        if k is None:
            continue
        qtext = b["line"][max(0, b["start"] - 200) : b["start"]].strip()
        if len(WORD_RE.findall(qtext)) < 4:
            continue
        raw = fp_raw(bank, qtext)
        st = trunk_state(model, char_table, tok, pad_id, device, qtext)
        if raw is None or st is None:
            continue
        keys.append(F.normalize(bank.fp([a["anchor"]])[0] + k, dim=-1))
        items.append({"ent": e, "slot": len(vals), "qtext": qtext, "raw": raw, "h": st,
                      "overlap": jaccard(ctx_words(wctx, e), ctx_words(qtext, e))})
        vals.append(e)
    if len(items) < 16:
        log("  not enough usable (write, ask) pairs")
        return 1
    # ---- noise: real wiki entities nobody asks about, so "point at an exam slot" is not a policy
    n_exam = len(keys)
    used = {it["ent"] for it in items}
    for ln in lines:
        if len(keys) >= n_exam + n_dist:
            break
        for m in ENT_RE.finditer(ln):
            e = m.group(1)
            if len(e) < 5 or e in used:
                continue
            lo, hi = max(0, m.start() - 140), min(len(ln), m.end() + 140)
            c = bank.ctx_fp(ln[lo:hi], exclude=e)
            if c is None:
                continue
            an = [w for w in ANCHOR_RE.findall(ln[lo : m.start()]) if w != e]
            if not an:
                continue
            keys.append(F.normalize(bank.fp([an[-1]])[0] + c, dim=-1))
            vals.append(e)
            used.add(e)
            if len(keys) >= n_exam + n_dist:
                break
    K = torch.stack(keys, 0).to(device).float()
    log(f"  bank: {n_exam} exam slots + {len(keys) - n_exam} wiki noise = {len(keys)}")
    n_fit = len(items) // 2
    fit, ev = items[:n_fit], items[n_fit:]
    med = float(np.median([it["overlap"] for it in ev]))
    log(f"  exam_slots={n_exam} fit={len(fit)} eval={len(ev)} | overlap median={med:.3f}")

    semq = SemQuery(int(fit[0]["h"].numel()), device)
    W_q = L.init_query_adapter(device)
    opt = torch.optim.AdamW(list(semq.parameters()) + list(W_q.parameters()), lr=2e-3, weight_decay=0.01)
    Rq = torch.stack([it["raw"] for it in fit]).to(device).float()
    Hq = torch.stack([it["h"] for it in fit]).to(device).float()
    Gq = torch.tensor([it["slot"] for it in fit], device=device)
    for step in range(1, steps + 1):
        sel = torch.randint(0, Rq.size(0), (min(32, Rq.size(0)),), device=device)
        q_fp = F.normalize(W_q(Rq[sel]), dim=-1)
        a = semq.a(Hq[sel], fp_conf(q_fp, K)).unsqueeze(-1)
        q = F.normalize((1 - a) * q_fp + a * semq.q(Hq[sel]), dim=-1)
        loss = F.cross_entropy((q @ K.t()) / args.tau, Gq[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(semq.parameters()) + list(W_q.parameters()), 1.0)
        opt.step()
        if step == 40 or step % max(1, steps // 5) == 0:
            log(f"  step {step}/{steps} loss={float(loss):.3f} a={float(a.mean()):.3f}")
    semq.eval()

    @torch.no_grad()
    def score(items_, use_sem, Kmat=K, n_way: int = 20):
        """top1 over the whole bank is open-domain entity linking over thousands of candidates.
        n-way asks the prior question - is there a signal at all - by scoring the gold slot
        against a fixed random sample of others."""
        wrng = random.Random(SEED + 5)
        ranks, alphas, lo, hi, nway = [], [], [], [], []
        for it in items_:
            q_fp = F.normalize(W_q(it["raw"].unsqueeze(0)), dim=-1)[0]
            if use_sem:
                a = semq.a(it["h"], fp_conf(q_fp, Kmat))
                q = F.normalize((1 - a) * q_fp + a * semq.q(it["h"]), dim=-1)
                alphas.append(float(a))
            else:
                q = q_fp
            sims = Kmat @ q
            r = 1 + int((sims > sims[it["slot"]]).sum())
            ranks.append(r)
            (hi if it["overlap"] > med else lo).append(int(r == 1))
            pool = [j for j in wrng.sample(range(Kmat.size(0)), min(n_way * 3, Kmat.size(0)))
                    if j != it["slot"]][: n_way - 1]
            nway.append(int(all(float(sims[it["slot"]]) >= float(sims[j]) for j in pool)))
        r = np.asarray(ranks, dtype=np.float64)
        return {
            "top1": float(np.mean(r == 1)), "mrr": float(np.mean(1.0 / r)),
            "median_rank": float(np.median(r)),
            "top1_low_overlap": float(np.mean(lo)) if lo else float("nan"),
            "top1_high_overlap": float(np.mean(hi)) if hi else float("nan"),
            "alpha": float(np.mean(alphas)) if alphas else 0.0, "n": len(ranks),
            f"acc_{n_way}way": float(np.mean(nway)) if nway else float("nan"),
            f"chance_{n_way}way": 1.0 / n_way,
        }

    fp_only, sem = score(ev, False), score(ev, True)
    perm = torch.randperm(K.size(0), generator=torch.Generator().manual_seed(SEED + 1))
    shuf = score(ev, True, Kmat=K[perm.to(K.device)])
    log(f"fp-only: {json.dumps(fp_only)}")
    log(f"fp+sem : {json.dumps(sem)}")
    log(f"shuffled keys: top1={shuf['top1']:.3f}")

    gpt = None
    if not args.no_gpt_control:
        try:
            gm = L.load_gpt(device)
            hs = []
            for it in items:
                e = L.gpt_emb(gm, tok, pad_id, device,
                              [i for i in tok.encode(it["qtext"]).ids if i != pad_id])
                hs.append(None if e is None else e.detach().float())
            if all(x is not None for x in hs):
                for it, x in zip(items, hs):
                    it["h_gpt"] = x
                semg = SemQuery(int(hs[0].numel()), device)
                Wg = L.init_query_adapter(device)
                og = torch.optim.AdamW(list(semg.parameters()) + list(Wg.parameters()), lr=2e-3)
                Hg = torch.stack([it["h_gpt"] for it in fit]).to(device).float()
                for _ in range(steps):
                    sel = torch.randint(0, Rq.size(0), (min(32, Rq.size(0)),), device=device)
                    qf = F.normalize(Wg(Rq[sel]), dim=-1)
                    a = semg.a(Hg[sel], fp_conf(qf, K)).unsqueeze(-1)
                    q = F.normalize((1 - a) * qf + a * semg.q(Hg[sel]), dim=-1)
                    lo_ = F.cross_entropy((q @ K.t()) / args.tau, Gq[sel])
                    og.zero_grad(set_to_none=True)
                    lo_.backward()
                    og.step()
                semg.eval()
                with torch.no_grad():
                    ranks = []
                    for it in ev:
                        qf = F.normalize(Wg(it["raw"].unsqueeze(0)), dim=-1)[0]
                        a = semg.a(it["h_gpt"], fp_conf(qf, K))
                        q = F.normalize((1 - a) * qf + a * semg.q(it["h_gpt"]), dim=-1)
                        s = K @ q
                        ranks.append(1 + int((s > s[it["slot"]]).sum()))
                    rr = np.asarray(ranks, dtype=np.float64)
                    gpt = {"top1": float(np.mean(rr == 1)), "mrr": float(np.mean(1.0 / rr))}
                log(f"gpt2+sem: {json.dumps(gpt)}")
        except Exception as e:  # noqa: BLE001 — control is optional
            log(f"  gpt control unavailable: {type(e).__name__}: {e}")

    chance = 1.0 / len(vals)
    g_works = sem["top1"] >= 0.30
    g_beats_fp = sem["top1"] >= fp_only["top1"] + 0.10
    g_low_overlap = (not math.isnan(sem["top1_low_overlap"])) and sem["top1_low_overlap"] >= 0.25
    # a ratio gate is vacuous when both halves are zero: 0.000 >= 0.6 * 0.000 passed in the
    # full run while nothing worked at all
    g_not_lexical = (
        not math.isnan(sem["top1_low_overlap"]) and not math.isnan(sem["top1_high_overlap"])
        and sem["top1_low_overlap"] > 0.0
        and sem["top1_low_overlap"] >= 0.6 * sem["top1_high_overlap"]
    )
    g_signal_20way = sem.get("acc_20way", 0.0) >= 0.05 + 1.0 / 20
    gpt_also_fails = gpt is not None and gpt["top1"] <= max(0.02, fp_only["top1"])
    g_causal = shuf["top1"] <= max(0.05, chance * 3)

    if g_works and g_beats_fp and g_causal and g_not_lexical and g_low_overlap:
        overall = "NL_QUERY_OK"
    elif g_works and g_causal and not g_not_lexical:
        overall = "NL_QUERY_LEXICAL_ONLY"  # only where the two natural contexts share words
    elif g_works and g_causal:
        overall = "NL_QUERY_PARTIAL"
    elif g_signal_20way and g_causal:
        overall = "NL_QUERY_NWAY_ONLY"   # signal exists, open-domain top1 does not
    elif gpt_also_fails:
        overall = "NL_QUERY_NO_AT_SCALE"  # matched GPT-2 is no better: a 3050 statement
    else:
        overall = "NL_QUERY_NO"

    out = {
        "stage": 261, "overall": overall, "trunk": trunk_ckpt.name, "steps": steps,
        "slots": len(vals), "exam_slots": n_exam, "noise_slots": len(vals) - n_exam,
        "n_fit": len(fit), "n_eval": len(ev), "chance": chance,
        "overlap_median": med,
        "gates": {
            "G_works": g_works, "G_beats_fp_only": g_beats_fp,
            "G_low_overlap_works": g_low_overlap, "G_not_lexical": g_not_lexical,
            "G_tape_causal": g_causal, "G_signal_20way": g_signal_20way,
        },
        "gpt_parity": bool(gpt_also_fails),
        "summary": {"fp_only": fp_only, "fp_plus_sem": sem, "shuffled_keys": shuf, "gpt_control": gpt},
        "note": "Full: 353 exam + 4000 wiki noise. Headline NL_QUERY_NO_AT_SCALE (GPT top1 0). "
        "acc_20way: fp-only beats chance (~4.4x); fp+sem at high alpha can harm fp — see results/stage261_close.md. "
        "G_not_lexical requires top1_low_overlap > 0.",
        "timestamp": datetime.now(timezone.utc).isoformat(), "wall_s": time.time() - t0,
    }
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 261 natural-question retrieval\n\n**{overall}** slots={len(vals)} eval={len(ev)} "
        f"chance={chance:.4f}\n\n"
        f"- top1: fp-only **{fp_only['top1']:.3f}** -> fp+sem **{sem['top1']:.3f}** "
        f"(shuffled {shuf['top1']:.3f})\n"
        f"- by overlap: low **{sem['top1_low_overlap']:.3f}** vs high **{sem['top1_high_overlap']:.3f}**"
        f" (median {med:.3f})\n"
        f"- 20-way (chance 0.05): fp-only **{fp_only.get('acc_20way', float('nan')):.3f}** -> "
        f"fp+sem **{sem.get('acc_20way', float('nan')):.3f}** (shuffled {shuf.get('acc_20way', float('nan')):.3f})\n"
        f"- mrr {sem['mrr']:.3f}, median rank {sem['median_rank']:.0f}, blend a {sem['alpha']:.3f}\n"
        + (f"- matched GPT-2: top1 {gpt['top1']:.3f}\n" if gpt else "- matched GPT-2: not run\n"),
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"]}, indent=2))
    if not args.smoke:
        CKPT_OUT.parent.mkdir(exist_ok=True)
        torch.save({"W_q": W_q.state_dict(), "sem": semq.state_dict(), "stage": 261}, CKPT_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
