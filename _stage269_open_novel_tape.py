"""
Stage 269 — 268's claim, measured on an exam that has room to fail.

268 came back MIND_LEARNS_TAPE_PARTIAL with EM 1.000 on the last training tape and 1.000 on a
tape never seen. G_novel_tape was true, but it compared two ceilings: retrieval on the planted
exam is saturated (256/263 spent months on exactly this), span-lock then makes EM follow
retrieval, and a gate that cannot fail proves nothing. 269 keeps 268's training verbatim and
changes only what is measured.

Every tape now carries a second population of slots that is never trained on:

    planted facts   cue template + value, half fit / half held out   -> TRAINS the procedure
    open entities   key written from real sentence A, question is
                    the prefix of a different real sentence B        -> SCORES it
    distractors     wiki noise

The open half is 261/264/267's exam, where zero-train word votes reach top1 0.246 on 4352 slots.
There is headroom by construction, and `G_headroom` asserts it rather than assuming it: if the
trained query lands above 0.90 on the training tape, the verdict is NOVEL_TAPE_SATURATED and the
run says so instead of claiming transfer.

Two comparisons matter, and 268 had neither:

  G_novel_tape   the same procedure on a tape whose open entities were never used in any rebuild.
                 Tapes are rebuilt every ~200 steps, so nothing factual survives; if this holds,
                 what transferred is procedure.

  G_beats_votes  the trained query against zero-train postings on the identical items. 266 showed
                 a learned query vector losing to plain words 0.062 vs 0.199. If the unfrozen mind
                 still loses, "the mind learned to use the tape" is not the right sentence.

268's G_beats_frozen_mind is dropped. It ran the trained glue against the untouched trunk and read
0.000 — but 265 got 0.975 from that same frozen trunk, so what it measured was the glue having
co-adapted to a drifting trunk, not the mind's contribution. The honest control is a paired run
with --frozen-baseline: identical budget, identical exam, set_train_mode("none"), its own glue.

  python _stage269_open_novel_tape.py --smoke
  python _stage269_open_novel_tape.py                     # night, upper trains
  python _stage269_open_novel_tape.py --frozen-baseline   # the paired control
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from collections import defaultdict
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
import _stage265_span_lock as s265
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _stage261_nl_query import WORD_RE, collect, ctx_words, jaccard
from _inprint_glue import (
    ANCHOR_RE,
    DEFAULT_CUE,
    DEFAULT_FACT_TMPL,
    SlotBias,
    TapeView,
    ctx_query,
)
from _tape_index import context_words

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_JOINT = Path("checkpoints/stage253_joint_l02.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 269

CUE = DEFAULT_CUE
FACT_TMPL = DEFAULT_FACT_TMPL
PLACEHOLDER = s265.PLACEHOLDER


def paths(frozen: bool):
    tag = "_frozen" if frozen else ""
    return (
        RES / f"stage269_decision{tag}.json",
        RES / f"stage269_mini{tag}.md",
        RES / f"_stage269_log{tag}.txt",
    )


LOG_PATH = RES / "_stage269_log.txt"


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def fp_version() -> str:
    fn = getattr(L, "canonical_fp_version", None)
    if callable(fn):
        try:
            return str(fn())
        except Exception:
            pass
    return CKPT_P1.name


def arc_enc_hash(model: SelfModelXL) -> str:
    h = hashlib.sha256()
    for _, t in sorted(model.arc_enc.state_dict().items()):
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# tape: planted facts (train) + open cross-mention entities (score) + distractors
# --------------------------------------------------------------------------------------
def build_tape(
    *, bank_can, tok, pad_id, device, rng, values_pool, lines, cands, used,
    n_facts, n_nonsense, n_open, n_dist,
) -> dict:
    """One tape. `used` carries across rebuilds so no planted value and no open entity repeats.

    268 silently reset its pool when it ran dry, which would have let values repeat between tapes
    and quietly broken the whole "nothing factual survives a rebuild" claim. Here exhaustion is an
    error, and the count that would have hidden it is reported.
    """
    available = [w for w in values_pool if w not in used and len(w) >= 5]
    rng.shuffle(available)
    subs = [
        w for w in gen_fakes(set(used) | set(available), rng, n_facts + n_nonsense + 80)
        if len(w) >= 5 and w not in used
    ]
    subs = list(dict.fromkeys(subs))
    fake_vals = [
        w for w in gen_fakes(set(used) | set(available) | set(subs), rng, n_nonsense + 40)
        if len(w) >= 6 and w not in subs and w not in used
    ]
    fake_vals = list(dict.fromkeys(fake_vals))[:n_nonsense]
    open_pool = [e for e in sorted(cands) if e not in used]
    rng.shuffle(open_pool)
    if len(subs) < n_facts + len(fake_vals) or len(available) < n_facts or len(open_pool) < n_open:
        raise RuntimeError(
            f"pool exhausted: subs={len(subs)} avail={len(available)} open={len(open_pool)} "
            f"(need facts={n_facts} nonsense={n_nonsense} open={n_open}) — raise the corpus, "
            f"do not recycle: repeated values would break the novel-tape claim"
        )

    facts = []
    for i in range(n_facts):
        facts.append({
            "S": subs[i], "value": available[i],
            "sent": FACT_TMPL.format(S=subs[i], V=available[i]),
            "glue_train": i % 2 == 0, "kind": "wiki",
        })
        used.add(available[i]); used.add(subs[i])
    for j, fv in enumerate(fake_vals):
        S = subs[n_facts + j]
        facts.append({
            "S": S, "value": fv, "sent": FACT_TMPL.format(S=S, V=fv),
            "glue_train": False, "kind": "nonsense",
        })
        used.add(fv); used.add(S)

    keys, vals, texts = [], [], []
    for f in facts:
        kf = bank_can.fp([f["S"]])[0]
        c = bank_can.ctx_fp(f["sent"], exclude=f["value"])
        keys.append(F.normalize(kf + c, dim=-1) if c is not None else kf)
        vals.append(f["value"])
        texts.append(f["sent"])

    # open half: key from mention A, question is the prefix of mention B. Never trained on.
    open_items = []
    for e in open_pool:
        if len(open_items) >= n_open:
            break
        occ = cands[e]
        a, b = occ[0], occ[1]
        wctx = a["line"][max(0, a["start"] - 140): min(len(a["line"]), a["end"] + 140)]
        qtext = b["line"][max(0, b["start"] - 200): b["start"]].strip()
        if len(WORD_RE.findall(qtext)) < 4:
            continue
        c = bank_can.ctx_fp(wctx, exclude=e)
        if c is None:
            continue
        keys.append(F.normalize(bank_can.fp([a["anchor"]])[0] + c, dim=-1))
        open_items.append({
            "ent": e, "slot": len(vals), "qtext": qtext, "wctx": wctx,
            "qwords": context_words(qtext, exclude=e),
            "overlap": jaccard(ctx_words(wctx, e), ctx_words(qtext, e)),
        })
        vals.append(e)
        texts.append(wctx)
        used.add(e)

    used_vals = set(vals)
    pair_q, pair_slot = [], []
    for ln in lines:
        if len(vals) >= len(facts) + len(open_items) + n_dist:
            break
        for m in ENT_RE.finditer(ln):
            ent = m.group(1)
            if len(ent) < 5 or ent in used_vals:
                continue
            lo, hi = max(0, m.start() - 120), min(len(ln), m.end() + 120)
            c = bank_can.ctx_fp(ln[lo:hi], exclude=ent)
            if c is None:
                continue
            anchors = [w for w in ANCHOR_RE.findall(ln[lo: m.start()]) if w != ent]
            if not anchors:
                continue
            keys.append(F.normalize(bank_can.fp([anchors[-1]])[0] + c, dim=-1))
            cq = bank_can.ctx_fp(ln[lo: m.start()])
            if cq is not None:
                pair_q.append(F.normalize(bank_can.fp([anchors[-1]])[0] + cq, dim=-1))
                pair_slot.append(len(vals))
            vals.append(ent)
            texts.append(ln[lo:hi])
            used_vals.add(ent)
            if len(vals) >= len(facts) + len(open_items) + n_dist:
                break

    postings: dict[str, list[int]] = defaultdict(list)
    for cid, (v, txt) in enumerate(zip(vals, texts)):
        for w in context_words(txt, exclude=v):
            postings[w].append(cid)
    idf = {w: 1.0 / math.log(2.0 + len(postings[w])) for w in postings}

    return {
        "tape": TapeView(torch.stack(keys, 0).to(device), vals, tok, pad_id),
        "fit_facts": [f for f in facts if f["glue_train"]],
        "eval_facts": [f for f in facts if not f["glue_train"]],
        "open_items": open_items,
        "postings": postings,
        "idf": idf,
        "nce_q": torch.stack(pair_q).to(device).float() if pair_q else None,
        "nce_slot": torch.tensor(pair_slot, device=device) if pair_slot else None,
        "n_slots": len(vals),
    }


# --------------------------------------------------------------------------------------
# scoring the open half
# --------------------------------------------------------------------------------------
def rank_stats(ranks, lows, n_slots) -> dict:
    r = np.asarray(ranks, dtype=np.float64)
    low = np.asarray(lows)
    hit = r == 1

    def _m(mask, arr):
        return float(arr[mask].mean()) if mask.any() else float("nan")

    return {
        "top1": float(hit.mean()),
        "mrr": float(np.mean(1.0 / r)),
        "median_rank": float(np.median(r)),
        "top1_low_overlap": _m(low, hit),
        "top1_high_overlap": _m(~low, hit),
        "n": len(ranks),
        "n_slots": n_slots,
    }


@torch.no_grad()
def score_trained_query(glue, bank_can, tok, tape, items, pad_id, med) -> dict:
    """Rank of the gold slot under the query the mind actually forms. No candidate pool."""
    ranks, lows = [], []
    for it in items:
        ids = [i for i in tok.encode(it["qtext"]).ids if i != pad_id]
        q = ctx_query(glue, bank_can, tok, ids)
        if q is None:
            ranks.append(tape.K.size(0))
            lows.append(it["overlap"] <= med)
            continue
        sims = tape.K @ q
        ranks.append(1 + int((sims > sims[it["slot"]]).sum()))
        lows.append(it["overlap"] <= med)
    return rank_stats(ranks, lows, tape.K.size(0))


def score_votes(items, postings, idf, n_slots, med) -> dict:
    """Zero-train postings on the identical items — the bar the mind has to clear (266)."""
    ranks, lows, silent = [], [], []
    for it in items:
        sc: dict[int, float] = defaultdict(float)
        for w in it["qwords"]:
            for cid in postings.get(w, ()):
                sc[cid] += idf.get(w, 0.0)
        g = sc.get(it["slot"], 0.0)
        # silence is last, not first (266's paraphrase arm read top1 0.477 on empty tables)
        ranks.append(n_slots if g <= 0.0 else 1 + sum(1 for v in sc.values() if v > g))
        lows.append(it["overlap"] <= med)
        silent.append(g <= 0.0)
    out = rank_stats(ranks, lows, n_slots)
    out["tie_at_zero_frac"] = float(np.mean(silent))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--tape-period", type=int, default=0)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--gate-l1", type=float, default=0.02)
    ap.add_argument("--nce-w", type=float, default=1.0)
    ap.add_argument("--nce-tau", type=float, default=0.05)
    ap.add_argument("--facts", type=int, default=0)
    ap.add_argument("--nonsense-facts", type=int, default=0)
    ap.add_argument("--open-items", type=int, default=0, help="never-trained cross-mention slots")
    ap.add_argument("--distractor-slots", type=int, default=0)
    ap.add_argument("--lr-glue", type=float, default=3e-3)
    ap.add_argument("--lr-upper", type=float, default=3e-5)
    ap.add_argument("--frozen-baseline", action="store_true",
                    help="paired control: identical budget, trunk NOT unfrozen, its own glue")
    args = ap.parse_args()

    global LOG_PATH
    DECISION, MINI, LOG_PATH = paths(args.frozen_baseline)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    steps = args.steps or (400 if args.smoke else 8000)
    tape_period = args.tape_period or (100 if args.smoke else 200)
    n_facts = args.facts or (8 if args.smoke else 48)
    n_nonsense = args.nonsense_facts or (4 if args.smoke else 16)
    n_open = args.open_items or (20 if args.smoke else 120)
    n_dist = args.distractor_slots or (150 if args.smoke else 1200)
    max_new = 6 if args.smoke else 12
    n_hold = 4 if args.smoke else 12
    max_lines = 1500 if args.smoke else 12000
    k = args.topk
    mode = "none" if args.frozen_baseline else "upper"

    log(f"Stage269 open-novel-tape start {datetime.now(timezone.utc).isoformat()} device={device} "
        f"steps={steps} tape_period={tape_period} open={n_open} dist={n_dist} trunk_mode={mode}")

    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)

    trunk_ckpt = CKPT_JOINT if CKPT_JOINT.exists() else CKPT_P1
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(trunk_ckpt, map_location=device, weights_only=False)["model"])
    s213.set_train_mode(model, mode)
    arc_hash0 = arc_enc_hash(model)
    n_live = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  trunk={trunk_ckpt.name} mode={mode} trainable_trunk_params={n_live} "
        f"arc_enc hash0={arc_hash0[:16]}…")

    model_can = SelfModelXL(n_char, V).to(device)
    model_can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model_can.eval()
    for p in model_can.parameters():
        p.requires_grad_(False)
    bank_can = FpBank(model_can, stoi, device)
    log(f"  fp_version={fp_version()}")

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(2_000_000 if args.smoke else 16_000_000)
    values_pool = list(dict.fromkeys(
        m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5
    ))
    rng.shuffle(values_pool)
    lines = [l.strip() for l in wtext.split("\n") if 60 <= len(l.strip()) <= 400][:max_lines]
    cands = collect(lines, bank_can)
    log(f"  entity pool={len(values_pool)} lines={len(lines)} multi_mention={len(cands)} "
        f"(need >= {n_open * (steps // tape_period + 2)} across rebuilds)")

    prose = "\n".join(lines + [PLACEHOLDER] * 32)
    flat, off = s213.build_flat_from_text(prose, tok, pad_id, max_lines=max_lines + 64, min_line_len=20)
    n_docs = len(off) - 1
    hold_docs = list(range(max(1, n_docs - max(2, n_docs // 20)), n_docs))
    train_docs = list(range(0, hold_docs[0]))
    hold_batches = s252.make_hold_batches(flat, off, hold_docs, pad_id, n_hold, SEED + 5)
    base_hold = s252.fixed_hold_ce(model, hold_batches, char_table, pad_id, device)
    log(f"  hold CE base={base_hold:.4f}")

    d_hidden = 2 * (model.head.in_features // 2)
    glue = SlotBias(d_hidden, device)
    opt_glue = torch.optim.AdamW(glue.trainable(), lr=args.lr_glue, weight_decay=0.01)
    live = [p for p in model.parameters() if p.requires_grad]
    opt_upper = torch.optim.AdamW(live, lr=args.lr_upper, weight_decay=0.01) if live else None

    used: set[str] = set()
    pack = None
    n_tapes = 0
    curve = []

    for step in range(1, steps + 1):
        if pack is None or (step - 1) % tape_period == 0:
            pack = build_tape(
                bank_can=bank_can, tok=tok, pad_id=pad_id, device=device, rng=rng,
                values_pool=values_pool, lines=lines, cands=cands, used=used,
                n_facts=n_facts, n_nonsense=n_nonsense, n_open=n_open, n_dist=n_dist,
            )
            n_tapes += 1
            log(f"  tape#{n_tapes} @step {step}: slots={pack['n_slots']} "
                f"fit={len(pack['fit_facts'])} open={len(pack['open_items'])} used={len(used)}")

        if mode != "none":
            s213.set_train_mode(model, mode)
        tape = pack["tape"]
        batch = [pack["fit_facts"][rng.randrange(len(pack["fit_facts"]))]
                 for _ in range(min(4, len(pack["fit_facts"])))]
        l_fact, g_fact = s265.fact_batch(
            glue, model, char_table, tok, bank_can, tape, batch, pad_id, V, device, k,
            open_only=True,
        )
        ids = s251.sample_windows_docs(flat, off, 1, rng, pad_id, train_docs).to(device)
        l_prose, g_prose = s265.prose_batch(
            glue, model, char_table, tok, bank_can, tape, ids, pad_id, V, device, k, args.gate_l1,
        )
        l_nce = None
        if pack["nce_q"] is not None and args.nce_w > 0:
            K_all = tape.K.float()
            sel = torch.randint(0, pack["nce_q"].size(0),
                                (min(64, pack["nce_q"].size(0)),), device=device)
            gold = F.one_hot(pack["nce_slot"][sel], K_all.size(0)).bool()
            l_nce = args.nce_w * s265.nce_loss(glue, pack["nce_q"][sel], gold, K_all, args.nce_tau)
        parts = [x for x in (l_fact, l_prose, l_nce) if x is not None]
        if not parts:
            continue
        loss = parts[0]
        for p in parts[1:]:
            loss = loss + p
        opt_glue.zero_grad(set_to_none=True)
        if opt_upper is not None:
            opt_upper.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(glue.trainable()) + live, 1.0)
        opt_glue.step()
        if opt_upper is not None:
            opt_upper.step()

        if step % max(1, steps // 10) == 0 or step == steps:
            model.eval()
            med_t = float(np.median([it["overlap"] for it in pack["open_items"]]))
            with torch.no_grad():
                q = score_trained_query(glue, bank_can, tok, tape, pack["open_items"], pad_id, med_t)
            curve.append({
                "step": step, "tape": n_tapes, "open_top1": q["top1"],
                "open_median_rank": q["median_rank"],
                "loss_fact": float(l_fact) if l_fact is not None else None,
                "gate_fact": g_fact, "gate_prose": g_prose,
            })
            log(f"  step {step}/{steps} tape#{n_tapes} open_top1={q['top1']:.3f} "
                f"median={q['median_rank']:.0f} ({time.time()-t0:.0f}s)")
            if mode != "none":
                s213.set_train_mode(model, mode)

    glue.eval()
    model.eval()
    arc_hash1 = arc_enc_hash(model)
    g_arc = arc_hash0 == arc_hash1

    med_train = float(np.median([it["overlap"] for it in pack["open_items"]]))
    train_q = score_trained_query(glue, bank_can, tok, pack["tape"], pack["open_items"], pad_id, med_train)
    train_v = score_votes(pack["open_items"], pack["postings"], pack["idf"], pack["n_slots"], med_train)

    pack_novel = build_tape(
        bank_can=bank_can, tok=tok, pad_id=pad_id, device=device, rng=random.Random(SEED + 99),
        values_pool=values_pool, lines=lines, cands=cands, used=used,
        n_facts=n_facts, n_nonsense=n_nonsense, n_open=n_open, n_dist=n_dist,
    )
    med_novel = float(np.median([it["overlap"] for it in pack_novel["open_items"]]))
    novel_q = score_trained_query(
        glue, bank_can, tok, pack_novel["tape"], pack_novel["open_items"], pad_id, med_novel
    )
    novel_v = score_votes(
        pack_novel["open_items"], pack_novel["postings"], pack_novel["idf"],
        pack_novel["n_slots"], med_novel
    )
    log(f"  TRAIN tape open: query top1={train_q['top1']:.3f} votes top1={train_v['top1']:.3f}")
    log(f"  NOVEL tape open: query top1={novel_q['top1']:.3f} votes top1={novel_v['top1']:.3f} "
        f"median={novel_q['median_rank']:.0f} slots={pack_novel['n_slots']}")

    # planted half on the novel tape keeps 268's sanitation readable
    planted = s265.exam(
        glue, model, char_table, tok, bank_can, pack_novel["tape"], pack_novel["eval_facts"],
        pad_id, V, device, k, max_new, locked=True,
    )
    empty = s265.exam(
        glue, model, char_table, tok, bank_can, pack_novel["tape"].emptied(),
        pack_novel["eval_facts"], pad_id, V, device, k, max_new, locked=True,
    )
    shuf_tape = pack_novel["tape"].shuffled(SEED + 1)
    shuf_q = score_trained_query(
        glue, bank_can, tok, shuf_tape, pack_novel["open_items"], pad_id, med_novel
    )
    hold_after = s252.fixed_hold_ce(model, hold_batches, char_table, pad_id, device)
    log(f"  planted EM={planted['em']:.3f} empty={empty['em']:.3f} "
        f"shuffled_open_top1={shuf_q['top1']:.3f} hold {base_hold:.4f}->{hold_after:.4f}")

    # ---- gates ----
    g_headroom = train_q["top1"] < 0.90
    g_novel = bool(g_headroom and novel_q["top1"] >= train_q["top1"] - 0.05)
    g_beats_votes = bool(novel_q["top1"] >= novel_v["top1"] + 0.03)
    g_leak = empty["em"] <= 0.10
    g_causal = shuf_q["top1"] <= max(0.05, novel_q["top1"] - 0.10)
    g_lang = hold_after <= base_hold + 0.05

    if not g_headroom:
        overall = "NOVEL_TAPE_SATURATED"
    elif g_novel and g_beats_votes and g_arc and g_leak and g_causal:
        overall = "OPEN_NOVEL_TAPE_OK"
    elif g_novel and g_arc and g_leak and g_causal:
        overall = "OPEN_NOVEL_TAPE_PARTIAL"
    else:
        overall = "OPEN_NOVEL_TAPE_NO"

    out = {
        "stage": 269,
        "overall": overall,
        "frozen_baseline": args.frozen_baseline,
        "trunk_mode": mode,
        "trainable_trunk_params": n_live,
        "smoke": args.smoke,
        "seed": SEED,
        "trunk": trunk_ckpt.name,
        "fp_version": fp_version(),
        "steps": steps,
        "tape_period": tape_period,
        "n_tapes": n_tapes,
        "n_open_per_tape": n_open,
        "distractor_slots": n_dist,
        "used_pool_final": len(used),
        "gates": {
            "G_headroom": g_headroom,
            "G_novel_tape": g_novel,
            "G_beats_votes": g_beats_votes,
            "G_arc_enc_frozen": g_arc,
            "G_no_param_leak": g_leak,
            "G_tape_causal": g_causal,
            "G_lang_intact": g_lang,
        },
        "open_exam": {
            "train_tape_query": train_q,
            "train_tape_votes": train_v,
            "novel_tape_query": novel_q,
            "novel_tape_votes": novel_v,
            "shuffled_keys_query": shuf_q,
            "delta_novel_minus_train": novel_q["top1"] - train_q["top1"],
            "delta_query_minus_votes": novel_q["top1"] - novel_v["top1"],
        },
        "planted_half": {
            "em": planted["em"], "verbatim": planted["verbatim"],
            "open_recall": planted["open_recall"], "em_empty_tape": empty["em"],
        },
        "controls": {
            "hold_ce_base": base_hold, "hold_ce_after": hold_after,
            "arc_enc_hash_before": arc_hash0, "arc_enc_hash_after": arc_hash1,
        },
        "curve": curve,
        "note": (
            "268 read EM 1.000 on both tapes — two ceilings, so its G_novel_tape could not fail. "
            "Here the scored half is 261/264/267's cross-mention exam, never trained on, and "
            "G_headroom asserts the room to fail instead of assuming it. G_beats_votes is the "
            "question 266 left open: does an unfrozen mind beat zero-train postings on the same "
            "items. 268's G_beats_frozen_mind is gone — it compared a co-adapted glue against an "
            "untouched trunk and read 0.000 where 265 got 0.975; the paired --frozen-baseline run "
            "is the honest control. Tapes rebuild every tape_period steps and the pool never "
            "recycles, so nothing factual survives a rebuild."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 269 open novel tape{' (frozen baseline)' if args.frozen_baseline else ''}\n\n"
        f"**{overall}** · mode={mode} · tapes={n_tapes} · open={n_open}/tape · "
        f"slots≈{pack_novel['n_slots']}{' · SMOKE' if args.smoke else ''}\n\n"
        f"| exam (open half) | top1 | median rank |\n|---|---:|---:|\n"
        f"| train tape, trained query | {train_q['top1']:.3f} | {train_q['median_rank']:.0f} |\n"
        f"| **novel tape, trained query** | **{novel_q['top1']:.3f}** | {novel_q['median_rank']:.0f} |\n"
        f"| novel tape, zero-train votes | {novel_v['top1']:.3f} | {novel_v['median_rank']:.0f} |\n"
        f"| novel tape, shuffled keys | {shuf_q['top1']:.3f} | {shuf_q['median_rank']:.0f} |\n\n"
        f"## Gates (read G_headroom first)\n\n"
        + "".join(f"- {kk}: **{vv}**\n" for kk, vv in out["gates"].items())
        + f"\n- planted half EM {planted['em']:.3f}, empty tape {empty['em']:.3f}\n"
        f"- hold CE {base_hold:.3f} → {hold_after:.3f}\n",
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"],
                    "open": out["open_exam"]["delta_query_minus_votes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
