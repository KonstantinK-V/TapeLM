"""
Stage 276 — Put the search back in the loop.

275 reached its numbers with a filter: after every ASK, candidates not mentioning the asked
subject were dropped. That was a legitimate computation — the subject comes from the cue, not from
gold — but it handed retrieval to the policy for free, so 275 measured aggregation and abstention
and nothing else. This stage removes the crutch and reports what it was hiding.

Two things had to change before that was fair.

The witnesses were five fixed sentences, so every subject in the bank shared the same boilerplate.
Cue words like "appointed" and "director" therefore pulled other subjects' witnesses, a clean item
could see three repeats belonging to a neighbour, and the teacher would manufacture a majority out
of them. Each subject now carries its own filler words, drawn from a wiki line chosen by the
subject, so content words actually separate subjects.

And the weighting was `1 / log(2 + df)`, which barely distinguishes a term appearing in five slots
from one appearing in five hundred. With boilerplate in every planted sentence, that mass drowned
the one term that identifies the subject. `--idf classic` uses `log(N / df)` and is the default
here.

Both retrieval modes run, and the gap between them is the price of search:

    --subject-filter off   headline: the policy must find the witnesses itself
    --subject-filter on    control: 275's setting, retrieval free, aggregation only

G_retrieval_usable is a validity gate on the `off` arm. If precision and recall of the retrieve
set are too low, the templates still collide and no aggregation number below it means anything.

  python _stage276_search_in_loop.py --smoke
  python _stage276_search_in_loop.py --subject-filter on --smoke   # the 275 control
  python _stage276_search_in_loop.py --bc-episodes 4000 --rl-episodes 3000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage271_controller as s271
import _stage274_truthfree_oracle as s274
import _stage275_abstain as s275
from _stage191_night import PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, TapeView
from _tape_index import context_words

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_JOINT = Path("checkpoints/stage253_joint_l02.pt")
CKPT_OUT = Path("checkpoints/stage276_search_in_loop.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 276
FAMILIES = s275.FAMILIES

# Same five flat assertions as 270/275, with room for filler that belongs to one subject only.
WITNESS_TMPL = (
    "{S} was appointed director of {V} in the regional chronicle of 1987 , {F} .",
    "The county register lists {S} as appointed director of {V} that year , {F} .",
    "According to the parish record , {S} was appointed director of {V} , {F} .",
    "{S} , appointed director of {V} , appears in the 1987 civil roll , {F} .",
    "A ledger entry names {S} as the appointed director of {V} , {F} .",
)


def paths(tag: str):
    return (RES / f"stage276_decision_{tag}.json", RES / f"stage276_mini_{tag}.md",
            RES / f"_stage276_log_{tag}.txt")


LOG_PATH = RES / "_stage276_log_off.txt"


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def build_tape(*, bank, tok, pad_id, device, rng, pool, lines, used, n_clean, n_dec, n_tie,
               n_wit, n_liars, n_dist, idf_mode="classic", n_filler=4):
    """Every subject gets its own filler words, so content words separate subjects."""
    avail = [w for w in pool if w not in used and len(w) >= 5]
    rng.shuffle(avail)
    n_items = n_clean + n_dec + n_tie
    subs = [w for w in gen_fakes(set(used) | set(avail), rng, n_items + 80)
            if len(w) >= 5 and w not in used]
    subs = list(dict.fromkeys(subs))
    need = n_clean + 2 * (n_dec + n_tie)
    if len(subs) < n_items or len(avail) < need:
        raise RuntimeError(f"pool exhausted: subs={len(subs)} vals={len(avail)} need={need}")

    keys, vals, texts, items = [], [], [], []
    subject_of: dict[int, str] = {}
    vi, si = 0, 0

    def filler_for(S: str) -> str:
        ln = lines[rng.randrange(len(lines))]
        ws = [w for w in context_words(ln) if len(w) >= 4][:n_filler]
        return " ".join(ws) if ws else "recorded locally"

    def add(S, value, ti, filler):
        sent = WITNESS_TMPL[ti % len(WITNESS_TMPL)].format(S=S, V=value, F=filler)
        c = bank.ctx_fp(sent, exclude=value)
        kf = bank.fp([S])[0]
        keys.append(F.normalize(kf + c, dim=-1) if c is not None else kf)
        vals.append(value)
        texts.append(sent)
        subject_of[len(vals) - 1] = S
        return len(vals) - 1

    def block(kind, n_true, n_false):
        nonlocal si, vi
        S = subs[si]; si += 1
        fill = filler_for(S)
        truth = avail[vi]; vi += 1
        other = avail[vi]; vi += 1
        order = [truth] * n_true + [other] * n_false
        rng.shuffle(order)
        sids = [add(S, v, j, fill) for j, v in enumerate(order)]
        items.append({"S": S, "truth": None if kind == "tie" else truth,
                      "slots": sids, "kind": kind})
        used.add(truth); used.add(other); used.add(S)

    for _ in range(n_clean):
        S = subs[si]; si += 1
        v = avail[vi]; vi += 1
        sid = add(S, v, 0, filler_for(S))
        items.append({"S": S, "truth": v, "slots": [sid], "kind": "clean"})
        used.add(v); used.add(S)
    for _ in range(n_dec):
        block("decidable", n_wit - n_liars, n_liars)
    for _ in range(n_tie):
        half = n_wit // 2
        block("tie", half, half)

    seen = set(vals)
    target = len(vals) + n_dist
    for ln in lines:
        if len(vals) >= target:
            break
        for m in ENT_RE.finditer(ln):
            e = m.group(1)
            if len(e) < 5 or e in seen:
                continue
            lo, hi = max(0, m.start() - 120), min(len(ln), m.end() + 120)
            c = bank.ctx_fp(ln[lo:hi], exclude=e)
            if c is None:
                continue
            an = [w for w in ANCHOR_RE.findall(ln[lo:m.start()]) if w != e]
            if not an:
                continue
            keys.append(F.normalize(bank.fp([an[-1]])[0] + c, dim=-1))
            vals.append(e); texts.append(ln[lo:hi]); seen.add(e)
            if len(vals) >= target:
                break

    postings: dict[str, list[int]] = defaultdict(list)
    for cid, (v, t) in enumerate(zip(vals, texts)):
        for w in context_words(t, exclude=v):
            postings[w].append(cid)
    n_slots = len(vals)
    if idf_mode == "classic":
        # 1/log(2+df) hardly separates a term in five slots from one in five hundred, so shared
        # boilerplate outweighed the single term that names the subject.
        idf = {w: math.log(max(2.0, n_slots / max(1, len(postings[w])))) for w in postings}
    else:
        idf = {w: 1.0 / math.log(2.0 + len(postings[w])) for w in postings}
    return {"tape": TapeView(torch.stack(keys, 0).to(device), vals, tok, pad_id),
            "texts": texts, "items": items, "postings": postings, "idf": idf,
            "subject_of": subject_of}


def rollout(policy, model, char_table, tok, pack, item, pad_id, device, *, k, max_steps,
            max_reads, read_cost, wrong_cost, subject_filter, bc=False, greedy=True,
            teacher_only=False):
    tape, postings, idf = pack["tape"], pack["postings"], pack["idf"]
    s274._VALUE_OF = {i: v for i, v in enumerate(tape.values)}
    qtext = s271.CUE.format(S=item["S"])
    qwords = context_words(qtext)
    transcript = qtext
    cands: list[int] = []
    last_read_words: list[str] = []
    seen_reads: set[int] = set()
    opened: list[str] = []
    losses, logps, ents, trace = [], [], [], []
    n_reads, answered, abstained = 0, None, False
    prec, rec = float("nan"), float("nan")
    own_slots = set(item["slots"])

    for _ in range(max_steps):
        if teacher_only:
            a = s275.teacher(cands=cands, seen_reads=seen_reads, opened_values=opened,
                             n_reads=n_reads, max_steps=max_steps, max_reads=max_reads, k=k,
                             cand_scores=pack.get("_sc"))
        else:
            st = s274.state_tensors(policy, model, char_table, tok, pack, transcript, cands,
                                    seen_reads, opened, last_read_words, n_reads, pad_id,
                                    device, k, max_steps)
            if st is None:
                break
            logits, _ = st
            if bc:
                a = s275.teacher(cands=cands, seen_reads=seen_reads, opened_values=opened,
                                 n_reads=n_reads, max_steps=max_steps, max_reads=max_reads, k=k,
                                 cand_scores=pack.get("_sc"))
                if not torch.isfinite(logits[a]) or logits[a] < -1e8:
                    break
                losses.append(F.cross_entropy(logits.unsqueeze(0),
                                              torch.tensor([a], device=device)))
            else:
                dist = torch.distributions.Categorical(logits=logits)
                a = int(logits.argmax()) if greedy else int(dist.sample())
                logps.append(dist.log_prob(torch.tensor(a, device=device)))
                ents.append(dist.entropy())
        trace.append(s271.act_names(k)[a])

        if a in (s271.ASK_Q, s271.ASK_READ):
            words = qwords if a == s271.ASK_Q else last_read_words
            cands, sc = s271.vote(words, postings, idf, k)
            if subject_filter:
                own = [c for c in cands if item["S"] in pack["texts"][c]]
                cands = own if own else cands
            pack["_sc"] = {c: sc.get(c, 0.0) for c in cands}
            if cands:  # measured on the first ASK only — that is what the policy acts on
                hit = sum(1 for c in cands if c in own_slots)
                if math.isnan(prec):
                    prec = hit / len(cands)
                    rec = hit / max(1, len(own_slots))
        elif a == 2 + 2 * k:
            abstained = True
            break
        elif a < 2 + k:
            i = a - 2
            if i >= len(cands):
                break
            slot = cands[i]
            transcript = (transcript + " | " + pack["texts"][slot])[-2000:]
            last_read_words = context_words(pack["texts"][slot], exclude=tape.values[slot])
            seen_reads.add(slot)
            opened.append(tape.values[slot])
            n_reads += 1
        else:
            i = a - 2 - k
            if i >= len(cands):
                break
            answered = tape.values[cands[i]]
            break

    if abstained or answered is None:
        correct, reward, abstained = 0, 0.0, True
    else:
        correct = int(item["truth"] is not None and answered == item["truth"])
        reward = 1.0 if correct else -wrong_cost
    reward -= read_cost * n_reads
    return {
        "loss": torch.stack(losses).mean() if losses else torch.zeros((), device=device),
        "logps": logps, "entropy": ents, "reward": reward, "correct": correct,
        "abstained": abstained, "n_reads": n_reads, "trace": trace, "kind": item["kind"],
        "answer_is_slot": answered is None or answered in set(tape.values),
        "retrieval_precision": prec, "witness_recall": rec,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bc-episodes", type=int, default=0)
    ap.add_argument("--rl-episodes", type=int, default=0)
    ap.add_argument("--tape-period", type=int, default=0)
    ap.add_argument("--clean", type=int, default=4)
    ap.add_argument("--decidable", type=int, default=4)
    ap.add_argument("--tie", type=int, default=4)
    ap.add_argument("--witnesses", type=int, default=5)
    ap.add_argument("--liars", type=int, default=2)
    ap.add_argument("--distractor-slots", type=int, default=0)
    ap.add_argument("--topk", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--max-reads", type=int, default=7)
    ap.add_argument("--read-cost", type=float, default=0.02)
    ap.add_argument("--wrong-cost", type=float, default=0.3)
    ap.add_argument("--entropy-bonus", type=float, default=0.01)
    ap.add_argument("--lr-policy", type=float, default=1e-3)
    ap.add_argument("--lr-upper", type=float, default=3e-5)
    ap.add_argument("--subject-filter", choices=("off", "on"), default="off")
    ap.add_argument("--idf", choices=("classic", "soft"), default="classic")
    ap.add_argument("--filler", type=int, default=4)
    ap.add_argument("--frozen-trunk", action="store_true")
    args = ap.parse_args()
    sf = args.subject_filter == "on"

    global LOG_PATH
    tag = args.subject_filter + ("_frozen" if args.frozen_trunk else "")
    DECISION, MINI, LOG_PATH = paths(tag)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_bc = args.bc_episodes or (400 if args.smoke else 4000)
    n_rl = max(0, args.rl_episodes)
    tape_period = args.tape_period or (50 if args.smoke else 200)
    n_dist = args.distractor_slots or (150 if args.smoke else 1000)
    k = args.topk
    mode = "none" if args.frozen_trunk else "upper"

    log(f"Stage276 search-in-loop start {datetime.now(timezone.utc).isoformat()} device={device} "
        f"subject_filter={args.subject_filter} idf={args.idf} filler={args.filler} bc={n_bc} "
        f"rl={n_rl} k={k} mode={mode}")

    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)

    trunk_ckpt = CKPT_JOINT if CKPT_JOINT.exists() else CKPT_P1
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(trunk_ckpt, map_location=device, weights_only=False)["model"])
    s213.set_train_mode(model, mode)
    arc0 = s271.arc_enc_hash(model)

    model_can = SelfModelXL(n_char, V).to(device)
    model_can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model_can.eval()
    for p in model_can.parameters():
        p.requires_grad_(False)
    bank = FpBank(model_can, stoi, device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(1_500_000 if args.smoke else 8_000_000)
    pool = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5))
    rng.shuffle(pool)
    lines = [l.strip() for l in wtext.split("\n") if len(l.strip()) >= 60][:(400 if args.smoke else 6000)]

    policy = s274.Policy(2 * (model.head.in_features // 2), k, device)
    live = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": policy.parameters(), "lr": args.lr_policy}]
        + ([{"params": live, "lr": args.lr_upper}] if live else []),
        weight_decay=0.01,
    )

    used: set[str] = set()
    pack, baseline, curve = None, 0.0, []
    common = dict(k=k, max_steps=args.max_steps, max_reads=args.max_reads,
                  read_cost=args.read_cost, wrong_cost=args.wrong_cost, subject_filter=sf)

    def new_tape(r):
        return build_tape(bank=bank, tok=tok, pad_id=pad_id, device=device, rng=r, pool=pool,
                          lines=lines, used=used, n_clean=args.clean, n_dec=args.decidable,
                          n_tie=args.tie, n_wit=args.witnesses, n_liars=args.liars,
                          n_dist=n_dist, idf_mode=args.idf, n_filler=args.filler)

    policy.train()
    model.train(mode != "none")
    for ep in range(1, n_bc + 1):
        if pack is None or (ep - 1) % tape_period == 0:
            pack = new_tape(rng)
        item = pack["items"][rng.randrange(len(pack["items"]))]
        out = rollout(policy, model, char_table, tok, pack, item, pad_id, device, bc=True, **common)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + live, 1.0)
        opt.step()
        if ep % max(1, n_bc // 8) == 0:
            curve.append({"phase": "bc", "episode": ep, "loss": float(out["loss"]),
                          "kind": out["kind"], "trace": out["trace"]})
            log(f"  bc {ep}/{n_bc} loss={float(out['loss']):.4f} [{out['kind']}] {out['trace']}")

    for ep in range(1, n_rl + 1):
        if (ep - 1) % tape_period == 0:
            pack = new_tape(rng)
        item = pack["items"][rng.randrange(len(pack["items"]))]
        out = rollout(policy, model, char_table, tok, pack, item, pad_id, device,
                      greedy=False, **common)
        if not out["logps"]:
            continue
        baseline = 0.99 * baseline + 0.01 * out["reward"]
        ent = torch.stack(out["entropy"]).sum() if out["entropy"] else torch.zeros((), device=device)
        loss = -(out["reward"] - baseline) * torch.stack(out["logps"]).sum() - args.entropy_bonus * ent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + live, 1.0)
        opt.step()
        if ep % max(1, n_rl // 8) == 0:
            curve.append({"phase": "rl", "episode": ep, "baseline": baseline,
                          "kind": out["kind"], "trace": out["trace"]})
            log(f"  rl {ep}/{n_rl} baseline={baseline:.3f} [{out['kind']}] {out['trace']}")

    policy.eval()
    model.eval()
    arc1 = s271.arc_enc_hash(model)

    @torch.no_grad()
    def evaluate(p):
        per = {f: {"correct": [], "abstain": [], "reads": [], "reward": [],
                   "prec": [], "rec": []} for f in FAMILIES}
        tper = {f: {"correct": [], "abstain": [], "reward": []} for f in FAMILIES}
        slot_ok, traces = [], []
        for it in p["items"]:
            o = rollout(policy, model, char_table, tok, p, it, pad_id, device, **common)
            t = rollout(policy, model, char_table, tok, p, it, pad_id, device,
                        teacher_only=True, **common)
            f = it["kind"]
            per[f]["correct"].append(o["correct"]); per[f]["abstain"].append(int(o["abstained"]))
            per[f]["reads"].append(o["n_reads"]); per[f]["reward"].append(o["reward"])
            if not math.isnan(o["retrieval_precision"]):
                per[f]["prec"].append(o["retrieval_precision"])
                per[f]["rec"].append(o["witness_recall"])
            tper[f]["correct"].append(t["correct"]); tper[f]["abstain"].append(int(t["abstained"]))
            tper[f]["reward"].append(t["reward"])
            slot_ok.append(int(o["answer_is_slot"]))
            traces.append({"kind": f, "trace": o["trace"], "correct": o["correct"],
                           "abstained": o["abstained"], "prec": o["retrieval_precision"]})
        m = lambda xs: float(np.mean(xs)) if xs else float("nan")
        out = {"answer_is_slot": m(slot_ok), "traces": traces,
               "reward_total": m([r for f in FAMILIES for r in per[f]["reward"]]),
               "teacher_reward_total": m([r for f in FAMILIES for r in tper[f]["reward"]]),
               "retrieval_precision": m([x for f in FAMILIES for x in per[f]["prec"]]),
               "witness_recall": m([x for f in FAMILIES for x in per[f]["rec"]])}
        ac, an = 0, 0
        for f in FAMILIES:
            n_ans = sum(1 for a in per[f]["abstain"] if not a)
            ac += sum(per[f]["correct"]); an += n_ans
            out[f] = {"coverage": 1.0 - m(per[f]["abstain"]),
                      "acc_answered": (sum(per[f]["correct"]) / n_ans) if n_ans else float("nan"),
                      "abstain": m(per[f]["abstain"]), "mean_reads": m(per[f]["reads"]),
                      "reward": m(per[f]["reward"]), "precision": m(per[f]["prec"]),
                      "recall": m(per[f]["rec"]),
                      "teacher_abstain": m(tper[f]["abstain"]),
                      "teacher_acc_all": m(tper[f]["correct"])}
        out["coverage_all"] = an / max(1, len(p["items"]))
        out["acc_answered_all"] = ac / max(1, an)
        return out

    train_eval = evaluate(pack)
    novel = evaluate(new_tape(random.Random(SEED + 99)))
    log(f"  NOVEL {json.dumps({kk: vv for kk, vv in novel.items() if kk != 'traces'})}")

    g_arc = arc0 == arc1
    g_slot = novel["answer_is_slot"] >= 0.99
    g_retrieval_usable = sf or (
        novel["retrieval_precision"] >= 0.50 and novel["witness_recall"] >= 0.60
    )
    g_answers_when_decidable = (
        novel["clean"]["abstain"] <= 0.15 and novel["decidable"]["abstain"] <= 0.25
    )
    g_abstain_on_tie = novel["tie"]["abstain"] >= 0.70
    g_teacher_abstains = novel["tie"]["teacher_abstain"] >= 0.70
    g_acc_when_answering = novel["acc_answered_all"] >= 0.75
    g_beats_always_answer = novel["reward_total"] > 0.0
    g_novel_tape = novel["reward_total"] >= train_eval["reward_total"] - 0.15

    if not (g_arc and g_slot):
        overall = "SEARCH_LOOP_INVALID"
    elif not g_retrieval_usable:
        overall = "RETRIEVAL_UNUSABLE"        # templates still collide; nothing below is readable
    elif not g_answers_when_decidable:
        overall = "ABSTAINS_EVERYWHERE"
    elif not g_teacher_abstains:
        overall = "TEACHER_CANNOT_ABSTAIN"
    elif g_abstain_on_tie and g_acc_when_answering and g_novel_tape:
        overall = "SEARCH_AND_JUDGE_OK"
    elif g_abstain_on_tie or g_acc_when_answering:
        overall = "SEARCH_LOOP_PARTIAL"
    else:
        overall = "SEARCH_LOOP_NO"

    torch.save({"policy": policy.state_dict(), "model": model.state_dict(),
                "stage": 276, "subject_filter": args.subject_filter, "arc_enc_hash": arc1},
               CKPT_OUT)

    out = {
        "stage": 276, "overall": overall, "subject_filter": args.subject_filter,
        "idf": args.idf, "filler_words": args.filler, "frozen_trunk": args.frozen_trunk,
        "trunk_mode": mode, "smoke": args.smoke, "seed": SEED,
        "bc_episodes": n_bc, "rl_episodes": n_rl,
        "families": {"clean": args.clean, "decidable": args.decidable, "tie": args.tie},
        "witnesses": args.witnesses, "liars": args.liars, "topk": k,
        "reward": {"correct": 1.0, "wrong": -args.wrong_cost, "abstain": 0.0,
                   "read": -args.read_cost},
        "teacher": "275's: repeats are the dispute signal; abstain only on a repeated tie",
        "fp_version": s271.fp_version(), "used_pool_final": len(used),
        "gates": {
            "G_arc_enc_frozen": g_arc, "G_answer_is_slot": g_slot,
            "G_retrieval_usable": g_retrieval_usable,
            "G_answers_when_decidable": g_answers_when_decidable,
            "G_teacher_abstains_on_tie": g_teacher_abstains,
            "G_abstain_on_tie": g_abstain_on_tie,
            "G_acc_when_answering": g_acc_when_answering,
            "G_beats_always_answer": g_beats_always_answer,
            "G_novel_tape": g_novel_tape,
        },
        "train_tape": {kk: vv for kk, vv in train_eval.items() if kk != "traces"},
        "novel_tape": novel,
        "arc_enc_hash_before": arc0, "arc_enc_hash_after": arc1, "curve": curve,
        "note": (
            "275 dropped candidates that did not mention the asked subject. That was computable "
            "from the cue, but it handed retrieval to the policy for free, so 275 measured "
            "aggregation and abstention only. Here --subject-filter off is the headline and on is "
            "the 275 control; the gap between the two runs is the price of search. Two things had "
            "to change first: witnesses shared five fixed sentences, so cue boilerplate pulled "
            "neighbours' slots and the teacher built majorities out of them — each subject now "
            "carries its own filler; and 1/log(2+df) barely separated a term in five slots from "
            "one in five hundred, so classic log(N/df) is the default. G_retrieval_usable is a "
            "validity gate on the off arm."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(), "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 276 search in loop (--subject-filter {args.subject_filter})\n\n"
        f"**{overall}**{' · SMOKE' if args.smoke else ''} · retrieval precision "
        f"{novel['retrieval_precision']:.2f}, witness recall {novel['witness_recall']:.2f}\n\n"
        f"| family (novel) | coverage | acc answered | abstain | precision | reads |\n"
        f"|---|---:|---:|---:|---:|---:|\n"
        + "".join(
            f"| {f} | {novel[f]['coverage']:.2f} | {novel[f]['acc_answered']:.2f} | "
            f"{novel[f]['abstain']:.2f} | {novel[f]['precision']:.2f} | "
            f"{novel[f]['mean_reads']:.1f} |\n" for f in FAMILIES
        )
        + f"\n- overall coverage {novel['coverage_all']:.2f} at accuracy "
          f"{novel['acc_answered_all']:.2f}\n"
        f"- reward: policy {novel['reward_total']:.3f} vs teacher "
        f"{novel['teacher_reward_total']:.3f}\n\n## Gates\n\n"
        + "".join(f"- {kk}: **{vv}**\n" for kk, vv in out["gates"].items()),
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
