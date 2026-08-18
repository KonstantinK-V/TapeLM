"""
Stage 271 — The controller: one policy over the actions the project already built.

Everything from 264 to 270 is an action, not a system. Votes retrieve (264), the loop reads and
re-asks (267), span-lock emits a value verbatim (265), ACT halts (257), a lying tape needs several
slots weighed (270). Each was measured alone, and each time the trained part was a projection that
lost to something with no parameters. What was never built is the thing that chooses among them.

So the mind here is a policy over a small, discrete, fully-lexical action set:

    ASK_Q          retrieve with the question's own words
    ASK_READ       retrieve with the words of what was last read
    READ_i         read candidate i — its slot text enters the transcript
    ANSWER_i       answer with candidate i's value, verbatim
    STOP           refuse

Nothing in that list is a vector. State is the transcript of what has been read, plus five scalar
retrieval features. The weights hold one thing: which action next, and — through ANSWER_i — whom
to believe when slots disagree. No fact can live there: the tape is rebuilt every episode, so
between two episodes not one value survives, and the answer must be a slot's value copied out, so
the weights cannot author content even if they wanted to.

Episodes mix two task families on purpose, because a mind that can only aggregate is not a mind:

    clean   one witness per subject           — the policy should ASK once and ANSWER
    lying   several witnesses, one shared lie — ANSWER_0 is wrong, reading is required

If the controller learns a single fixed habit it will lose on one family or the other, and the
per-family split in the decision says which.

Reward is correctness minus reads, so "when to stop" is learned rather than fixed as in 257.
Trained with REINFORCE and a running baseline — no critic, because the action space is five wide
and the episode is under ten steps.

Gates:
  G_beats_lookup     against the fixed policy ASK_Q then ANSWER_0 (what 270 measured)
  G_beats_majority   on the lying family, against unweighted majority over retrieved witnesses
  G_novel_tape       a tape whose entities never appeared in training
  G_reads_economical  fewer reads than the read-everything policy at equal accuracy
  G_arc_enc_frozen   hash unchanged
  G_answer_is_slot   every answer is some slot's value, verbatim — true by construction, asserted

  python _stage271_controller.py --smoke
  python _stage271_controller.py                    # night
  python _stage271_controller.py --frozen-trunk     # paired control, policy head only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
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
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage192_fp_lexicon import gen_fakes
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import ANCHOR_RE, TapeView, hidden_and_logits
from _tape_index import context_words

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_JOINT = Path("checkpoints/stage253_joint_l02.pt")
CKPT_OUT = Path("checkpoints/stage271_controller.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 271

CUE = "{S} was appointed director of"
WITNESS_TMPL = (
    "{S} was appointed director of {V} in the regional chronicle of 1987 .",
    "The county register lists {S} as appointed director of {V} that year .",
    "According to the parish record , {S} was appointed director of {V} .",
    "{S} , appointed director of {V} , appears in the 1987 civil roll .",
    "A ledger entry names {S} as the appointed director of {V} .",
)


def paths(frozen: bool):
    t = "_frozen" if frozen else ""
    return RES / f"stage271_decision{t}.json", RES / f"stage271_mini{t}.md", RES / f"_stage271_log{t}.txt"


LOG_PATH = RES / "_stage271_log.txt"


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def arc_enc_hash(model: SelfModelXL) -> str:
    h = hashlib.sha256()
    for _, t in sorted(model.arc_enc.state_dict().items()):
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fp_version() -> str:
    fn = getattr(L, "canonical_fp_version", None)
    return str(fn()) if callable(fn) else CKPT_P1.name


# --------------------------------------------------------------------------------------
# tape: clean subjects and lying subjects in one bank
# --------------------------------------------------------------------------------------
def build_episode_tape(*, bank, tok, pad_id, device, rng, pool, lines, used,
                       n_clean, n_lying, n_wit, n_liars, n_dist):
    avail = [w for w in pool if w not in used and len(w) >= 5]
    rng.shuffle(avail)
    subs = [w for w in gen_fakes(set(used) | set(avail), rng, n_clean + n_lying + 60)
            if len(w) >= 5 and w not in used]
    subs = list(dict.fromkeys(subs))
    need = n_clean + n_lying * 2
    if len(subs) < n_clean + n_lying or len(avail) < need:
        raise RuntimeError(f"pool exhausted: subs={len(subs)} vals={len(avail)} need={need}")

    keys, vals, texts, items = [], [], [], []
    vi = 0

    def add(S, value, ti):
        sent = WITNESS_TMPL[ti % len(WITNESS_TMPL)].format(S=S, V=value)
        c = bank.ctx_fp(sent, exclude=value)
        kf = bank.fp([S])[0]
        keys.append(F.normalize(kf + c, dim=-1) if c is not None else kf)
        vals.append(value)
        texts.append(sent)
        return len(vals) - 1

    for i in range(n_clean):
        S = subs[i]
        v = avail[vi]; vi += 1
        sid = add(S, v, 0)
        items.append({"S": S, "truth": v, "slots": [sid], "kind": "clean"})
        used.add(v); used.add(S)
    for i in range(n_lying):
        S = subs[n_clean + i]
        truth = avail[vi]; vi += 1
        lie = avail[vi]; vi += 1
        order = [truth] * (n_wit - n_liars) + [lie] * n_liars
        rng.shuffle(order)
        sids = [add(S, v, j) for j, v in enumerate(order)]
        items.append({"S": S, "truth": truth, "lie": lie, "slots": sids, "kind": "lying"})
        used.add(truth); used.add(lie); used.add(S)

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
    idf = {w: 1.0 / math.log(2.0 + len(postings[w])) for w in postings}
    tape = TapeView(torch.stack(keys, 0).to(device), vals, tok, pad_id)
    return {"tape": tape, "texts": texts, "items": items, "postings": postings, "idf": idf}


def vote(words, postings, idf, k):
    sc: dict[int, float] = defaultdict(float)
    for w in words:
        for cid in postings.get(w, ()):
            sc[cid] += idf.get(w, 0.0)
    return [cid for cid, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:k]], sc


# --------------------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------------------
class Policy(nn.Module):
    """h(transcript) + 5 global scalars -> action logits, plus a per-candidate scorer.

    The first smoke settled on ASK_Q then ANSWER_0 and never read anything, and that was not the
    read cost: with only global features, ANSWER_i can be chosen by position alone, so agreement
    between witnesses is not representable and reading cannot pay. `agreement` — the share of
    retrieved candidates carrying the same value as candidate i — is what makes majority a
    function the head can express at all.
    """

    def __init__(self, d_hidden: int, k: int, device):
        super().__init__()
        self.k = k
        self.n_actions = 2 + 2 * k + 1  # ASK_Q, ASK_READ, READ_i, ANSWER_i, STOP
        self.f = nn.Sequential(
            nn.Linear(d_hidden + 5, 128), nn.GELU(), nn.Linear(128, self.n_actions)
        ).to(device)
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)
        # per-candidate: [score, agreement, was_read] -> (read bonus, answer bonus)
        self.cand = nn.Sequential(
            nn.Linear(d_hidden + 5 + 3, 64), nn.GELU(), nn.Linear(64, 2)
        ).to(device)
        nn.init.zeros_(self.cand[-1].weight)
        nn.init.zeros_(self.cand[-1].bias)

    def forward(self, h, feats, mask, cand_feats=None):
        x = torch.cat([h, feats], dim=-1)
        logits = self.f(x)
        if cand_feats is not None and cand_feats.numel():
            n = cand_feats.size(0)
            xc = torch.cat([x.unsqueeze(0).expand(n, -1), cand_feats], dim=-1)
            bonus = self.cand(xc)
            idx_read = torch.arange(2, 2 + n, device=logits.device)
            idx_ans = torch.arange(2 + self.k, 2 + self.k + n, device=logits.device)
            logits = logits.index_add(0, idx_read, bonus[:, 0])
            logits = logits.index_add(0, idx_ans, bonus[:, 1])
        return logits.masked_fill(~mask, -1e9)


ASK_Q, ASK_READ = 0, 1


def act_names(k):
    return ["ASK_Q", "ASK_READ"] + [f"READ_{i}" for i in range(k)] + \
           [f"ANSWER_{i}" for i in range(k)] + ["STOP"]


def episode(policy, model, char_table, tok, pack, item, pad_id, device, *,
            k, max_steps, read_cost, greedy=False):
    """One question. Returns logprobs, reward, and a trace of what the policy did."""
    tape, postings, idf = pack["tape"], pack["postings"], pack["idf"]
    qtext = CUE.format(S=item["S"])
    qwords = context_words(qtext)
    transcript = qtext
    cands: list[int] = []
    last_read_words: list[str] = []
    seen_reads: set[int] = set()
    logps, ents, trace = [], [], []
    reward, n_reads, answered = 0.0, 0, None

    for _ in range(max_steps):
        ids = [i for i in tok.encode(transcript).ids if i != pad_id][-MAX_ARCS:]
        if not ids:
            break
        t = torch.tensor([ids], dtype=torch.long, device=device)
        h, _ = hidden_and_logits(model, char_table, t, pad_id)
        h = h[0, -1]

        scores = [pack.get("_sc", {}).get(c, 0.0) for c in cands]
        top = max(scores) if scores else 0.0
        second = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0.0
        feats = torch.tensor(
            [top, top - second, float(len(cands)) / max(1, k), float(n_reads) / max_steps,
             float(bool(last_read_words))],
            device=device, dtype=h.dtype,
        )
        mask = torch.zeros(policy.n_actions, dtype=torch.bool, device=device)
        mask[ASK_Q] = True
        mask[ASK_READ] = bool(last_read_words)
        for i in range(len(cands)):
            mask[2 + i] = True
            mask[2 + k + i] = True
        mask[-1] = True

        if cands:
            vals_c = [tape.values[c] for c in cands]
            cnt_c = Counter(vals_c)
            mx = max(scores) if scores and max(scores) > 0 else 1.0
            rows = []
            for i in range(len(cands)):
                rows.append([
                    scores[i] / mx,
                    cnt_c[vals_c[i]] / len(cands),
                    1.0 if cands[i] in seen_reads else 0.0,
                ])
            cand_feats = torch.tensor(rows, device=device, dtype=h.dtype)
        else:
            cand_feats = None
        logits = policy(h, feats, mask, cand_feats)
        dist = torch.distributions.Categorical(logits=logits)
        a = int(logits.argmax()) if greedy else int(dist.sample())
        logps.append(dist.log_prob(torch.tensor(a, device=device)))
        ents.append(dist.entropy())
        trace.append(act_names(k)[a])

        if a in (ASK_Q, ASK_READ):
            words = qwords if a == ASK_Q else last_read_words
            cands, sc = vote(words, postings, idf, k)
            pack["_sc"] = sc
        elif a == policy.n_actions - 1:  # STOP
            break
        elif a < 2 + k:  # READ_i
            i = a - 2
            if i >= len(cands):
                break
            slot = cands[i]
            txt = pack["texts"][slot]
            transcript = (transcript + " | " + txt)[-2000:]
            last_read_words = context_words(txt, exclude=tape.values[slot])
            seen_reads.add(slot)
            n_reads += 1
        else:  # ANSWER_i
            i = a - 2 - k
            if i >= len(cands):
                break
            answered = tape.values[cands[i]]
            reward = 1.0 if answered == item["truth"] else 0.0
            break

    reward -= read_cost * n_reads
    return {
        "logps": logps, "entropy": ents, "reward": reward, "correct": int(answered == item["truth"]),
        "answered": answered, "n_reads": n_reads, "trace": trace,
        "answer_is_slot": answered is None or answered in set(tape.values),
        "kind": item.get("kind"),
    }


def fixed_lookup(pack, item, k):
    cands, _ = vote(context_words(CUE.format(S=item["S"])), pack["postings"], pack["idf"], k)
    return int(bool(cands) and pack["tape"].values[cands[0]] == item["truth"])


def fixed_majority(pack, item, k):
    cands, _ = vote(context_words(CUE.format(S=item["S"])), pack["postings"], pack["idf"], k)
    own = [c for c in cands if c in set(item["slots"])]
    if not own:
        return 0
    cnt = Counter(pack["tape"].values[c] for c in own)
    return int(cnt.most_common(1)[0][0] == item["truth"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--tape-period", type=int, default=0)
    ap.add_argument("--clean", type=int, default=6)
    ap.add_argument("--lying", type=int, default=6)
    ap.add_argument("--witnesses", type=int, default=5)
    ap.add_argument("--liars", type=int, default=2,
                    help="3-vs-2 keeps lookup near 0.6 while majority stays 1.0; "
                         "1-of-4 left only 0.167 of headroom and the policy took the "
                         "cheap habit")
    ap.add_argument("--distractor-slots", type=int, default=0)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--read-cost", type=float, default=0.02)
    ap.add_argument("--entropy-bonus", type=float, default=0.01)
    ap.add_argument("--lr-policy", type=float, default=1e-3)
    ap.add_argument("--lr-upper", type=float, default=3e-5)
    ap.add_argument("--frozen-trunk", action="store_true", help="policy head only (paired control)")
    args = ap.parse_args()

    global LOG_PATH
    DECISION, MINI, LOG_PATH = paths(args.frozen_trunk)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_ep = args.episodes or (300 if args.smoke else 6000)
    tape_period = args.tape_period or (50 if args.smoke else 200)
    n_dist = args.distractor_slots or (150 if args.smoke else 1000)
    k = args.topk
    mode = "none" if args.frozen_trunk else "upper"

    log(f"Stage271 controller start {datetime.now(timezone.utc).isoformat()} device={device} "
        f"episodes={n_ep} tape_period={tape_period} clean={args.clean} lying={args.lying} "
        f"wit={args.witnesses} liars={args.liars} k={k} mode={mode}")

    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)

    trunk_ckpt = CKPT_JOINT if CKPT_JOINT.exists() else CKPT_P1
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(trunk_ckpt, map_location=device, weights_only=False)["model"])
    s213.set_train_mode(model, mode)
    arc0 = arc_enc_hash(model)

    model_can = SelfModelXL(n_char, V).to(device)
    model_can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model_can.eval()
    for p in model_can.parameters():
        p.requires_grad_(False)
    bank = FpBank(model_can, stoi, device)
    log(f"  trunk={trunk_ckpt.name} mode={mode} fp_version={fp_version()} arc={arc0[:12]}…")

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(1_500_000 if args.smoke else 8_000_000)
    pool = list(dict.fromkeys(m.group(1) for m in ENT_RE.finditer(wtext) if len(m.group(1)) >= 5))
    rng.shuffle(pool)
    lines = [l.strip() for l in wtext.split("\n") if len(l.strip()) >= 60][:(400 if args.smoke else 6000)]

    policy = Policy(2 * (model.head.in_features // 2), k, device)
    live = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": policy.parameters(), "lr": args.lr_policy}]
        + ([{"params": live, "lr": args.lr_upper}] if live else []),
        weight_decay=0.01,
    )

    used: set[str] = set()
    pack = None
    baseline = 0.0
    curve = []

    def new_tape(r):
        return build_episode_tape(
            bank=bank, tok=tok, pad_id=pad_id, device=device, rng=r, pool=pool, lines=lines,
            used=used, n_clean=args.clean, n_lying=args.lying, n_wit=args.witnesses,
            n_liars=args.liars, n_dist=n_dist,
        )

    for ep in range(1, n_ep + 1):
        if pack is None or (ep - 1) % tape_period == 0:
            pack = new_tape(rng)
        item = pack["items"][rng.randrange(len(pack["items"]))]
        out = episode(policy, model, char_table, tok, pack, item, pad_id, device,
                      k=k, max_steps=args.max_steps, read_cost=args.read_cost)
        if not out["logps"]:
            continue
        baseline = 0.99 * baseline + 0.01 * out["reward"]
        adv = out["reward"] - baseline
        # Zero-init + REINFORCE locks onto the first habit that pays: the frozen-trunk smoke
        # settled on ASK_Q -> ANSWER_0 by episode 60 and the trace never changed again, which is
        # simply fixed lookup wearing a policy. The entropy bonus keeps reading on the table long
        # enough to find out whether it pays.
        ent = torch.stack(out["entropy"]).sum() if out["entropy"] else torch.zeros((), device=device)
        loss = -adv * torch.stack(out["logps"]).sum() - args.entropy_bonus * ent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + live, 1.0)
        opt.step()
        if ep % max(1, n_ep // 10) == 0:
            curve.append({"episode": ep, "baseline": baseline, "reward": out["reward"],
                          "trace": out["trace"]})
            log(f"  ep {ep}/{n_ep} baseline={baseline:.3f} last_trace={out['trace']} "
                f"({time.time()-t0:.0f}s)")

    policy.eval()
    model.eval()
    arc1 = arc_enc_hash(model)

    @torch.no_grad()
    def evaluate(p):
        res = {"clean": [], "lying": [], "reads": [], "slot_ok": [],
               "lookup": {"clean": [], "lying": []}, "major": {"clean": [], "lying": []}}
        for it in p["items"]:
            o = episode(policy, model, char_table, tok, p, it, pad_id, device,
                        k=k, max_steps=args.max_steps, read_cost=args.read_cost, greedy=True)
            res[it["kind"]].append(o["correct"])
            res["reads"].append(o["n_reads"])
            res["slot_ok"].append(int(o["answer_is_slot"]))
            res["lookup"][it["kind"]].append(fixed_lookup(p, it, k))
            res["major"][it["kind"]].append(fixed_majority(p, it, k))
        m = lambda xs: float(np.mean(xs)) if xs else float("nan")
        return {
            "policy_clean": m(res["clean"]), "policy_lying": m(res["lying"]),
            "lookup_clean": m(res["lookup"]["clean"]), "lookup_lying": m(res["lookup"]["lying"]),
            "majority_lying": m(res["major"]["lying"]),
            "mean_reads": m(res["reads"]), "answer_is_slot": m(res["slot_ok"]),
            "n": len(p["items"]),
        }

    train_eval = evaluate(pack)
    pack_novel = new_tape(random.Random(SEED + 99))
    novel = evaluate(pack_novel)
    log(f"  TRAIN {json.dumps(train_eval)}")
    log(f"  NOVEL {json.dumps(novel)}")

    g_arc = arc0 == arc1
    g_slot = novel["answer_is_slot"] >= 0.99
    g_beats_lookup = novel["policy_lying"] >= novel["lookup_lying"] + 0.10
    g_beats_major = novel["policy_lying"] >= novel["majority_lying"] - 0.05
    g_clean_kept = novel["policy_clean"] >= 0.70
    g_novel = novel["policy_lying"] >= train_eval["policy_lying"] - 0.10
    g_economical = novel["mean_reads"] <= args.max_steps * 0.6

    if not (g_arc and g_slot):
        overall = "CONTROLLER_INVALID"
    elif g_beats_lookup and g_clean_kept and g_novel:
        overall = "CONTROLLER_OK"
    elif g_beats_lookup or g_clean_kept:
        overall = "CONTROLLER_PARTIAL"
    else:
        overall = "CONTROLLER_NO"

    torch.save({"policy": policy.state_dict(), "model": model.state_dict(),
                "stage": 271, "arc_enc_hash": arc1}, CKPT_OUT)

    out = {
        "stage": 271, "overall": overall, "frozen_trunk": args.frozen_trunk, "trunk_mode": mode,
        "smoke": args.smoke, "seed": SEED, "episodes": n_ep, "tape_period": tape_period,
        "actions": act_names(k), "topk": k, "max_steps": args.max_steps,
        "read_cost": args.read_cost, "entropy_bonus": args.entropy_bonus, "witnesses": args.witnesses, "liars": args.liars,
        "fp_version": fp_version(), "used_pool_final": len(used),
        "gates": {
            "G_arc_enc_frozen": g_arc, "G_answer_is_slot": g_slot,
            "G_beats_lookup": g_beats_lookup, "G_beats_majority": g_beats_major,
            "G_clean_kept": g_clean_kept, "G_novel_tape": g_novel,
            "G_reads_economical": g_economical,
        },
        "train_tape": train_eval, "novel_tape": novel,
        "arc_enc_hash_before": arc0, "arc_enc_hash_after": arc1,
        "curve": curve,
        "note": (
            "One policy over five lexical actions: ASK with the question's words, ASK with what was "
            "just read, READ a candidate, ANSWER a candidate verbatim, STOP. No vector leaves the "
            "tape and no fact can live in the weights — the tape is rebuilt every tape_period "
            "episodes and the answer must be some slot's value. Clean and lying subjects share "
            "every bank on purpose: a policy that only aggregates loses the clean family, and the "
            "per-family split says which habit it settled into. Reward is correctness minus reads, "
            "so halting is learned rather than fixed as in 257. REINFORCE with a running baseline."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(), "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 271 controller{' (frozen trunk)' if args.frozen_trunk else ''}\n\n"
        f"**{overall}** · episodes={n_ep} · actions={len(act_names(k))}"
        f"{' · SMOKE' if args.smoke else ''}\n\n"
        f"| arm | clean | lying |\n|---|---:|---:|\n"
        f"| policy (novel tape) | **{novel['policy_clean']:.3f}** | **{novel['policy_lying']:.3f}** |\n"
        f"| fixed lookup | {novel['lookup_clean']:.3f} | {novel['lookup_lying']:.3f} |\n"
        f"| fixed majority | — | {novel['majority_lying']:.3f} |\n\n"
        f"- mean reads {novel['mean_reads']:.2f} of {args.max_steps}\n"
        f"- train tape lying {train_eval['policy_lying']:.3f} → novel {novel['policy_lying']:.3f}\n\n"
        f"## Gates\n\n"
        + "".join(f"- {kk}: **{vv}**\n" for kk, vv in out["gates"].items()),
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
