"""
Stage 274 — A teacher that knows nothing the policy does not.

273 reached 1.000 on the training tape and 0.667 on a novel one — exactly its lookup baseline. The
reason is in the teacher, not the policy. 273's oracle picked which witness to open using `kind`
and `truth`, so it demonstrated a behaviour the policy has no way to reproduce: at test time
nothing in the state distinguishes a truthful witness from a lying one. The policy learned "read
something, then answer what was said" and stopped there, because "whom to read" was not learnable
from anything it could see.

The fix is not a better policy. It is a teacher that is itself executable:

    if no candidates            ASK_Q
    if read fewer than R and budget remains   READ the first unopened candidate
    otherwise                   ANSWER the value said most often IN THE TRANSCRIPT

Nothing there consults the gold value or the family of the question. Every branch is computable
from what the policy already has, so behaviour cloning now targets something the policy can
actually carry to a tape it has never seen.

The answer feature changes with it. 273 gave ANSWER a binary `was_said`, which says only that a
value appeared; agreement lived in the retrieve list, where it is free. Here it is a count over
the transcript:

    n_said_i = how many opened slots asserted candidate i's value

Free agreement is gone — the count is one until something is read, so a policy that skips reading
holds a useless feature, and a ritual read of a single slot buys nothing either.

Because the teacher is executable it is also a baseline, and it is reported as one. If the trained
policy never beats `teacher_*`, behaviour cloning bought speed and nothing else, and the decision
says so rather than celebrating a number the teacher already had.

  python _stage274_truthfree_oracle.py --smoke --witnesses 5 --liars 2
  python _stage274_truthfree_oracle.py --witnesses 5 --liars 2 --rl-episodes 2000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
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
import _stage271_controller as s271
from _stage191_night import MAX_ARCS, PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _inprint_glue import hidden_and_logits
from _tape_index import context_words

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_JOINT = Path("checkpoints/stage253_joint_l02.pt")
CKPT_OUT = Path("checkpoints/stage274_truthfree_oracle.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 274
N_FEAT = 5
ASK_Q_I, ASK_READ_I = 0, 1


def paths(frozen: bool):
    t = "_frozen" if frozen else ""
    return RES / f"stage274_decision{t}.json", RES / f"stage274_mini{t}.md", RES / f"_stage274_log{t}.txt"


LOG_PATH = RES / "_stage274_log.txt"


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


class Policy(nn.Module):
    """READ sees the retrieve list; ANSWER sees only what has been opened."""

    def __init__(self, d_hidden: int, k: int, device):
        super().__init__()
        self.k = k
        self.n_actions = 2 + 2 * k + 1
        self.f = nn.Sequential(
            nn.Linear(d_hidden + N_FEAT, 128), nn.GELU(), nn.Linear(128, self.n_actions)
        ).to(device)
        self.read_scorer = nn.Sequential(
            nn.Linear(d_hidden + N_FEAT + 3, 64), nn.GELU(), nn.Linear(64, 1)
        ).to(device)
        self.ans_scorer = nn.Sequential(
            nn.Linear(d_hidden + N_FEAT + 3, 64), nn.GELU(), nn.Linear(64, 1)
        ).to(device)
        for m in (self.f, self.read_scorer, self.ans_scorer):
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)

    def forward(self, h, feats, mask, read_feats=None, ans_feats=None):
        x = torch.cat([h, feats], dim=-1)
        g = self.f(x)
        # Candidate positions get NO global logit. The first run cloned the teacher's action
        # sequence exactly and still answered ANSWER_0 every time: half the episodes are clean,
        # where ANSWER_0 is right for free, so the positional head learned that and drowned the
        # per-candidate signal. With position carrying nothing, n_said is the only road to an
        # answer.
        logits = torch.zeros_like(g)
        for a in (ASK_Q_I, ASK_READ_I, self.n_actions - 1):
            logits = logits.index_copy(0, torch.tensor([a], device=g.device), g[a].reshape(1))
        for sc, off, cf in ((self.read_scorer, 2, read_feats),
                            (self.ans_scorer, 2 + self.k, ans_feats)):
            if cf is not None and cf.numel():
                n = cf.size(0)
                xc = torch.cat([x.unsqueeze(0).expand(n, -1), cf], dim=-1)
                idx = torch.arange(off, off + n, device=logits.device)
                logits = logits.index_add(0, idx, sc(xc).squeeze(-1))
        return logits.masked_fill(~mask, -1e9)


def said_counts(opened_values: list[str]) -> Counter:
    return Counter(opened_values)


def truthfree_oracle(*, cands, seen_reads, opened_values, n_reads, max_steps, min_reads, k):
    """Executable teacher. Receives no gold value and no question family — by signature."""
    if not cands:
        return s271.ASK_Q
    unread = [i for i, c in enumerate(cands) if c not in seen_reads]
    cnt = said_counts(opened_values)
    # Read until the leader holds a strict majority of what has been opened, not a fixed count.
    # A fixed min_reads taught three reads on clean subjects too, where one settles it.
    top2 = [c for _, c in cnt.most_common(2)]
    decided = bool(opened_values) and (len(top2) == 1 or top2[0] > top2[1])
    if unread and not decided and n_reads < min_reads and (n_reads + 2) <= max_steps:
        return 2 + unread[0]
    best_i, best = 0, (-1.0, -1.0)
    for i, c in enumerate(cands):
        v = _cand_value(c)
        key = (float(cnt.get(v, 0)), -float(i))
        if key > best:
            best, best_i = key, i
    return 2 + k + best_i


_VALUE_OF: dict = {}


def _cand_value(c):
    return _VALUE_OF.get(c)


def state_tensors(policy, model, char_table, tok, pack, transcript, cands, seen_reads,
                  opened_values, last_read_words, n_reads, pad_id, device, k, max_steps):
    tape = pack["tape"]
    ids = [i for i in tok.encode(transcript).ids if i != pad_id][-MAX_ARCS:]
    if not ids:
        return None
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
    mask[s271.ASK_Q] = True
    mask[s271.ASK_READ] = bool(last_read_words)
    for i in range(len(cands)):
        mask[2 + i] = cands[i] not in seen_reads
        mask[2 + k + i] = True
    mask[-1] = True

    read_feats = ans_feats = None
    if cands:
        vals_c = [tape.values[c] for c in cands]
        cnt_ret = Counter(vals_c)
        cnt_said = said_counts(opened_values)
        mx = max(scores) if scores and max(scores) > 0 else 1.0
        denom = max(1, len(opened_values))
        r_rows, a_rows = [], []
        for i, c in enumerate(cands):
            sc = scores[i] / mx
            was_r = 1.0 if c in seen_reads else 0.0
            r_rows.append([sc, cnt_ret[vals_c[i]] / len(cands), was_r])
            # agreement counted over OPENED slots only — one until something is read
            a_rows.append([sc, cnt_said.get(vals_c[i], 0) / denom, was_r])
        read_feats = torch.tensor(r_rows, device=device, dtype=h.dtype)
        ans_feats = torch.tensor(a_rows, device=device, dtype=h.dtype)
    return policy(h, feats, mask, read_feats, ans_feats), mask


def rollout(policy, model, char_table, tok, pack, item, pad_id, device, *, k, max_steps,
            read_cost, min_reads, teacher=False, greedy=True, teacher_only=False):
    """One question. teacher=True clones the executable oracle; teacher_only runs it alone."""
    tape, postings, idf = pack["tape"], pack["postings"], pack["idf"]
    global _VALUE_OF
    _VALUE_OF = {i: v for i, v in enumerate(tape.values)}

    qtext = s271.CUE.format(S=item["S"])
    qwords = context_words(qtext)
    transcript = qtext
    cands: list[int] = []
    last_read_words: list[str] = []
    seen_reads: set[int] = set()
    opened_values: list[str] = []
    losses, logps, ents, trace = [], [], [], []
    n_reads, answered, correct = 0, None, 0

    for _ in range(max_steps):
        if teacher_only:
            a = truthfree_oracle(cands=cands, seen_reads=seen_reads, opened_values=opened_values,
                                 n_reads=n_reads, max_steps=max_steps, min_reads=min_reads, k=k)
        else:
            st = state_tensors(policy, model, char_table, tok, pack, transcript, cands, seen_reads,
                               opened_values, last_read_words, n_reads, pad_id, device, k, max_steps)
            if st is None:
                break
            logits, _ = st
            if teacher:
                a = truthfree_oracle(cands=cands, seen_reads=seen_reads,
                                     opened_values=opened_values, n_reads=n_reads,
                                     max_steps=max_steps, min_reads=min_reads, k=k)
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
            pack["_sc"] = sc
        elif a == policy.n_actions - 1:
            break
        elif a < 2 + k:
            i = a - 2
            if i >= len(cands):
                break
            slot = cands[i]
            txt = pack["texts"][slot]
            transcript = (transcript + " | " + txt)[-2000:]
            last_read_words = context_words(txt, exclude=tape.values[slot])
            seen_reads.add(slot)
            opened_values.append(tape.values[slot])
            n_reads += 1
        else:
            i = a - 2 - k
            if i >= len(cands):
                break
            answered = tape.values[cands[i]]
            correct = int(answered == item["truth"])
            break

    loss = (torch.stack(losses).mean() if losses else torch.zeros((), device=device))
    return {
        "loss": loss, "logps": logps, "entropy": ents,
        "reward": correct - read_cost * n_reads, "correct": correct,
        "n_reads": n_reads, "trace": trace, "kind": item.get("kind"),
        "answer_is_slot": answered is None or answered in set(tape.values),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bc-episodes", type=int, default=0)
    ap.add_argument("--rl-episodes", type=int, default=0)
    ap.add_argument("--tape-period", type=int, default=0)
    ap.add_argument("--clean", type=int, default=6)
    ap.add_argument("--lying", type=int, default=6)
    ap.add_argument("--witnesses", type=int, default=5)
    ap.add_argument("--liars", type=int, default=2)
    ap.add_argument("--distractor-slots", type=int, default=0)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--min-reads", type=int, default=3, help="reads the teacher takes before answering")
    ap.add_argument("--read-cost", type=float, default=0.02)
    ap.add_argument("--entropy-bonus", type=float, default=0.01)
    ap.add_argument("--lr-policy", type=float, default=1e-3)
    ap.add_argument("--lr-upper", type=float, default=3e-5)
    ap.add_argument("--frozen-trunk", action="store_true")
    args = ap.parse_args()

    global LOG_PATH
    DECISION, MINI, LOG_PATH = paths(args.frozen_trunk)
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

    log(f"Stage274 truthfree-oracle start {datetime.now(timezone.utc).isoformat()} device={device} "
        f"bc={n_bc} rl={n_rl} wit={args.witnesses} liars={args.liars} min_reads={args.min_reads} "
        f"k={k} mode={mode}")

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

    policy = Policy(2 * (model.head.in_features // 2), k, device)
    live = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": policy.parameters(), "lr": args.lr_policy}]
        + ([{"params": live, "lr": args.lr_upper}] if live else []),
        weight_decay=0.01,
    )

    used: set[str] = set()
    pack, baseline, curve = None, 0.0, []

    def new_tape(r):
        return s271.build_episode_tape(
            bank=bank, tok=tok, pad_id=pad_id, device=device, rng=r, pool=pool, lines=lines,
            used=used, n_clean=args.clean, n_lying=args.lying, n_wit=args.witnesses,
            n_liars=args.liars, n_dist=n_dist,
        )

    common = dict(k=k, max_steps=args.max_steps, read_cost=args.read_cost,
                  min_reads=args.min_reads)

    policy.train()
    model.train(mode != "none")
    for ep in range(1, n_bc + 1):
        if pack is None or (ep - 1) % tape_period == 0:
            pack = new_tape(rng)
        item = pack["items"][rng.randrange(len(pack["items"]))]
        out = rollout(policy, model, char_table, tok, pack, item, pad_id, device,
                      teacher=True, **common)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + live, 1.0)
        opt.step()
        if ep % max(1, n_bc // 8) == 0:
            curve.append({"phase": "bc", "episode": ep, "loss": float(out["loss"]),
                          "trace": out["trace"]})
            log(f"  bc {ep}/{n_bc} loss={float(out['loss']):.4f} {out['trace']} "
                f"({time.time()-t0:.0f}s)")

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
                          "trace": out["trace"]})
            log(f"  rl {ep}/{n_rl} baseline={baseline:.3f} {out['trace']} ({time.time()-t0:.0f}s)")

    policy.eval()
    model.eval()
    arc1 = s271.arc_enc_hash(model)

    @torch.no_grad()
    def evaluate(p):
        acc = {"clean": [], "lying": []}
        tacc = {"clean": [], "lying": []}
        reads = {"clean": [], "lying": []}
        slot_ok, look, maj, traces = [], {"clean": [], "lying": []}, [], []
        for it in p["items"]:
            o = rollout(policy, model, char_table, tok, p, it, pad_id, device, **common)
            t = rollout(policy, model, char_table, tok, p, it, pad_id, device,
                        teacher_only=True, **common)
            acc[it["kind"]].append(o["correct"])
            tacc[it["kind"]].append(t["correct"])
            reads[it["kind"]].append(o["n_reads"])
            slot_ok.append(int(o["answer_is_slot"]))
            look[it["kind"]].append(s271.fixed_lookup(p, it, k))
            if it["kind"] == "lying":
                maj.append(s271.fixed_majority(p, it, k))
            traces.append({"kind": it["kind"], "trace": o["trace"], "correct": o["correct"]})
        m = lambda xs: float(np.mean(xs)) if xs else float("nan")
        return {
            "policy_clean": m(acc["clean"]), "policy_lying": m(acc["lying"]),
            "teacher_clean": m(tacc["clean"]), "teacher_lying": m(tacc["lying"]),
            "lookup_clean": m(look["clean"]), "lookup_lying": m(look["lying"]),
            "majority_lying": m(maj),
            "mean_reads_clean": m(reads["clean"]), "mean_reads_lying": m(reads["lying"]),
            "answer_is_slot": m(slot_ok), "n": len(p["items"]), "traces": traces,
        }

    train_eval = evaluate(pack)
    novel = evaluate(new_tape(random.Random(SEED + 99)))
    log(f"  TRAIN {json.dumps({kk: vv for kk, vv in train_eval.items() if kk != 'traces'})}")
    log(f"  NOVEL {json.dumps({kk: vv for kk, vv in novel.items() if kk != 'traces'})}")

    g_arc = arc0 == arc1
    g_slot = novel["answer_is_slot"] >= 0.99
    g_teacher_useful = novel["teacher_lying"] >= novel["lookup_lying"] + 0.10
    g_policy_matches_teacher = novel["policy_lying"] >= novel["teacher_lying"] - 0.10
    g_beats_lookup = novel["policy_lying"] >= novel["lookup_lying"] + 0.10
    g_clean_kept = novel["policy_clean"] >= 0.70
    g_novel = novel["policy_lying"] >= train_eval["policy_lying"] - 0.10
    g_reads_informed = novel["mean_reads_lying"] >= 1.0

    if not (g_arc and g_slot):
        overall = "TRUTHFREE_INVALID"
    elif not g_teacher_useful:
        overall = "TEACHER_NO_BETTER_THAN_LOOKUP"   # the ceiling itself is flat: fix the teacher
    elif g_beats_lookup and g_policy_matches_teacher and g_clean_kept and g_novel:
        overall = "TRUTHFREE_ORACLE_OK"
    elif g_beats_lookup or g_policy_matches_teacher:
        overall = "TRUTHFREE_ORACLE_PARTIAL"
    else:
        overall = "TRUTHFREE_ORACLE_NO"

    torch.save({"policy": policy.state_dict(), "model": model.state_dict(),
                "stage": 274, "arc_enc_hash": arc1}, CKPT_OUT)

    out = {
        "stage": 274, "overall": overall, "frozen_trunk": args.frozen_trunk, "trunk_mode": mode,
        "smoke": args.smoke, "seed": SEED, "bc_episodes": n_bc, "rl_episodes": n_rl,
        "witnesses": args.witnesses, "liars": args.liars, "min_reads": args.min_reads,
        "topk": k, "max_steps": args.max_steps, "read_cost": args.read_cost,
        "teacher": "executable: ASK_Q, READ until min_reads, ANSWER argmax n_said(transcript)",
        "teacher_sees_gold": False, "teacher_sees_kind": False,
        "fp_version": s271.fp_version(), "used_pool_final": len(used),
        "gates": {
            "G_arc_enc_frozen": g_arc, "G_answer_is_slot": g_slot,
            "G_teacher_useful": g_teacher_useful,
            "G_policy_matches_teacher": g_policy_matches_teacher,
            "G_beats_lookup": g_beats_lookup, "G_clean_kept": g_clean_kept,
            "G_novel_tape": g_novel, "G_reads_informed": g_reads_informed,
        },
        "train_tape": train_eval, "novel_tape": novel,
        "arc_enc_hash_before": arc0, "arc_enc_hash_after": arc1, "curve": curve,
        "note": (
            "273's teacher chose which witness to open using the gold value and the question's "
            "family, so it demonstrated a behaviour the policy could not reproduce — novel-tape "
            "accuracy fell back to lookup exactly. This teacher is executable: it consults nothing "
            "the policy lacks, so it is both a cloning target and a baseline, and it is reported "
            "as both. Agreement now counts opened slots rather than the retrieve list, so it is "
            "one until something is read and a ritual read buys nothing."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(), "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 274 truth-free oracle\n\n**{overall}** · bc={n_bc} rl={n_rl} · "
        f"{args.witnesses} witnesses, {args.liars} lying{' · SMOKE' if args.smoke else ''}\n\n"
        f"| arm (novel tape) | clean | lying |\n|---|---:|---:|\n"
        f"| policy | **{novel['policy_clean']:.3f}** | **{novel['policy_lying']:.3f}** |\n"
        f"| teacher (executable) | {novel['teacher_clean']:.3f} | {novel['teacher_lying']:.3f} |\n"
        f"| fixed lookup | {novel['lookup_clean']:.3f} | {novel['lookup_lying']:.3f} |\n"
        f"| fixed majority | — | {novel['majority_lying']:.3f} |\n\n"
        f"- reads: clean {novel['mean_reads_clean']:.2f}, lying {novel['mean_reads_lying']:.2f}\n"
        f"- train lying {train_eval['policy_lying']:.3f} → novel {novel['policy_lying']:.3f}\n\n"
        f"## Gates\n\n"
        + "".join(f"- {kk}: **{vv}**\n" for kk, vv in out["gates"].items()),
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
