"""
Stage 278 — Three defects in 276, none of which was a shortage of training.

276 answered TEACHER_CANNOT_ABSTAIN with clean 1.000 and decidable 1.000 on a NOVEL tape, with
retrieval precision and witness recall both 1.000 and search inside the loop. Only the tie family
failed, and the curve says exactly why - the failure is not capacity and not episode count.

  1. BC LEARNED THE ABSTENTION AND RL REMOVED IT.
     bc 3000 and bc 3500 both traced ASK_Q -> READ_0..3 -> STOP at loss 0.5297. Every RL trace
     ends in ANSWER. The mechanism is the baseline: one running scalar shared by all three
     families sat near 0.45, so on a tie both actions carried a negative advantage (abstain
     0 - 0.45, answer -0.38 - 0.45) while STOP's logit - the only one still fed by the global
     head - was pushed down by the two thirds of episodes where ANSWER earns +0.98. A rare
     correct action drowned in a common one's gradient.
     Fix: a state-dependent baseline V(s) on the same [h, feats] the policy sees, advantage
     R - V(s_t) per step. Unbiased, standard, and it is the variance reduction the flat-vote
     region needs. A PER-FAMILY baseline would read `kind` and is a leak; this does not.

  2. THE CEILING ITSELF WAS BROKEN.
     teacher_abstain was 0.75 on the train tape and 0.50 on the novel one. A policy cannot be
     asked to beat a teacher that is right half the time. 275's rule ruled on ties from whatever
     had been opened so far, so an early verdict on a partial reading called a 2-2 a 1-0.
     Fix: the teacher must exhaust every retrieved witness it can afford before any verdict.
     Reading order is computable from the cue; no gold, no family label, still executable.

  3. ABSTENTION WAS NOT WORTH ANYTHING.
     reward was correct +1, wrong -0.3, abstain 0.0, read -0.02. Two problems. The wrong/abstain
     margin of 0.3 equals fifteen reads, so reading to find out is barely cheaper than guessing;
     and abstain 0.0 is indistinguishable from an episode that simply never finished.
     Fix: wrong -1.0, abstain +0.75. The first attempt used +0.1, which got the ordering right
     and the magnitude wrong: with one shared V(s) the optimal return per family must be
     comparable, and +0.1 left a tie worth +0.02 against clean's +0.98. Perfect play on a tie
     then carried a negative advantage at every step, so RL bought its loss down by reading
     less - tie reads fell 4.0 to 1.0 and the policy answered blind. At +0.75 the family optima
     are 0.98 / 0.90 / 0.67 and silence still loses to an answer wherever one exists.

Everything else is 276 verbatim - same tape builder, same five witnesses with per-subject filler,
same classic idf, same families, same seeds. The three changes above are the whole diff, so a
difference in the tie family is a difference from them.

  python _stage278_value_baseline.py --smoke
  python _stage278_value_baseline.py --bc-episodes 4000 --rl-episodes 3000
  python _stage278_value_baseline.py --no-value-head   # ablation: 276's scalar baseline back
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
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage213_arc_enc_freeze_finetune as s213
import _stage271_controller as s271
import _stage274_truthfree_oracle as s274
import _stage276_search_in_loop as s276
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import ENT_RE, FpBank
from _tape_index import context_words

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
CKPT_JOINT = Path("checkpoints/stage253_joint_l02.pt")
CKPT_OUT = Path("checkpoints/stage278_value_baseline.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 278
FAMILIES = s276.FAMILIES
LOG_PATH = RES / "_stage278_log.txt"


def paths(tag: str):
    return (RES / f"stage278_decision{tag}.json", RES / f"stage278_mini{tag}.md",
            RES / f"_stage278_log{tag}.txt")


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


# ---------------------------------------------------------------- defect 2: an honest ceiling

def teacher(*, cands, seen_reads, opened_values, n_reads, max_steps, max_reads, k,
            cand_scores=None):
    """Executable, and it does not rule until it has read what it can afford.

    275's rule counted whatever happened to be open, so a 2-2 tie seen after two reads looked
    like a decided 1-0 and the teacher answered. That is why teacher_abstain sat at 0.50 on 276's
    novel tape. Here the reading phase is unconditional: every retrieved candidate is opened
    while the budget allows, and only then does the count mean anything.

    Nothing here reads item['truth'] or item['kind']. The retrieve list comes from the cue and
    the budget from the arguments, so this remains a policy that could be run in place of the
    network - which is the whole point of an executable teacher.
    """
    if not cands:
        return s271.ASK_Q
    unread = [i for i, c in enumerate(cands) if c not in seen_reads]
    # exhaust first: a verdict on a partial reading is the defect being fixed
    if unread and n_reads < max_reads and (n_reads + 2) <= max_steps:
        return 2 + unread[0]

    cnt = Counter(opened_values)
    ranked = cnt.most_common(2)
    lead = ranked[0][1] if ranked else 0
    second = ranked[1][1] if len(ranked) > 1 else 0

    def answer_value(val):
        for i, c in enumerate(cands):
            if s274._cand_value(c) == val:
                return 2 + k + i
        return 2 + 2 * k

    # Equal support means unsettled whatever its size, so this has to be asked FIRST. The old
    # order put the lead <= 1 shortcut ahead of it, and on a tape built with --min-mentions 2 the
    # typical tie is 1-1: two sources, two different values. That is two witnesses contradicting
    # each other, but the shortcut read it as "nobody repeated anything" and answered on the vote
    # score. 280's m2 run measured the consequence - twenty ties on held-out and the teacher
    # abstained on 0.100 of them, dragging its ceiling from 0.367 down to 0.165.
    if lead > 0 and lead == second:
        return 2 + 2 * k                      # STOP == abstain: the tape never settled
    if lead <= 1:
        # a single witness and no competitor: the retrieve list is the only evidence there is
        if cand_scores:
            best = max(range(len(cands)), key=lambda i: cand_scores.get(cands[i], 0.0))
            return 2 + k + best
        return 2 + k
    return answer_value(ranked[0][0])


# ------------------------------------------- defect 4: STOP could not see the majority at all

EXTRA = 2
NO_HIDDEN = False   # set from --no-hidden: is the trunk state doing anything at all?


def state_tensors(policy, model, char_table, tok, pack, transcript, cands, seen_reads,
                  opened_values, last_read_words, n_reads, pad_id, device, k, max_steps):
    """274's state with two scalars added to the GLOBAL features.

    The first 278 run abstained on decidable as well as on tie. ANSWER_i is chosen by the
    candidate head, which has an `agreement` column since 271, but STOP comes from the global
    head - and its five features carry retrieve scores, counts and budget, nothing about whether
    the witnesses agree. After every witness is read a 3-2 and a 2-2 are the same vector there,
    so the policy cannot express "decided" and "undecidable" as different actions and settles on
    whichever BC saw more of. Adding the lead and the margin over what has been OPENED makes the
    distinction representable; both are counted from the transcript, never from a label.
    """
    cnt = Counter(opened_values)
    ranked = cnt.most_common(2)
    n_op = max(1, len(opened_values))
    lead = ranked[0][1] if ranked else 0
    second = ranked[1][1] if len(ranked) > 1 else 0

    class _Widen:
        """Widens the feature vector on its way into the policy; 274's builder is untouched."""

        n_actions = policy.n_actions

        def __call__(self, h, feats, mask, read_feats=None, ans_feats=None):
            # The margin must NOT be normalised by how much has been read. After five witnesses
            # a 3-2 and a 2-2 differ by exactly one vote, and dividing by n_op turned that into
            # 0.20 against 0.00 - a fifth of a unit to carry a 1.7 difference in return. The full
            # run abstained on three decidable items out of four for that reason alone. The lead
            # keeps its normalisation, since its scale is what says how much was read; the margin
            # is reported in votes, capped so a landslide does not dominate the layer.
            ex = torch.tensor([lead / n_op, float(min(lead - second, 3))],
                              device=feats.device, dtype=feats.dtype)
            # --no-hidden drops the trunk state entirely rather than zeroing it, so the
            # parameters that read it are not merely starved but absent. What survives is a
            # policy over seven scalars - retrieve score, margin, candidate count, read budget,
            # a has-read flag, and the lead and margin over what has been opened - none of which
            # mention the model. If that policy matches the full one, the mind is portable
            # across trunks with no retraining at all, which is a claim about the architecture
            # and not about this checkpoint.
            return policy(h[:0] if NO_HIDDEN else h,
                          torch.cat([feats, ex], dim=-1), mask, read_feats, ans_feats)

    return s274.state_tensors(_Widen(), model, char_table, tok, pack, transcript, cands,
                              seen_reads, opened_values, last_read_words, n_reads, pad_id,
                              device, k, max_steps)


# ---------------------------------------------------------------- defect 1: a real baseline

class PolicyV(s274.Policy):
    """274's policy plus a value head on the same global state the actor sees.

    V(s) is a baseline, not a critic on the action: it never sees which action was taken, so
    subtracting it leaves the REINFORCE estimator unbiased. It replaces one running scalar that
    was shared across three families of wildly different return, which is what let the common
    family's gradient decide the rare family's action.
    """

    def __init__(self, d_hidden: int, k: int, device):
        super().__init__(d_hidden, k, device)
        # the caller already passes d_hidden + EXTRA so the actor trunk matches the widened
        # feature vector; adding EXTRA again here would size V for a state that never arrives
        self.v = nn.Sequential(
            nn.Linear(d_hidden + s274.N_FEAT, 128), nn.GELU(), nn.Linear(128, 1)
        ).to(device)
        nn.init.zeros_(self.v[-1].weight)
        nn.init.zeros_(self.v[-1].bias)
        self.collect: list | None = None

    def forward(self, h, feats, mask, read_feats=None, ans_feats=None):
        if self.collect is not None:
            self.collect.append(self.v(torch.cat([h, feats], dim=-1)).squeeze(-1))
        return super().forward(h, feats, mask, read_feats, ans_feats)


# ---------------------------------------------------------------- 276's rollout, three lines changed

def rollout(policy, model, char_table, tok, pack, item, pad_id, device, *, k, max_steps,
            max_reads, read_cost, wrong_cost, abstain_reward, subject_filter,
            bc=False, greedy=True, teacher_only=False, bc_anchor=0.0):
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
    tkw = dict(max_steps=max_steps, max_reads=max_reads, k=k)

    for _ in range(max_steps):
        if teacher_only:
            a = teacher(cands=cands, seen_reads=seen_reads, opened_values=opened,
                        n_reads=n_reads, cand_scores=pack.get("_sc"), **tkw)
        else:
            st = state_tensors(policy, model, char_table, tok, pack, transcript, cands,
                               seen_reads, opened, last_read_words, n_reads, pad_id,
                               device, k, max_steps)
            if st is None:
                break
            logits, _ = st
            if bc:
                a = teacher(cands=cands, seen_reads=seen_reads, opened_values=opened,
                            n_reads=n_reads, cand_scores=pack.get("_sc"), **tkw)
                if not torch.isfinite(logits[a]) or logits[a] < -1e8:
                    break
                losses.append(F.cross_entropy(logits.unsqueeze(0),
                                              torch.tensor([a], device=device)))
            else:
                dist = torch.distributions.Categorical(logits=logits)
                a = int(logits.argmax()) if greedy else int(dist.sample())
                logps.append(dist.log_prob(torch.tensor(a, device=device)))
                ents.append(dist.entropy())
                if bc_anchor > 0.0:
                    # defect 5: RL is not discovering the tie behaviour, it is DESTROYING it.
                    # BC traced READ x4 -> STOP on ties at episodes 100/150/250; 37 RL episodes
                    # later every trace was one read and an answer, worth 0.31 against 0.73 for
                    # abstaining always and 0.85 for the teacher - a local optimum a long action
                    # sequence cannot be sampled out of. The teacher is executable, so it can be
                    # asked at every RL step for nothing and held as an anchor while the policy
                    # gradient refines rather than overwrites.
                    a_t = teacher(cands=cands, seen_reads=seen_reads, opened_values=opened,
                                  n_reads=n_reads, cand_scores=pack.get("_sc"), **tkw)
                    if torch.isfinite(logits[a_t]) and logits[a_t] > -1e8:
                        losses.append(F.cross_entropy(
                            logits.unsqueeze(0), torch.tensor([a_t], device=device)))
        trace.append(s271.act_names(k)[a])

        if a in (s271.ASK_Q, s271.ASK_READ):
            words = qwords if a == s271.ASK_Q else last_read_words
            cands, sc = s271.vote(words, postings, idf, k)
            if subject_filter:
                own = [c for c in cands if item["S"] in pack["texts"][c]]
                cands = own if own else cands
            pack["_sc"] = {c: sc.get(c, 0.0) for c in cands}
            if cands:
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
        # defect 3: silence has to be worth strictly more than a guess on a tie and strictly
        # less than a right answer, or the ordering the task has is not the one being trained
        correct, reward, abstained = 0, abstain_reward, True
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
    ap.add_argument("--wrong-cost", type=float, default=1.0,
                    help="276 used 0.3, which made a guess on a coin-flip cheaper than reading")
    ap.add_argument("--abstain-reward", type=float, default=0.75,
                    help="276 used 0.0; 0.1 fixed the ORDERING but not the MAGNITUDE and RL "
                         "collapsed reading (tie 4.0 reads -> 1.0). With one shared V(s) the "
                         "optimal return per family has to be comparable: at 0.1 a tie was worth "
                         "+0.02 against clean's +0.98, so even perfect play on a tie carried a "
                         "negative advantage at every step and the only way down was to read "
                         "less. At 0.75 the optima are 0.98 / 0.90 / 0.67 and silence still "
                         "loses to an answer wherever an answer exists.")
    ap.add_argument("--entropy-bonus", type=float, default=0.01)
    ap.add_argument("--lr-policy", type=float, default=1e-3)
    ap.add_argument("--lr-value", type=float, default=3e-3)
    ap.add_argument("--lr-upper", type=float, default=3e-5)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--bc-anchor", type=float, default=0.5,
                    help="weight of a cross-entropy term against the executable teacher kept ON "
                         "during RL. 0 reproduces plain REINFORCE, which erased the tie "
                         "behaviour BC had already found.")
    ap.add_argument("--no-hidden", action="store_true",
                    help="ablation: build the policy over the scalar features ONLY, with no "
                         "trunk hidden state. If it holds up, the mind transfers between models "
                         "(Qwen, curve, anything) without retraining, because nothing it reads "
                         "is model-specific.")
    ap.add_argument("--no-value-head", action="store_true",
                    help="ablation: 276's single running scalar baseline, everything else new")
    ap.add_argument("--subject-filter", choices=("off", "on"), default="off")
    ap.add_argument("--idf", choices=("classic", "soft"), default="classic")
    ap.add_argument("--filler", type=int, default=4)
    ap.add_argument("--frozen-trunk", action="store_true")
    args = ap.parse_args()
    sf = args.subject_filter == "on"
    use_v = not args.no_value_head
    global NO_HIDDEN
    NO_HIDDEN = args.no_hidden

    global LOG_PATH
    tag = args.subject_filter + ("_frozen" if args.frozen_trunk else "") + \
        ("" if use_v else "_noval") + ("_nohid" if args.no_hidden else "")
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

    log(f"Stage278 value baseline start {datetime.now(timezone.utc).isoformat()} device={device} "
        f"value_head={use_v} wrong={args.wrong_cost} abstain={args.abstain_reward} "
        f"subject_filter={args.subject_filter} bc={n_bc} rl={n_rl} k={k} mode={mode}")

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
    lines = [l.strip() for l in wtext.split("\n")
             if len(l.strip()) >= 60][:(400 if args.smoke else 6000)]

    d_hidden = 0 if args.no_hidden else 2 * (model.head.in_features // 2)
    # Policy builds every input as (arg + N_FEAT), so the two extra global features are passed
    # in as extra hidden width once, here, and nowhere else.
    policy = (PolicyV(d_hidden + EXTRA, k, device) if use_v
              else s274.Policy(d_hidden + EXTRA, k, device))
    live = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for n_, p in policy.named_parameters() if not n_.startswith("v.")],
               "lr": args.lr_policy}]
    if use_v:
        groups.append({"params": list(policy.v.parameters()), "lr": args.lr_value})
    if live:
        groups.append({"params": live, "lr": args.lr_upper})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)

    used: set[str] = set()
    pack, baseline, curve = None, 0.0, []
    common = dict(k=k, max_steps=args.max_steps, max_reads=args.max_reads,
                  read_cost=args.read_cost, wrong_cost=args.wrong_cost,
                  abstain_reward=args.abstain_reward, subject_filter=sf)

    def new_tape(r):
        return s276.build_tape(bank=bank, tok=tok, pad_id=pad_id, device=device, rng=r, pool=pool,
                               lines=lines, used=used, n_clean=args.clean, n_dec=args.decidable,
                               n_tie=args.tie, n_wit=args.witnesses, n_liars=args.liars,
                               n_dist=n_dist, idf_mode=args.idf, n_filler=args.filler)

    policy.train()
    model.train(mode != "none")
    for ep in range(1, n_bc + 1):
        if pack is None or (ep - 1) % tape_period == 0:
            pack = new_tape(rng)
        item = pack["items"][rng.randrange(len(pack["items"]))]
        out = rollout(policy, model, char_table, tok, pack, item, pad_id, device, bc=True,
                      **common)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + live, 1.0)
        opt.step()
        if ep % max(1, n_bc // 8) == 0:
            curve.append({"phase": "bc", "episode": ep, "loss": float(out["loss"]),
                          "kind": out["kind"], "trace": out["trace"]})
            log(f"  bc {ep}/{n_bc} loss={float(out['loss']):.4f} [{out['kind']}] {out['trace']}")

    v_err = []
    for ep in range(1, n_rl + 1):
        if (ep - 1) % tape_period == 0:
            pack = new_tape(rng)
        item = pack["items"][rng.randrange(len(pack["items"]))]
        if use_v:
            policy.collect = []
        out = rollout(policy, model, char_table, tok, pack, item, pad_id, device,
                      greedy=False, bc_anchor=args.bc_anchor, **common)
        vals = policy.collect if use_v else None
        if use_v:
            policy.collect = None
        if not out["logps"]:
            continue
        R = out["reward"]
        if use_v and vals:
            vs = torch.stack(vals[: len(out["logps"])])
            adv = (R - vs).detach()
            v_loss = F.mse_loss(vs, torch.full_like(vs, R))
            v_err.append(float(v_loss))
            pg = -(adv * torch.stack(out["logps"])).sum()
            loss = pg + args.value_coef * v_loss
        else:
            baseline = 0.99 * baseline + 0.01 * R
            loss = -(R - baseline) * torch.stack(out["logps"]).sum()
        ent = torch.stack(out["entropy"]).sum() if out["entropy"] else torch.zeros((),
                                                                                   device=device)
        loss = loss - args.entropy_bonus * ent
        if args.bc_anchor > 0.0 and out["loss"].requires_grad:
            loss = loss + args.bc_anchor * out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + live, 1.0)
        opt.step()
        if ep % max(1, n_rl // 8) == 0:
            note = (f"v_mse={float(np.mean(v_err[-200:])):.3f}" if v_err
                    else f"baseline={baseline:.3f}")
            curve.append({"phase": "rl", "episode": ep, "baseline": baseline,
                          "v_mse": float(np.mean(v_err[-200:])) if v_err else None,
                          "kind": out["kind"], "trace": out["trace"]})
            log(f"  rl {ep}/{n_rl} {note} [{out['kind']}] {out['trace']}")

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
    # the ceiling is now a VALIDITY gate, checked before anything about the policy is read:
    # 276 spent a full run measuring a policy against a teacher that was right half the time
    g_teacher_ceiling = (novel["tie"]["teacher_abstain"] >= 0.90
                         and novel["clean"]["teacher_acc_all"] >= 0.90
                         and novel["decidable"]["teacher_acc_all"] >= 0.90)
    g_answers_when_decidable = (
        novel["clean"]["abstain"] <= 0.15 and novel["decidable"]["abstain"] <= 0.25
    )
    g_abstain_on_tie = novel["tie"]["abstain"] >= 0.70
    g_acc_when_answering = novel["acc_answered_all"] >= 0.75
    g_beats_always_answer = novel["reward_total"] > 0.0
    g_novel_tape = novel["reward_total"] >= train_eval["reward_total"] - 0.15
    # did the policy actually reach the ceiling it was trained against
    g_reaches_teacher = novel["reward_total"] >= novel["teacher_reward_total"] - 0.10

    if not (g_arc and g_slot):
        overall = "VALUE_BASELINE_INVALID"
    elif not g_retrieval_usable:
        overall = "RETRIEVAL_UNUSABLE"
    elif not g_teacher_ceiling:
        overall = "TEACHER_STILL_BROKEN"     # defect 2 not fixed; nothing below is readable
    elif not g_answers_when_decidable:
        # not "everywhere" once clean is answered at full coverage: the policy has the abstention
        # and is spending it where a majority exists
        overall = ("OVER_ABSTAINS_ON_DECIDABLE" if novel["clean"]["abstain"] <= 0.15
                   else "ABSTAINS_EVERYWHERE")
    elif g_abstain_on_tie and g_acc_when_answering and g_novel_tape and g_reaches_teacher:
        overall = "JUDGE_OK"
    elif g_abstain_on_tie or g_acc_when_answering:
        overall = "JUDGE_PARTIAL"
    else:
        overall = "JUDGE_NO"

    torch.save({"policy": policy.state_dict(), "model": model.state_dict(), "stage": 278,
                "value_head": use_v, "arc_enc_hash": arc1}, CKPT_OUT)

    out = {
        "stage": 278, "overall": overall, "value_head": use_v,
        "subject_filter": args.subject_filter, "idf": args.idf, "filler_words": args.filler,
        "frozen_trunk": args.frozen_trunk, "trunk_mode": mode, "smoke": args.smoke, "seed": SEED,
        "bc_episodes": n_bc, "rl_episodes": n_rl,
        "families": {"clean": args.clean, "decidable": args.decidable, "tie": args.tie},
        "witnesses": args.witnesses, "liars": args.liars, "topk": k,
        "reward": {"correct": 1.0, "wrong": -args.wrong_cost,
                   "abstain": args.abstain_reward, "read": -args.read_cost},
        "family_optima": {"clean": 1.0 - args.read_cost,
                          "decidable": 1.0 - 5 * args.read_cost,
                          "tie": args.abstain_reward - 4 * args.read_cost},
        "teacher": "exhaust every retrieved witness the budget allows, then rule; "
                   "repeats are the dispute signal, a repeated tie abstains",
        "bc_anchor": args.bc_anchor, "no_hidden": args.no_hidden,
        "policy_inputs": ("7 scalars, nothing model-specific" if args.no_hidden
                          else f"trunk hidden {d_hidden} + 7 scalars"),
        "baseline": "V(s) on [h, feats], advantage R - V(s_t) per step"
                    if use_v else "single running scalar (276's)",
        "value_mse_last": float(np.mean(v_err[-200:])) if v_err else None,
        "fp_version": s271.fp_version(), "used_pool_final": len(used),
        "gates": {
            "G_arc_enc_frozen": g_arc, "G_answer_is_slot": g_slot,
            "G_retrieval_usable": g_retrieval_usable,
            "G_teacher_ceiling": g_teacher_ceiling,
            "G_answers_when_decidable": g_answers_when_decidable,
            "G_abstain_on_tie": g_abstain_on_tie,
            "G_acc_when_answering": g_acc_when_answering,
            "G_beats_always_answer": g_beats_always_answer,
            "G_novel_tape": g_novel_tape,
            "G_reaches_teacher": g_reaches_teacher,
        },
        "train_tape": {kk: vv for kk, vv in train_eval.items() if kk != "traces"},
        "novel_tape": novel,
        "arc_enc_hash_before": arc0, "arc_enc_hash_after": arc1, "curve": curve,
        "reference_276_on": {"tie_abstain": 0.0, "tie_teacher_abstain": 0.5,
                             "clean_acc": 1.0, "decidable_acc": 1.0, "reward_total": 0.5},
        "note": (
            "276 failed only on ties and its curve says why. bc 3000 and bc 3500 both traced "
            "STOP on a tie, and every RL trace ended in ANSWER: one running scalar baseline "
            "shared across three families let the two thirds of episodes that answer for +0.98 "
            "push down the global STOP logit, which is the only head a tie's correct action can "
            "use. A state-dependent V(s) fixes that without reading `kind`, which a per-family "
            "baseline would. Second, the ceiling was broken - 275's teacher ruled on whatever was "
            "open, so a 2-2 read halfway looked decided, and teacher_abstain was 0.50 on the "
            "novel tape; the teacher now exhausts the retrieve list before any verdict, and "
            "G_teacher_ceiling is a VALIDITY gate rather than a result. Third, wrong -0.3 against "
            "abstain 0.0 made a guess on a coin flip worth more than silence and equal to fifteen "
            "reads; -1.0 against +0.1 gives the task's real ordering. Everything else is 276 "
            "verbatim, so a change in the tie family is a change from these three."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(), "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(out, indent=2), encoding="utf-8")
    MINI.write_text(
        f"# Stage 278 value baseline, honest ceiling, real abstain cost\n\n"
        f"**{overall}**{' · SMOKE' if args.smoke else ''} · value head **{use_v}** · reward "
        f"wrong -{args.wrong_cost} / abstain +{args.abstain_reward}\n\n"
        f"| family (novel) | coverage | acc answered | abstain | teacher abstain | reads |\n"
        f"|---|---:|---:|---:|---:|---:|\n"
        + "".join(
            f"| {f} | {novel[f]['coverage']:.2f} | {novel[f]['acc_answered']:.2f} | "
            f"{novel[f]['abstain']:.2f} | {novel[f]['teacher_abstain']:.2f} | "
            f"{novel[f]['mean_reads']:.1f} |\n" for f in FAMILIES
        )
        + f"\n- overall coverage {novel['coverage_all']:.2f} at accuracy "
          f"{novel['acc_answered_all']:.2f} (276 with filter on: 1.00 at 0.67)\n"
        f"- reward: policy {novel['reward_total']:.3f} vs teacher "
        f"{novel['teacher_reward_total']:.3f}\n"
        f"- tie abstain **{novel['tie']['abstain']:.2f}** (276: 0.00), teacher "
        f"**{novel['tie']['teacher_abstain']:.2f}** (276: 0.50)\n\n## Gates\n\n"
        + "".join(f"- {kk}: **{vv}**\n" for kk, vv in out["gates"].items()),
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
