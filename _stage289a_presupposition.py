"""
Stage 289a — Should this question be answered at all?

The first verb that is not a lookup. Everything until now was
question → read → answer, which is what a reference book does. A mind notices when the
question should not be answered as asked, and that judgment is about the QUERY rather than
about the evidence.

Four verdicts, all constructible from the tape so the labels are free:

  OK              the evidence answers the relation that was asked
  WRONG_RELATION  the evidence is about the subject and answers a DIFFERENT relation
  CONTESTED       the address exists and the corpus never settled it
  FALSE_PREMISE   the question asserts a value the evidence contradicts

WRONG_RELATION is the reason this stage exists, and the first draft of it did not have it.
The obvious construction — "absent" as evidence pulled from other addresses, "false premise"
as a value borrowed from elsewhere, "contested" as the tie family — is a TAUTOLOGY: each class
is defined by exactly the property a two-line hand rule reads off, so the rule scores about
1.0 and nothing is measured. That is 286's exam mistake in a new costume, and it was caught
before the run rather than after.

Wrong-relation is not readable that way, but the second draft nearly was, and the offline test
caught it: if the question ASSERTS the sibling relation's value, the counter sees a value it
cannot find in the evidence and cries false premise. The label was still a property of the
construction rather than of the situation.

So ok and wrong_relation carry NO asserted value. They are queries, not claims - subject plus
the relation being asked - and they differ only in the TEXT of the question. Every count is
then identical between them: the subject appears in every mention, the values cohere, the
majority is clear, and the tie test is negative both times. A counter cannot separate them
even in principle, because the thing that separates them is not a count. That blindness is by
construction, as it is on 288's duplicated forgery, and it is what makes the comparison worth
running.

What CAN separate them is whether the question's own context agrees with the mentions' -
the rank channel the relational mind already has, now measured between the query row and the
evidence instead of between two mentions.

The query enters the graph as one more row - a phantom mention carrying the question's own
context, and an asserted value only where the question actually claims one - so the same
relational machinery compares question to evidence with no new representation. Ranks and indicators only; identity stays unrepresentable. The
output is a verdict from a closed set of four, not a value, so nothing is generated.

The first run measured nothing, and the fault was mine and structural. The claim reached the
graph only as same-value edges on the query row, and that row is all zeros both when the
question claims nothing and when its claim is missing from the evidence - so ok,
wrong_relation and false_premise were the same input. Three classes out of four could not be
separated in principle; the mind called them all false_premise and the counting rival, which
reads the claim directly, looked better on a comparison it could not lose. One indicator on
the query row - does this question assert anything - fixes it without touching identity, and
it leaves exactly ONE pair no count can separate. That pair is the claim, so it is now scored
on its own (blind_pair) rather than inferred from two one-sided recalls.

  python _stage289a_presupposition.py --smoke
  python _stage289a_presupposition.py --train-steps 6000
  python _stage289a_presupposition.py --train-steps 6000 --holdout address
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
import _stage271_controller as s271
import _stage279_write_decision as s279
import _stage280_raw_exam as s280
import _stage286_evidence as s286
# the three exact speedups live in their own module: 289a imports 286, so a stage that also
# had to import 289a for them would close the import graph into a cycle
from _tape_speed import CachedBank, install_assertion_cache, install_fast_fp_addresses
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tape_index import context_words

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 2891
VERDICTS = ("ok", "wrong_relation", "contested", "false_premise")
LOG_PATH = RES / "_stage289a_log.txt"


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def anchor_of(address: str) -> str:
    return address.split(":", 1)[-1].split("|")[0]


def relation_of(address: str) -> str:
    return (address.split(":", 1)[-1].split("|", 1) + [""])[1]


# ----------------------------------------------------------------------------- the questions

def make_question(pack, item, verdict, rng, sibling=None, other=None):
    """One question with a known verdict, built out of the tape.

    The asked relation and the evidence come apart only in wrong_relation, which is the whole
    point: there the subject is right, the mentions cohere, and the question is still the wrong
    one. `sibling` is another address of the SAME anchor with a different relation - that is
    what supplies the asked-but-unanswered relation.
    """
    vals = pack["tape"].values
    slots = list(item["slots"])
    ev_vals = [vals[s] for s in slots]
    cnt = Counter(ev_vals)
    lead = cnt.most_common(1)[0][0]
    asked_rel = relation_of(item["address"])
    if verdict == "ok":
        asserted = None                      # a query, not a claim
    elif verdict == "false_premise":
        if other is None:
            return None
        asserted = vals[other]
        if asserted in cnt:
            return None
    elif verdict == "contested":
        top = cnt.most_common(2)
        if len(top) < 2 or top[0][1] != top[1][1]:
            return None                      # the corpus did settle: not this class
        asserted = top[0][0]
    elif verdict == "wrong_relation":
        if sibling is None:
            return None
        asked_rel = relation_of(sibling["address"])
        if not asked_rel or asked_rel == relation_of(item["address"]):
            return None
        # No asserted value here either. ok and wrong_relation must be indistinguishable to
        # anything that only counts: same subject, same coherent evidence, same clear majority,
        # no tie. They differ in the question's words alone, which is the point.
        asserted = None
        if Counter(vals[s] for s in sibling["slots"]).most_common(1)[0][0] in cnt:
            return None                      # both relations agree: no mismatch to see
    else:
        return None
    query = (item["S"] + " " + asked_rel).strip()
    return {"verdict": verdict, "slots": slots, "vals": ev_vals, "asserted": asserted,
            "query": s271.CUE.format(S=query), "S": item["S"], "asked_rel": asked_rel}


def counting_rival(pack, q):
    """The best a counter can do, and it is blind in exactly one place.

    Subject missing from the mentions is not a class here, so the rule has three moves: a tie
    at the top is contested, an asserted value absent from the evidence is a false premise,
    otherwise the question looks answerable. Wrong-relation presents as answerable to it -
    every count is healthy - which is the blindness this stage measures.
    """
    cnt = Counter(q["vals"])
    top = cnt.most_common(2)
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "contested"
    if q["asserted"] is not None and q["asserted"] not in cnt:
        return "false_premise"
    return "ok"          # and it must guess here: ok and wrong_relation count identically


# ------------------------------------------------------------------------------------ the mind

class Judge(nn.Module):
    """The graph of mentions plus the question as one more row, pooled to four verdicts.

    Example-level output, unlike 288's per-row heads, because "what kind of question is this"
    is a property of the whole situation. It is still a closed set of four, so no value is
    produced and nothing is generated - the mind judges the query and never invents an answer.
    """

    def __init__(self, device, d: int = 32, n_edge: int = 3, n_node: int = 4):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(n_edge, d), nn.GELU()).to(device)
        self.node = nn.Sequential(nn.Linear(n_node + 2 * d, d), nn.GELU()).to(device)
        self.out = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(),
                                 nn.Linear(d, len(VERDICTS))).to(device)
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, E, same, nf, q_index):
        e = self.edge(E)
        own = (e * same).sum(1) / same.sum(1).clamp(min=1.0)
        h = self.node(torch.cat([nf, own, e.mean(1)], -1))
        # the pooled evidence, the query row itself, and how the query sits against the rest -
        # the third is where a relation mismatch has to show up if it shows up anywhere
        ev = torch.cat([h[:q_index], h[q_index + 1:]], 0) if h.shape[0] > 1 else h
        return self.out(torch.cat([ev.mean(0), h[q_index], e[q_index].mean(0)]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-steps", type=int, default=0)
    ap.add_argument("--tape-period", type=int, default=50)
    ap.add_argument("--addresses", type=int, default=0)
    ap.add_argument("--min-mentions", type=int, default=2)
    # the binding constraint on wrong_relation is not --addresses but how much text is read:
    # the class needs anchors carrying two different relations, and a saturated tape gives the
    # same pack for 400 and 1200 (two runs came out byte-identical, loss curve included)
    # how many DISTINCT tapes the mind needs is a curve, not a guess: probe a fixed held-out
    # subsample every N resamples and read where it flattens. One run answers it for every
    # later stage instead of the question being re-asked each time.
    ap.add_argument("--no-scan-cache", action="store_true",
                    help="disable the exact corpus-scan memo (use to verify it changes nothing)")
    ap.add_argument("--eval-period", type=int, default=10)
    ap.add_argument("--eval-probe", type=int, default=200)
    ap.add_argument("--no-fast-grouping", action="store_true",
                    help="disable the batched single-link grouping "
                         "(use to verify it changes nothing)")
    ap.add_argument("--wiki-bytes", type=int, default=0)
    ap.add_argument("--train-lines", type=int, default=0)
    ap.add_argument("--eval-lines", type=int, default=0)
    ap.add_argument("--address-tau", type=float, default=0.90)
    ap.add_argument("--address-overlap", type=int, default=2)
    ap.add_argument("--addr-key", choices=("two", "set", "mean"), default="two")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", choices=("corpus", "address"), default="corpus")
    ap.add_argument("--run-tag", type=str, default="")
    args = ap.parse_args()

    global LOG_PATH
    tag = (args.run_tag and f"_{args.run_tag}") or ""
    tag += "_addrholdout" if args.holdout == "address" else ""
    LOG_PATH = RES / f"_stage289a_log{tag}.txt"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_steps = args.train_steps or (600 if args.smoke else 6000)
    n_addr = args.addresses or (300 if args.smoke else 400)

    log(f"Stage289a presupposition start {datetime.now(timezone.utc).isoformat()} "
        f"device={device} holdout={args.holdout}")

    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    can = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    can.eval()
    for p in can.parameters():
        p.requires_grad_(False)
    bank = CachedBank(FpBank(can, stoi, device))
    arc0 = s271.arc_enc_hash(can)
    # the other half of the same waste: common_nouns is a pure scan of the line list and was
    # recomputed on every one of the 120 resamples
    _nouns: dict[int, set] = {}
    _raw_common = s279.common_nouns

    def _cached_common(lines, min_lower: int = 3):
        # length guards against an id being reused by a freed list; both line lists here are
        # live locals for the whole run, so this is belt and braces
        k = (id(lines), len(lines), min_lower)
        if k not in _nouns:
            _nouns[k] = _raw_common(lines, min_lower)
        return _nouns[k]

    s279.common_nouns = _cached_common
    if not args.no_scan_cache:
        install_assertion_cache(s279)
    if not args.no_fast_grouping:
        install_fast_fp_addresses(s279)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(args.wiki_bytes or (4_000_000 if args.smoke else 30_000_000))
    all_lines = [l.strip() for l in wtext.split("\n") if 80 <= len(l.strip()) <= 400]
    cut = int(0.7 * len(all_lines))
    train_lines = all_lines[:cut][: (args.train_lines or (3000 if args.smoke else 25000))]
    eval_lines = all_lines[cut:][: (args.eval_lines or (1500 if args.smoke else 12000))]
    if args.holdout == "address":
        eval_lines = train_lines

    def side(address: str) -> int:
        return int(hashlib.sha1(anchor_of(address).encode("utf-8")).hexdigest(), 16) & 1

    def new_pack(r, lines, want):
        p = s280.pack_from_corpus(lines, bank=bank, tok=tok, pad_id=pad_id, device=device,
                                  rng=r, n_addr=n_addr, min_mentions=args.min_mentions,
                                  tau=args.address_tau, overlap=args.address_overlap,
                                  soft_match=0.0, min_per_family=8, addr_key=args.addr_key)
        if args.holdout == "address":
            p = dict(p)
            p["items"] = [it for it in p["items"] if side(it["address"]) == want]
        return p

    def questions(p, r):
        """Every verdict the tape can supply, balanced by construction where it can be."""
        by_anchor = defaultdict(list)
        for it in p["items"]:
            if len(it["slots"]) >= 2:
                by_anchor[anchor_of(it["address"])].append(it)
        out = []
        for anc, group in by_anchor.items():
            for it in group:
                sibs = [g for g in group
                        if relation_of(g["address"]) != relation_of(it["address"])]
                for v in VERDICTS:
                    if v == "wrong_relation":
                        # every eligible sibling is its own question. One random draw per item
                        # gave n=6 held out - below MIN_ANSWERED - so the class the stage exists
                        # for had no denominator and could not have shown a win if it were there.
                        for sb in sibs:
                            q = make_question(p, it, v, r, sibling=sb)
                            if q is not None:
                                out.append(q)
                        continue
                    q = make_question(
                        p, it, v, r,
                        other=(r.randrange(p["n_slots"]) if v == "false_premise" else None))
                    if q is not None:
                        out.append(q)
        return out

    def by_verdict(qq):
        """Class-uniform training, the way 286 samples pair-uniformly: the four verdicts arrive
        in wildly different numbers off the tape, and a mind trained on that imbalance buys
        accuracy by never naming the rare class - which is exactly the failure the first run
        showed. The examiner still sees the natural mix."""
        d = defaultdict(list)
        for q in qq:
            d[q["verdict"]].append(q)
        return [d[v] for v in VERDICTS if d[v]]

    # ---------------------------------------------------------------- graph, ranks only
    def graph(p, q):
        slots, vals_e = q["slots"], q["vals"]
        n = len(slots)
        ck, ws = p.setdefault("_ctx", {}), p.setdefault("_words", {})
        for sl in set(slots):
            if sl not in ck:
                c = bank.ctx_fp(p["texts"][sl], exclude=p["tape"].values[sl])
                ck[sl] = F.normalize(c, dim=-1) if c is not None else None
                ws[sl] = set(context_words(p["texts"][sl], exclude=p["tape"].values[sl]))
        med = p.get("_median")
        if med is None:
            lens = sorted(len(v) for v in p["postings"].values())
            med = lens[len(lens) // 2] if lens else 1
            p["_median"] = med
        # the asserted value never enters the query's text: ctx_fp excluded it anyway, so
        # appending it was decoration. The claim enters as an indicator and as same-value
        # edges, both of which the mind can actually read.
        qc = bank.ctx_fp(q["query"], exclude=q["asserted"])
        qc = F.normalize(qc, dim=-1) if qc is not None else None
        qw = set(context_words(q["query"], exclude=q["asserted"]))
        # the query is row n: same shape as a mention, so the machinery does not change
        allc = [ck[s] for s in slots] + [qc]
        allw = [ws[s] for s in slots] + [qw]
        # a query with no claim matches nobody on the value channel, which is correct: it is
        # asking, not asserting, and the same-value edge should stay silent
        allv = list(vals_e) + [q["asserted"] if q["asserted"] is not None else object()]
        m = n + 1
        same = torch.zeros(m, m)
        cos = torch.zeros(m, m)
        shared = torch.zeros(m, m)
        for i in range(m):
            for j in range(i + 1, m):
                same[i, j] = same[j, i] = float(allv[i] == allv[j])
                if allc[i] is not None and allc[j] is not None:
                    cos[i, j] = cos[j, i] = float(allc[i] @ allc[j])
                inter = allw[i] & allw[j]
                rare = sum(1 for w in inter if len(p["postings"].get(w, ())) < med)
                shared[i, j] = shared[j, i] = rare / max(1, min(len(allw[i]), len(allw[j])))
        iu = torch.triu_indices(m, m, offset=1)

        def rank_norm(M):
            if iu.numel() == 0:
                return M
            v = M[iu[0], iu[1]]
            order = v.argsort()
            r = torch.empty_like(order, dtype=torch.float32)
            r[order] = torch.arange(len(v), dtype=torch.float32)
            uniq, inv = v.unique(return_inverse=True)
            if len(uniq) > 1:
                mean_r = torch.zeros(len(uniq)).index_reduce_(0, inv, r, "mean",
                                                              include_self=False)
                r = mean_r[inv] / (len(v) - 1 if len(v) > 1 else 1)
            else:
                r = torch.zeros_like(r)
            R = torch.zeros_like(M)
            R[iu[0], iu[1]] = r
            R[iu[1], iu[0]] = r
            return R

        E = torch.stack([same, rank_norm(cos), rank_norm(shared)], -1).to(device)
        cnt = Counter(allv[:n])
        subj = q["S"]
        # WITHOUT this indicator the first run was unmeasurable: a question that claims nothing
        # and a question whose claim is absent from the evidence both produce an all-zero
        # same-value row, so ok, wrong_relation and false_premise were the SAME input and three
        # of four classes could not be told apart even in principle. The rival read the claim
        # directly in python; the mind could not see it at all. One indicator - does the query
        # assert anything - restores the comparison. It carries no identity, and it leaves
        # exactly one pair, ok / wrong_relation, blind to every count. That pair is the claim.
        claims = float(q["asserted"] is not None)
        nf = torch.tensor(
            [[cnt.get(allv[i], 0) / n,
              # does this mention name the asked subject at all - an indicator about the
              # RELATION between query and mention, not about who anybody is
              float(subj in p["texts_lc"][slots[i]]) if i < n else 0.0,
              float(i == n),                     # the query row flag
              claims if i == n else 0.0]
             for i in range(m)], dtype=torch.float32, device=device)
        return E, same.unsqueeze(-1).to(device), nf, n

    net = Judge(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    n_params = int(sum(x.numel() for x in net.parameters()))

    t_first = time.time()
    pack = new_pack(rng, train_lines, 0)
    qs = questions(pack, rng)
    buckets = by_verdict(qs)
    log(f"  first pack: {time.time() - t_first:.1f}s (cold - the memo pays back from the second)")
    log(f"  tape: {pack['n_addresses']} addresses, {pack['n_slots']} slots | "
        f"questions {json.dumps(dict(Counter(q['verdict'] for q in qs)))} | params {n_params}")
    if len(qs) < 4 * s286.MIN_ANSWERED or \
            Counter(q["verdict"] for q in qs)["wrong_relation"] < s286.MIN_ANSWERED:
        log("  too few questions, or no wrong_relation pairs: raise --addresses. "
            "wrong_relation needs one anchor to carry two different relations.")
        return 1

    # built once and reused: the saturation probe has to hold the tape fixed, otherwise a
    # change in the curve could be the mind improving or the eval tape wandering
    held = new_pack(random.Random(SEED + 99), eval_lines, 1)
    held_qs = questions(held, random.Random(SEED + 7))
    probe_rng = random.Random(SEED + 11)
    probe_qs = [q for q in held_qs if q["verdict"] in ("ok", "wrong_relation")]
    probe_rng.shuffle(probe_qs)
    probe_qs = probe_qs[: args.eval_probe]
    probe_floor = (max(Counter(q["verdict"] for q in probe_qs).values(), default=0)
                   / max(1, len(probe_qs)))
    log(f"  saturation probe: {len(probe_qs)} blind-pair questions, floor {probe_floor:.4f}")

    @torch.no_grad()
    def probe():
        net.eval()
        hit = sum(int((VERDICTS[int(net(*graph(held, q)).argmax())] == "wrong_relation")
                      == (q["verdict"] == "wrong_relation")) for q in probe_qs)
        net.train()
        return hit / max(1, len(probe_qs))

    # ---------------------------------------------------------------- train
    losses, curve, tape_curve, n_tapes = [], [], [], 1
    for step in range(1, n_steps + 1):
        if (step - 1) % args.tape_period == 0 and step > 1:
            # timed and printed every time: rule 2 of HANDOFF 9b. Two runs were lost to not
            # knowing whether a resample costs one minute or fifteen, and the answer has to be
            # readable in the first minutes rather than inferred from silence hours later.
            t_res = time.time()
            pack = new_pack(rng, train_lines, 0)
            qs = questions(pack, rng)
            buckets = by_verdict(qs)
            log(f"  resample {n_tapes + 1} at step {step}: {time.time() - t_res:.1f}s")
            if not qs:
                log("  empty tape after resample")
                return 1
            n_tapes += 1
            if args.eval_period and n_tapes % args.eval_period == 0:
                a = probe()
                tape_curve.append({"tapes": n_tapes, "step": step, "blind_pair_accuracy": a})
                log(f"  [tape {n_tapes}] step {step} blind_pair={a:.4f} "
                    f"floor={probe_floor:.4f}")
        b = buckets[rng.randrange(len(buckets))]
        q = b[rng.randrange(len(b))]
        E, same, nf, qi = graph(pack, q)
        logits = net(E, same, nf, qi)
        loss = F.cross_entropy(logits.unsqueeze(0),
                               torch.tensor([VERDICTS.index(q["verdict"])], device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if step % max(1, n_steps // 8) == 0:
            curve.append({"step": step, "loss": float(np.mean(losses[-200:]))})
            log(f"  step {step}/{n_steps} loss={np.mean(losses[-200:]):.4f}")

    net.eval()
    arc1 = s271.arc_enc_hash(can)

    # The prior the mind was NOT trained at. Training draws the four verdicts uniformly, so
    # its logits are calibrated to 1/4 each; adding the log of the tape's natural frequencies
    # is the exact Bayes correction back to the mix an examiner actually sees. Taken from the
    # TRAIN tape only - using the held-out mix would be leaking the answer sheet into the
    # decision rule. A constant offset would not change an argmax, so no normalisation is needed.
    _freq = Counter(q["verdict"] for q in qs)
    log_prior = torch.tensor(
        [math.log(max(_freq[v], 1) / max(1, sum(_freq.values()))) for v in VERDICTS],
        dtype=torch.float32, device=device)
    log(f"  train prior {json.dumps({v: _freq[v] for v in VERDICTS})}")

    # ---------------------------------------------------------------- score
    @torch.no_grad()
    def examine(p, r, qq=None):
        qq = questions(p, r) if qq is None else qq
        per = {v: {"n": 0, "model": 0, "rival": 0} for v in VERDICTS}
        conf = Counter()
        blind = {"n": 0, "model": 0, "rival": 0, "by_true": Counter(),
                 "pos": [], "neg": [], "prior": [], "prior_hit": 0}
        for q in qq:
            E, same, nf, qi = graph(p, q)
            lg = net(E, same, nf, qi)
            pred = VERDICTS[int(lg.argmax())]
            riv = counting_rival(p, q)
            per[q["verdict"]]["n"] += 1
            per[q["verdict"]]["model"] += int(pred == q["verdict"])
            per[q["verdict"]]["rival"] += int(riv == q["verdict"])
            conf[(q["verdict"], pred)] += 1
            if q["verdict"] in ("ok", "wrong_relation"):
                # the only pair no count can separate, read as the binary it is: any verdict
                # other than wrong_relation is charitably scored as "answerable", so the mind
                # is not punished here for confusing ok with contested
                blind["n"] += 1
                blind["by_true"][q["verdict"]] += 1
                tgt = q["verdict"] == "wrong_relation"
                blind["model"] += int((pred == "wrong_relation") == tgt)
                # argmax alone cannot be the verdict here and the reason is structural, not a
                # convenience: training samples the four classes UNIFORMLY (failure mode 14's
                # fix), while the examiner sees the tape's natural mix - here about 76/24. A
                # decision rule trained at one prior and scored at another loses to the majority
                # class by construction, so the argmax number charges the mind for a shift the
                # examiner introduced. Two instruments that do not have that defect: the AUC,
                # which no prior can move, and the argmax taken after the exact Bayes correction
                # for the prior the TRAIN tape actually shows - train, never the held-out set,
                # so nothing about the eval leaks into the decision.
                sc = float(torch.softmax(lg, -1)[VERDICTS.index("wrong_relation")])
                (blind["pos"] if tgt else blind["neg"]).append(sc)
                adj = lg + log_prior
                blind["prior_hit"] += int(
                    (VERDICTS[int(adj.argmax())] == "wrong_relation") == tgt)
                # measured, not assumed: the counter should never say wrong_relation here, but
                # an evidence tie can still make it answer contested, so let it be scored
                blind["rival"] += int((riv == "wrong_relation") == tgt)
        bp_auc = s286.auc(blind["pos"], blind["neg"])
        bp_z = s286.auc_z(bp_auc, len(blind["pos"]), len(blind["neg"]))
        n = sum(per[v]["n"] for v in VERDICTS)
        out = {"n": n,
               "model_accuracy": sum(per[v]["model"] for v in VERDICTS) / max(1, n),
               "rival_accuracy": sum(per[v]["rival"] for v in VERDICTS) / max(1, n),
               "majority_floor": (max(per[v]["n"] for v in VERDICTS) / max(1, n)),
               "per_verdict": {v: {"n": per[v]["n"],
                                   "model_recall": per[v]["model"] / max(1, per[v]["n"]),
                                   "rival_recall": per[v]["rival"] / max(1, per[v]["n"])}
                               for v in VERDICTS},
               "blind_pair": {"n": blind["n"],
                              "model_accuracy": blind["model"] / max(1, blind["n"]),
                              "prior_corrected_accuracy": (blind["prior_hit"]
                                                           / max(1, blind["n"])),
                              "auc": bp_auc, "auc_z": bp_z,
                              "n_wrong_relation": len(blind["pos"]),
                              "n_ok": len(blind["neg"]),
                              "rival_accuracy": blind["rival"] / max(1, blind["n"]),
                              "majority_floor": (max(blind["by_true"].values(), default=0)
                                                 / max(1, blind["n"]))},
               "confusion": {f"{a}->{b}": c for (a, b), c in sorted(conf.items())}}
        return out

    ctrl = examine(pack, random.Random(SEED + 5))
    ex = examine(held, random.Random(SEED + 7), qq=held_qs)
    log(f"  CONTROL {json.dumps(ctrl)}")
    log(f"  HELD {json.dumps(ex)}")

    wr = ex["per_verdict"]["wrong_relation"]
    g_arc = arc0 == arc1
    g_task = all(ex["per_verdict"][v]["n"] >= s286.MIN_ANSWERED for v in VERDICTS)
    # The rival's ceiling is 0.50 by construction on a balanced set - it solves contested and
    # false_premise and cannot touch the ok / wrong_relation split - so beating it overall is
    # necessary and easy. The gate that matters is the one below it.
    g_beats_rival = bool(ex["model_accuracy"] > ex["rival_accuracy"])
    g_beats_floor = bool(ex["model_accuracy"] > ex["majority_floor"])
    # the class the rival is blind to by construction - this is the stage's real claim
    g_wrong_relation = bool(wr["n"] >= s286.MIN_ANSWERED
                            and wr["model_recall"] > wr["rival_recall"])
    # and it must not buy that by crying wolf on good questions. Relative to the rival, so
    # there is no threshold: catching mismatches is worthless if it costs more false alarms
    # than the counter already makes.
    okv = ex["per_verdict"]["ok"]
    g_keeps_ok = bool(okv["n"] >= s286.MIN_ANSWERED
                      and okv["model_recall"] >= okv["rival_recall"])
    # the two gates above can both be bought cheaply in opposite directions - recall on a rare
    # class by crying wolf, recall on ok by never crying at all. Their joint statement is one
    # number: on the pair no count can separate, beat the pair's own majority.
    # The gate reads the AUC, not the argmax, and the reason is written above the measurement:
    # the mind is trained at a uniform prior and examined at the tape's natural one, so an
    # argmax comparison against the majority floor charges it for a shift the examiner made.
    # The AUC cannot be moved by a prior at all, and 1.645 is the same one-sided null point 286
    # uses everywhere else - no new threshold enters the project.
    bp = ex["blind_pair"]
    g_blind_pair = bool(bp["n"] >= 2 * s286.MIN_ANSWERED
                        and not math.isnan(bp["auc_z"]) and bp["auc_z"] > 1.645)
    # kept and reported, because it is what a user of this mind would actually get
    g_blind_pair_argmax = bool(bp["prior_corrected_accuracy"] > bp["majority_floor"])

    overall = ("NO_TASK" if not (g_task and g_arc)
               else "PRESUPPOSITION_OK" if (g_beats_rival and g_beats_floor and g_blind_pair
                                            and g_wrong_relation and g_keeps_ok)
               else "PRESUPPOSITION_PARTIAL" if (g_wrong_relation or g_beats_rival)
               else "PRESUPPOSITION_NO")

    out = {
        "stage": "289a", "overall": overall, "seed": SEED, "smoke": args.smoke,
        "holdout": args.holdout, "run_tag": args.run_tag, "train_steps": n_steps,
        "params": n_params, "verdicts": list(VERDICTS),
        "gates": {
            "G_arc_enc_frozen": g_arc,
            "G_task_exists": g_task,
            "G_beats_counting_rival": g_beats_rival,
            "G_beats_majority_floor": g_beats_floor,
            "G_catches_wrong_relation": g_wrong_relation,
            "G_does_not_cry_wolf": g_keeps_ok,
            "G_separates_blind_pair": g_blind_pair,
            "G_blind_pair_usable_at_argmax": g_blind_pair_argmax,
        },
        "rival_ceiling_note": ("the counting rival solves contested and false_premise and is "
                               "blind to the ok / wrong_relation split by construction, so its "
                               "ceiling on a balanced set is about 0.50; overall accuracy above "
                               "it is necessary, and G_catches_wrong_relation is the claim"),
        "blind_pair_note": (
            "G_separates_blind_pair reads the AUC because the argmax cannot settle it: training "
            "draws the four verdicts uniformly, by design, and the examiner sees the tape's "
            "natural mix, so comparing an argmax against the majority floor charges the mind "
            "for a prior shift the examiner introduced. The AUC is prior-free and its null "
            "point is 286's usual 1.645. prior_corrected_accuracy is the argmax after the exact "
            "Bayes correction using the TRAIN tape's frequencies - never the held-out ones - "
            "and it is what someone using this mind would actually get"),
        "held_out": ex, "train_control": ctrl, "curve": curve,
        "n_tapes": n_tapes, "tape_curve": tape_curve,
        "tape_curve_note": ("blind-pair accuracy on ONE fixed held-out tape, measured every "
                            f"{args.eval_period} distinct training tapes against floor "
                            f"{probe_floor:.4f}. Where it flattens is how many tapes the mind "
                            "actually needs; resampling is not a simulation of use - real use "
                            "is one growing tape - it is the proof that no single tape was "
                            "memorised, which is the whole separation claim"),
        "tape_curve_floor": probe_floor, "tape_curve_n": len(probe_qs),
        "arc_enc_hash_before": arc0, "arc_enc_hash_after": arc1,
        "fp_version": s271.fp_version(),
        "note": (
            "The first verb that is not a lookup: should this question be answered at all. Four "
            "verdicts built from the tape so the labels are free - ok, wrong_relation, "
            "contested, false_premise. The first draft was a tautology and was caught before "
            "the run: building 'absent' as evidence from other addresses and 'false premise' as "
            "a borrowed value makes each class exactly the property a two-line hand rule reads "
            "off, so the rule scores 1.0 and nothing is measured, which is 286's exam mistake "
            "again. wrong_relation replaces that: ask about one relation of a subject while the "
            "evidence answers another, and every counting signal stays HEALTHY - the subject is "
            "in every mention, the values cohere, the majority is clear - so a counter answers "
            "confidently and answers a different question. Its blindness there is by "
            "construction, as it is on 288's duplicated forgery. The query joins the graph as "
            "one more row carrying the asserted value and the question's own context, so the "
            "same relational machinery compares question to evidence with no new "
            "representation. Ranks and indicators only; the output is one of four verdicts, so "
            "no value is produced and nothing is generated. THE FIRST RUN WAS UNMEASURABLE and "
            "the defect was in the graph, not in the mind: the asserted value reached the graph "
            "only through the query row's same-value edges, and that row is all-zero both when "
            "the question claims nothing and when its claim is absent from the evidence, while "
            "ctx_fp excluded the value from the text as well - so ok, wrong_relation and "
            "false_premise were literally the same input and three of four classes were "
            "indistinguishable in principle. The mind put all of them in one class (ok recall "
            "0.0, wrong_relation recall 0.0, on the held-out set AND on the train control) "
            "while the rival read the claim directly in python, so it was being compared "
            "against a strictly better-informed opponent. Fixed by one identity-free indicator "
            "- does this query assert anything - which leaves exactly one pair, ok / "
            "wrong_relation, blind to every count, and that pair is now scored on its own as "
            "blind_pair. Second defect, same run: one random sibling per item gave "
            "wrong_relation n=6 held out, under MIN_ANSWERED, so the class the stage exists for "
            "had no denominator; every eligible sibling is now its own question, and training "
            "samples the four verdicts uniformly so accuracy cannot be bought by never naming "
            "the rare one."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(), "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"stage289a_decision{tag}.json").write_text(json.dumps(out, indent=2),
                                                       encoding="utf-8")
    log(json.dumps({"overall": overall, "gates": out["gates"],
                    "model": ex["model_accuracy"], "rival": ex["rival_accuracy"],
                    "wrong_relation": wr}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
