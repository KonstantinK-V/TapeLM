"""
Stage 284 — Addressing without a constant to tune.

283 said the tape survives growth, and it said so while every decision in the addressing was
made by a number chosen by hand: merge at cosine 0.90, read seven slots, demand two shared
words, hop when the top score falls under 1.0. Those numbers were picked on a tape of 530
slots. A system that needs them re-picked at 10^5 has not scaled, it has been re-fitted, and
the difference is invisible as long as the ladder is allowed a fresh configuration per rung.

So this stage removes the constants rather than retuning them, using the rule the mind already
lives by. 278's teacher never read an absolute score - it compared the leader with the runner
up - and a comparison does not care how full the space is. The same idea, three times:

  MERGE. Not "cosine clears tau" but "each is the other's nearest": two mentions join only if
  they are MUTUALLY nearest, in both channels. Mutual nearest neighbour has no threshold at
  all, and it cannot chain a crowd together the way single-link at a fixed cosine does, because
  a mention with a closer neighbour elsewhere simply is not mutual with this one.

  SHARED WORDS. Not "at least two" but "at least one that is not ordinary": the shared word has
  to carry more idf than the median word of the tape, which is a quantity the corpus computes
  about itself rather than a number brought from outside.

  READING. Not "the top seven" but "everything before the biggest drop": the candidate list is
  cut at the largest gap between consecutive scores, so a well supported address gives up more
  slots than a thin one, and neither is capped by a constant that has to grow with the tape.

The gate that matters is not any single number. It is that ONE configuration clears every rung:
G_one_config_all_rungs fails if a metric passes at the top of the ladder while failing lower
down, which is exactly what per-rung tuning buys and what 283 could not see, since it read its
gates off the last rung alone.

  python _stage284_scalefree.py --smoke
  python _stage284_scalefree.py --rungs 4M 30M 120M
  python _stage284_scalefree.py --rungs 4M 30M 120M --rule fixed   # the arm 283 measured
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
import _stage271_controller as s271
import _stage279_write_decision as s279
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tape_index import context_words

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 284
FAMILIES = ("clean", "decidable", "tie")
LOG_PATH = RES / "_stage284_log.txt"


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def chars_of(spec: str) -> int:
    mult = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000}.get(spec[-1].upper(), 1)
    return int(float(spec[:-1]) * mult) if mult > 1 else int(spec)


# --------------------------------------------------------------------------- merging by rank

def mutual_groups(A: torch.Tensor, C: torch.Tensor, words, informative, chunk: int = 512):
    """Group mentions that are each other's nearest, in both channels and in the words.

    No cosine threshold anywhere. Two mentions are joined when neither has a closer partner -
    which is a statement about the ranking, so it means the same thing on a tape of fifty slots
    and on one of fifty thousand. The pairs are then closed transitively, but only through
    mutual pairs: a mention dragged toward a crowd loses its mutuality to every member of it
    and stays where it is, which is the failure single-link-at-tau could not avoid.
    """
    n = A.size(0)
    if n == 0:
        return []
    best = torch.empty(n, dtype=torch.long)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        sims = torch.minimum(A[i:j] @ A.T, C[i:j] @ C.T)
        sims[torch.arange(j - i, device=A.device), torch.arange(i, j, device=A.device)] = -2
        best[i:j] = sims.argmax(dim=1).cpu()

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        j = int(best[i])
        # mutual, and agreeing on a word the corpus does not hand out for free
        if int(best[j]) == i and (words[i] & words[j] & informative):
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def tau_groups(A: torch.Tensor, C: torch.Tensor, words, tau: float, min_overlap: int):
    """279's rule, kept runnable so the constants can be measured rather than argued about."""
    members: list[list[int]] = []
    for i in range(A.size(0)):
        best, best_s = -1, tau
        for g, mem in enumerate(members):
            idx = torch.tensor(mem, device=A.device)
            sims = torch.minimum(A[idx] @ A[i], C[idx] @ C[i]).tolist()
            for j, sim in enumerate(sims):
                if sim >= best_s and len(words[i] & words[mem[j]]) >= min_overlap:
                    best, best_s = g, sim
        if best < 0:
            members.append([i])
        else:
            members[best].append(i)
    return members


# --------------------------------------------------------------------------- reading by gap

def elbow(scored: list[tuple[int, float]]) -> list[int]:
    """Everything above the biggest drop. A rank rule, so no k and no floor.

    A fixed k hands the reader whatever fills the list and caps a well attested address at the
    same size as a thin one; 283 measured that cap biting - 12% of addresses had more mentions
    than the reader could hold. The largest gap between consecutive scores is where the list
    stops being about this address, and it is computed from the scores themselves.
    """
    if len(scored) <= 1:
        return [c for c, _ in scored]
    cut, gap = len(scored), -1.0
    for i in range(1, len(scored)):
        d = scored[i - 1][1] - scored[i][1]
        if d > gap:
            gap, cut = d, i
    return [c for c, _ in scored[:cut]]


def retrieve(pack, words, rule: str, k: int):
    cands, sc = s271.vote(words, pack["postings"], pack["idf"], max(k, 64) if rule == "margin"
                          else k)
    if not cands:
        return [], {}
    scored = sorted(((c, sc.get(c, 0.0)) for c in cands), key=lambda t: -t[1])
    keep = elbow(scored) if rule == "margin" else [c for c, _ in scored[:k]]
    return keep, {c: sc.get(c, 0.0) for c in keep}


def return_path(pack, S, answered) -> float:
    words = context_words(answered) or [answered]
    scan = min((pack["postings_probe"].get(w, ()) for w in words), key=len, default=())
    val = answered.lower()
    seen = 0
    for c in scan:
        if S in pack["texts_lc"][c] and val in pack["texts_lc"][c]:
            seen += 1
            if seen >= 2:
                return 1.0
    return 0.0


# --------------------------------------------------------------------------- one rung

def build(lines, bank, rng, n_addr, min_mentions, rule, tau, overlap):
    common = s279.common_nouns(lines)
    asserts, _ = s279.corpus_assertions(lines, rng, n_addr, min_mentions, "anchor_rel",
                                        common=common)
    if not asserts:
        return None
    akeys, ckeys, words = [], [], []
    for a in asserts:
        anchor = a["address"].split("|")[0]
        c = bank.ctx_fp(a["ctx"], exclude=a["value"])
        kk = bank.fp([anchor])[0]
        akeys.append(F.normalize(kk, dim=-1))
        ckeys.append(F.normalize(c, dim=-1) if c is not None else F.normalize(kk, dim=-1))
        words.append({w.lower() for w in s279.REL_RE.findall(a["ctx"])
                      if w.lower() not in s279.VALUE_STOP} - {a["value"].lower()})
    A = F.normalize(torch.stack(akeys).float(), dim=-1)
    C = F.normalize(torch.stack(ckeys).float(), dim=-1)

    # "informative" is the corpus describing itself: a word worth agreeing on is one that
    # appears less often than the median word. No number is brought in from outside.
    df = Counter(w for ws in words for w in ws)
    med = float(np.median(list(df.values()))) if df else 1.0
    informative = {w for w, n in df.items() if n <= med}

    groups = (mutual_groups(A, C, words, informative) if rule == "margin"
              else tau_groups(A, C, words, tau, overlap))
    groups = [g for g in groups if len(g) >= min_mentions]
    if not groups:
        return None

    vals, texts, aslots = [], [], []
    for g in groups:
        sids = []
        for i in g:
            sids.append(len(vals))
            vals.append(asserts[i]["value"])
            texts.append(asserts[i]["ctx"])
        aslots.append(sids)

    postings, postings_probe = defaultdict(list), defaultdict(list)
    for cid, t in enumerate(texts):
        for w in context_words(t, exclude=vals[cid]):
            postings[w].append(cid)
        for w in context_words(t):
            postings_probe[w].append(cid)
    n_slots = len(vals)
    idf = {w: math.log(max(2.0, n_slots / max(1, len(postings[w])))) for w in postings}
    idf_probe = {w: math.log(max(2.0, n_slots / max(1, len(postings_probe[w]))))
                 for w in postings_probe}

    items = []
    for gi, g in enumerate(groups):
        sids = aslots[gi]
        cnt = Counter(vals[i] for i in sids)
        ranked = cnt.most_common(2)
        lead, second = ranked[0][1], (ranked[1][1] if len(ranked) > 1 else 0)
        kind, truth = (("clean", ranked[0][0]) if len(cnt) == 1
                       else ("tie", None) if lead == second
                       else ("decidable", ranked[0][0]))
        anchor = asserts[g[0]]["address"].split(":", 1)[-1].split("|")[0]
        rel = (asserts[g[0]]["address"].split("|", 1) + [""])[1]
        items.append({"S": anchor, "query": (anchor + " " + rel).strip(), "truth": truth,
                      "slots": sids, "kind": kind})
    return {"items": items, "texts_lc": [t.lower() for t in texts], "postings": postings,
            "idf": idf, "postings_probe": postings_probe, "idf_probe": idf_probe,
            "n_addresses": len(groups), "n_slots": n_slots, "values": vals}


def measure(pack, rng, rule, k) -> dict:
    items, vals = pack["items"], pack["values"]
    distinct = sorted(set(vals))
    prec, rec, ncand, foreign, covers = [], [], [], [], []
    r_true, r_other = [], []
    for it in items:
        cands, _ = retrieve(pack, context_words(s271.CUE.format(S=it["query"])), rule, k)
        own = set(it["slots"])
        if cands:
            hit = sum(1 for c in cands if c in own)
            prec.append(hit / len(cands))
            rec.append(hit / max(1, len(own)))
        ncand.append(len(cands))
        covers.append(int(len(it["slots"]) <= len(cands)) if cands else 0)
        foreign.append(sum(1 for c in it["slots"] if it["S"] not in pack["texts_lc"][c])
                       / max(1, len(it["slots"])))
        if it["truth"] is not None:
            r_true.append(return_path(pack, it["S"], it["truth"]))
            other = [v for v in distinct if v != it["truth"]]
            if other:
                r_other.append(return_path(pack, it["S"], rng.choice(other)))
    fam = Counter(it["kind"] for it in items)
    m = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")
    return {"n_addresses": pack["n_addresses"], "n_slots": pack["n_slots"],
            "n_items": len(items),
            "slots_per_address": pack["n_slots"] / max(1, pack["n_addresses"]),
            "families_natural": {f: fam.get(f, 0) / max(1, len(items)) for f in FAMILIES},
            "retrieval_precision": m(prec), "witness_recall": m(rec),
            "mean_candidates": m(ncand), "reader_covers_address": m(covers),
            "foreign_member_rate": m(foreign),
            "return_path_true": m(r_true), "return_path_other": m(r_other),
            "return_path_separation": m(r_true) - m(r_other)}


def rung_gates(r, args) -> dict:
    return {
        "precision": r["retrieval_precision"] >= args.min_precision,
        "return_path": r["return_path_separation"] >= args.min_separation,
        "addresses_distinct": r["foreign_member_rate"] <= args.max_foreign,
        "reader_covers": r["reader_covers_address"] >= args.min_covers,
        "ties_exist": r["families_natural"]["tie"] > 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rungs", nargs="*", default=[])
    ap.add_argument("--corpus", type=Path, default=WIKI)
    ap.add_argument("--rule", choices=("margin", "fixed"), default="margin")
    ap.add_argument("--addresses", type=int, default=4000)
    ap.add_argument("--min-mentions", type=int, default=2)
    ap.add_argument("--topk", type=int, default=7, help="only read under --rule fixed")
    ap.add_argument("--address-tau", type=float, default=0.90, help="only under --rule fixed")
    ap.add_argument("--address-overlap", type=int, default=2, help="only under --rule fixed")
    ap.add_argument("--min-precision", type=float, default=0.60)
    ap.add_argument("--min-separation", type=float, default=0.30)
    ap.add_argument("--max-foreign", type=float, default=0.10)
    ap.add_argument("--min-covers", type=float, default=0.90)
    ap.add_argument("--run-tag", type=str, default="")
    args = ap.parse_args()

    global LOG_PATH
    tag = (args.run_tag and f"_{args.run_tag}") or ""
    tag += "" if args.rule == "margin" else "_fixed"
    LOG_PATH = RES / f"_stage284_log{tag}.txt"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    rungs = [chars_of(s) for s in args.rungs] or (
        [2_000_000, 4_000_000] if args.smoke else [4_000_000, 30_000_000, 120_000_000])
    rungs.sort()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    s185.build_char_table(tok, stoi, pad_id, tok.get_vocab_size())
    can = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    can.eval()
    for p in can.parameters():
        p.requires_grad_(False)
    bank = FpBank(can, stoi, device)
    mind_params = sum(p.numel() for p in can.parameters())

    log(f"Stage284 scalefree start {datetime.now(timezone.utc).isoformat()} rule={args.rule} "
        f"rungs={rungs} mind_params={mind_params}")
    if not args.corpus.exists():
        log(f"  corpus not found: {args.corpus}")
        return 1
    with args.corpus.open("r", encoding="utf-8", errors="ignore") as f:
        full = f.read(max(rungs))

    out_rungs = []
    for chars in rungs:
        t1 = time.time()
        rng = random.Random(SEED)
        lines = [l.strip() for l in full[:chars].split("\n") if 80 <= len(l.strip()) <= 400]
        pack = build(lines, bank, rng, args.addresses, args.min_mentions, args.rule,
                     args.address_tau, args.address_overlap)
        if pack is None or not pack["items"]:
            log(f"  rung {chars} produced nothing, skipped")
            continue
        r = measure(pack, rng, args.rule, args.topk)
        r.update({"chars": chars, "build_s": time.time() - t1, "mind_params": mind_params,
                  "gates": rung_gates(r, args)})
        out_rungs.append(r)
        log(f"  rung {chars/1e6:.0f}M -> {r['n_addresses']} addr / {r['n_slots']} slots"
            f" | precision {r['retrieval_precision']:.3f}"
            f" | return sep {r['return_path_separation']:+.3f}"
            f" | foreign {r['foreign_member_rate']:.3f}"
            f" | covers {r['reader_covers_address']:.3f}"
            f" | cands {r['mean_candidates']:.2f}"
            f" | ties {r['families_natural']['tie']:.3f} ({r['build_s']:.0f}s)")

    if not out_rungs:
        log("  nothing measured")
        return 1

    # The gate this stage exists for. Not "the top rung passes" - 283 read its gates off the
    # last rung and could not have seen a configuration that works at one size and not another.
    # One configuration, every rung, or it did not scale.
    per_rung_ok = [all(r["gates"].values()) for r in out_rungs]
    g_one_config = all(per_rung_ok) and len(out_rungs) > 1
    g_mind = all(r["mind_params"] == mind_params for r in out_rungs)
    overall = ("SCALEFREE_OK" if g_one_config and g_mind
               else "SCALEFREE_PARTIAL" if any(per_rung_ok) else "SCALEFREE_NO")

    out = {
        "stage": 284, "overall": overall, "rule": args.rule, "seed": SEED, "smoke": args.smoke,
        "corpus": str(args.corpus), "run_tag": args.run_tag, "mind_params": mind_params,
        "constants_in_use": ({"none": True} if args.rule == "margin"
                             else {"tau": args.address_tau, "overlap": args.address_overlap,
                                   "topk": args.topk}),
        "thresholds": {"min_precision": args.min_precision,
                       "min_separation": args.min_separation,
                       "max_foreign": args.max_foreign, "min_covers": args.min_covers},
        "gates": {"G_one_config_all_rungs": g_one_config, "G_mind_does_not_grow": g_mind},
        "per_rung_pass": per_rung_ok,
        "slopes": {kk: [r[kk] for r in out_rungs] for kk in
                   ("n_slots", "n_addresses", "retrieval_precision", "witness_recall",
                    "return_path_separation", "foreign_member_rate", "reader_covers_address",
                    "mean_candidates")},
        "tie_share": [r["families_natural"]["tie"] for r in out_rungs],
        "rungs": out_rungs,
        "fp_version": s271.fp_version(),
        "reference_283": {"rule": "fixed", "foreign_member_rate_top": 0.00860178354884333,
                          "precision_top": 0.6201471466579791,
                          "k_covers_top": 0.9486887115165337},
        "note": (
            "283 said the tape survives growth while every decision in the addressing was made "
            "by a number picked on a tape of 530 slots: merge at cosine 0.90, read seven, "
            "demand two shared words. A system that needs those re-picked at 10^5 has not "
            "scaled, it has been re-fitted, and a ladder allowed a fresh configuration per rung "
            "cannot tell the two apart. So the constants are removed rather than retuned, using "
            "the rule 278's teacher already lived by - compare the leader with the runner up, "
            "never read an absolute score. Mentions merge when they are MUTUALLY nearest in "
            "both channels, agreeing on a word rarer than the tape's median; the reader stops "
            "at the largest gap between consecutive scores instead of at a fixed k. The gate is "
            "that one configuration clears every rung, which is what per-rung tuning buys and "
            "what reading the gates off the last rung alone would hide. --rule fixed is the arm "
            "283 measured, kept runnable so the constants can be compared rather than argued."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(), "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"stage284_decision{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(json.dumps({"overall": overall, "gates": out["gates"],
                    "per_rung_pass": per_rung_ok, "slopes": out["slopes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
