"""
Stage 283 — Does the tape survive its own growth?

280 reached RAW_EXAM_OK on 187 addresses over 530 slots, and every number in it is a statement
about an index of 530 entries. The concept's own invariant is that the mind does not grow with
the knowledge, so the only way that invariant can fail is on the tape side: addressing, not
parameters. This measures the tape side alone - no policy, no BC, no RL, nothing trained - so a
ladder that would take days with a mind attached takes minutes.

Four things are expected to degrade, and each has a number here rather than an opinion.

  RETRIEVAL PRECISION. Votes over an inverted index return k slots whatever the index holds.
  With 530 slots a wrong slot is a coincidence; with 10^5 it is a population. If precision
  falls with N, every downstream number in 280 is a number about small tapes.

  THE RETURN PATH. 282's probe asks whether some other mention carries the subject and the
  value together. Accidental co-occurrence of two strings grows with the corpus - the 4MB smoke
  already showed it, where anchors like "september" made every value look corroborated. So the
  check is run twice at each rung: on the value the corpus actually settled on, and on a value
  taken from elsewhere in the tape. Only the DISTANCE between those two rates says whether the
  check still discriminates.

  ADDRESS COLLISIONS. An address is norm(fp(anchor) + ctx_fp(context)) and two addresses are
  merged when they pass tau with enough shared words. Distinct entities crowd as N grows, so the
  nearest-other-address cosine is reported as a distribution, not a mean.

  THE NATURAL FAMILY MIX. 280 quotas the families with --min-per-family so that abstention can
  be measured at all. That quota is a property of the exam, not of the corpus. Here the mix is
  left alone, which is the only way to see what a tie rate really is.

Nothing is trained and nothing is claimed about accuracy: an exam needs a mind, and this is a
measurement of what the mind would be handed.

  python _stage283_scale.py --smoke
  python _stage283_scale.py --rungs 4M:400 30M:400 120M:2000 400M:10000
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
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage271_controller as s271
import _stage280_raw_exam as s280
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tape_index import context_words

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 283
LOG_PATH = RES / "_stage283_log.txt"


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def parse_rungs(specs: list[str]) -> list[tuple[int, int]]:
    """'30M:400' -> (30_000_000 chars, 400 addresses). Both have to move: the corpus caps how
    many addresses exist and n_addr caps how many are taken, so raising one alone measures the
    other's ceiling rather than scale."""
    out = []
    for s in specs:
        chars, _, addrs = s.partition(":")
        mult = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000}.get(chars[-1].upper(), 1)
        n = int(float(chars[:-1]) * mult) if mult > 1 else int(chars)
        out.append((n, int(addrs)))
    return sorted(out)


# --------------------------------------------------------------------------- the measurements

def nearest_other(keys: torch.Tensor, chunk: int = 512) -> np.ndarray:
    """Cosine to the closest OTHER address. Chunked, because the pairwise matrix is the one
    thing here that grows quadratically and the whole point is to run at N where that bites."""
    n = keys.size(0)
    best = torch.empty(n, dtype=torch.float32, device=keys.device)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        sims = keys[i:j] @ keys.T
        sims[torch.arange(j - i, device=keys.device), torch.arange(i, j, device=keys.device)] = -2
        best[i:j] = sims.max(dim=1).values
    return best.cpu().numpy()


def nearest_other_two(anc, ctx, group: torch.Tensor, chunk: int = 256):
    """The same closest-other-member number, scored the way --addr-key two scores it.

    Written because the first version of this measurement read the summed key whatever rule was
    in force, so an arm that changes the rule would have moved nothing and the run would have
    proved only that the metric was blind.
    """
    n = anc.size(0)
    n_g = int(group.max()) + 1 if n else 0
    best = torch.full((n_g,), -2.0, device=anc.device)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        sims = torch.minimum(anc[i:j] @ anc.T, ctx[i:j] @ ctx.T)
        sims[group[i:j].unsqueeze(1) == group.unsqueeze(0)] = -2
        best.scatter_reduce_(0, group[i:j], sims.max(dim=1).values, reduce="amax")
    return best.cpu().numpy()


def nearest_other_member(keys: torch.Tensor, group: torch.Tensor, chunk: int = 256):
    """Closest member of a DIFFERENT address, per address. The set rule's own crowding.

    Averaging members hides how they sit: two addresses whose means are far apart can still
    have one mention each that all but coincide, and under MaxSim that pair is what decides a
    hop. So the mean-key number stays as the old arm's and this one measures the rule in use.
    """
    n = keys.size(0)
    n_g = int(group.max()) + 1 if n else 0
    best = torch.full((n_g,), -2.0, device=keys.device)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        sims = keys[i:j] @ keys.T
        sims[group[i:j].unsqueeze(1) == group.unsqueeze(0)] = -2
        best.scatter_reduce_(0, group[i:j], sims.max(dim=1).values, reduce="amax")
    return best.cpu().numpy()


def q(xs, *ps):
    a = np.asarray([x for x in xs if not math.isnan(x)], dtype=np.float64)
    if a.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{int(p)}": float(np.percentile(a, p)) for p in ps}


def measure(pack, bank, rng, k, hop, hop_min, k_gap, subject_filter, tau) -> dict:
    items = pack["items"]
    values = pack["tape"].values
    distinct_values = sorted(set(values))

    prec, rec, ncand, silent = [], [], [], []
    ret_true, ret_other = [], []
    foreign, k_covers = [], []
    for it in items:
        # Crowding is proximity between keys; the harm is a merge that actually happened. A slot
        # filed under an address whose anchor its own text never mentions is that harm, exactly,
        # and it needs no labels to see. Keys may crowd harmlessly - fp_addresses also demands
        # shared words - so the two numbers have to be read together.
        foreign.append(sum(1 for c in it["slots"]
                           if it["S"] not in pack["texts_lc"][c]) / max(1, len(it["slots"])))
        k_covers.append(int(len(it["slots"]) <= k))
        words = context_words(s271.CUE.format(S=it.get("query") or it["S"]))
        cands, sc, _ = s280.retrieve(pack, words, k, hop, it, subject_filter, hop_min, k_gap)
        own = set(it["slots"])
        if cands:
            hit = sum(1 for c in cands if c in own)
            prec.append(hit / len(cands))
            rec.append(hit / max(1, len(own)))
        ncand.append(len(cands))
        silent.append(int(not sc or max(sc.values(), default=0.0) <= 0.0))

        # The return path, measured against itself. truth is read here and ONLY here, to score
        # a check - the same way accuracy reads it. Ties have no settled value, so they have no
        # true side to score and are skipped rather than counted as a failure.
        if it["truth"] is not None:
            ret_true.append(s280.return_path(pack, it, it["truth"]))
            other = [v for v in distinct_values if v != it["truth"]]
            if other:
                ret_other.append(s280.return_path(pack, it, rng.choice(other)))

    near = nearest_other(pack["addr_keys"]) if pack["addr_keys"] is not None \
        else np.array([float("nan")])
    # Two ways for a tape to run out of room, and they call for opposite repairs. Either the
    # corpus stops producing new anchors - vocabulary is sublinear in text, so this is the
    # benign one - or the anchors stay distinct as strings and crowd anyway, because fp is a
    # character histogram with about 26 effective dimensions (277) and 26 dimensions hold only
    # so many points however many different names there are. The address key carries context as
    # well as anchor, so measuring the anchor ALONE says which limit is being approached: if
    # bare anchors crowd while full addresses do not, the context is what is saving this and
    # the ink cannot address a large tape by itself.
    if pack.get("slot_keys") is not None:
        g = torch.tensor([pack["slot_addr"][i] for i in pack["slot_keys_slot"]],
                         device=pack["slot_keys"].device)
        near_m = nearest_other_member(pack["slot_keys"], g)
        near_t = (nearest_other_two(pack["anc_keys"], pack["ctx_keys"], g)
                  if pack.get("anc_keys") is not None else np.array([float("nan")]))
    else:
        near_m = near_t = np.array([float("nan")])
    anchors = sorted({it["S"] for it in items})
    a_keys = torch.nn.functional.normalize(bank.fp(anchors).float(), dim=-1) \
        if anchors else None
    a_near = nearest_other(a_keys.to(pack["addr_keys"].device)) \
        if a_keys is not None and a_keys.size(0) > 1 else np.array([float("nan")])
    fam = Counter(it["kind"] for it in items)
    m = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")
    return {
        "n_addresses": pack["n_addresses"], "n_slots": pack["n_slots"], "n_items": len(items),
        "slots_per_address": pack["n_slots"] / max(1, pack["n_addresses"]),
        "write_actions": pack["write_actions"],
        "families_natural": {f: fam.get(f, 0) / max(1, len(items))
                             for f in ("clean", "decidable", "tie")},
        "retrieval_precision": m(prec), "witness_recall": m(rec),
        "precision_q": q(prec, 5, 50, 95),
        "mean_candidates": m(ncand), "words_silent_rate": m(silent),
        "foreign_member_rate": m(foreign), "k_covers_address": m(k_covers),
        "return_path_true": m(ret_true), "return_path_other": m(ret_other),
        "return_path_separation": m(ret_true) - m(ret_other),
        "nearest_other_q": q(near, 50, 95, 99),
        "nearest_other_max": float(np.nanmax(near)),
        "address_crowding": float(np.mean(near >= tau)),
        "member_crowding_q": q(near_m, 50, 95, 99),
        "member_crowding": float(np.mean(near_m >= tau)),
        "two_channel_crowding": float(np.mean(near_t >= tau)),
        "two_channel_q": q(near_t, 50, 95, 99),
        "distinct_anchors": len(anchors),
        "anchor_only_q": q(a_near, 50, 95, 99),
        "anchor_only_crowding": float(np.mean(a_near >= tau)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rungs", nargs="*", default=[],
                    help="chars:addresses, e.g. 30M:400 120M:2000")
    ap.add_argument("--corpus", type=Path, default=WIKI)
    ap.add_argument("--min-mentions", type=int, default=2)
    ap.add_argument("--address-tau", type=float, default=0.90)
    ap.add_argument("--address-overlap", type=int, default=2)
    ap.add_argument("--soft-match", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=7)
    ap.add_argument("--hop", choices=("none", "fp"), default="fp")
    ap.add_argument("--hop-min", type=float, default=1.0)
    ap.add_argument("--k-gap", type=float, default=0.35)
    ap.add_argument("--subject-filter", choices=("off", "on"), default="on")
    ap.add_argument("--addr-key", choices=("two", "set", "mean"), default="two")
    ap.add_argument("--min-precision", type=float, default=0.60)
    ap.add_argument("--min-separation", type=float, default=0.30)
    ap.add_argument("--max-crowding", type=float, default=0.20)
    ap.add_argument("--max-foreign", type=float, default=0.10)
    ap.add_argument("--run-tag", type=str, default="")
    args = ap.parse_args()

    global LOG_PATH
    tag = (args.run_tag and f"_{args.run_tag}") or ""
    LOG_PATH = RES / f"_stage283_log{tag}.txt"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    rungs = parse_rungs(args.rungs) if args.rungs else (
        [(2_000_000, 100), (4_000_000, 200)] if args.smoke
        else [(4_000_000, 400), (30_000_000, 400), (120_000_000, 2000)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    s185.build_char_table(tok, stoi, pad_id, V)
    can = SelfModelXL(n_char, V).to(device)
    can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    can.eval()
    for p in can.parameters():
        p.requires_grad_(False)
    bank = FpBank(can, stoi, device)
    # The invariant, stated as a number rather than as a promise: this is the whole mind that
    # addresses the tape, and it is the same object at every rung below.
    mind_params = sum(p.numel() for p in can.parameters())

    log(f"Stage283 scale start {datetime.now(timezone.utc).isoformat()} device={device} "
        f"rungs={rungs} mind_params={mind_params}")

    if not args.corpus.exists():
        log(f"  corpus not found: {args.corpus}")
        return 1
    with args.corpus.open("r", encoding="utf-8", errors="ignore") as f:
        full = f.read(max(c for c, _ in rungs))
    log(f"  corpus {len(full)/1e6:.1f}M chars read ({time.time()-t0:.0f}s)")

    out_rungs = []
    for chars, n_addr in rungs:
        t1 = time.time()
        rng = random.Random(SEED)
        lines = [l.strip() for l in full[:chars].split("\n") if 80 <= len(l.strip()) <= 400]
        pack = s280.pack_from_corpus(
            lines, bank=bank, tok=tok, pad_id=pad_id, device=device, rng=rng, n_addr=n_addr,
            min_mentions=args.min_mentions, tau=args.address_tau,
            overlap=args.address_overlap, soft_match=args.soft_match,
            min_per_family=0,                     # natural mix: the quota is the exam's, not the corpus's
            addr_key=args.addr_key)
        build_s = time.time() - t1
        if not pack["items"]:
            log(f"  rung {chars}:{n_addr} produced no items, skipped")
            continue
        r = measure(pack, bank, rng, args.topk, args.hop, args.hop_min, args.k_gap,
                    args.subject_filter == "on", args.address_tau)
        r.update({"chars": chars, "n_addr_requested": n_addr, "build_s": build_s,
                  "mind_params": mind_params, "lines": len(lines)})
        out_rungs.append(r)
        log(f"  rung {chars/1e6:.0f}M:{n_addr} -> {r['n_addresses']} addr / {r['n_slots']} slots"
            f" | precision {r['retrieval_precision']:.3f}"
            f" | return true {r['return_path_true']:.3f} other {r['return_path_other']:.3f}"
            f" (sep {r['return_path_separation']:+.3f})"
            f" | crowding {r['address_crowding']:.3f}"
            f" (members {r['member_crowding']:.3f} two {r['two_channel_crowding']:.3f}"
            f" anchors alone {r['anchor_only_crowding']:.3f})"
            f" foreign {r['foreign_member_rate']:.3f}"
            f" | k covers {r['k_covers_address']:.3f}"
            f" | ties {r['families_natural']['tie']:.3f} ({build_s:.0f}s)")

    if not out_rungs:
        log("  nothing measured")
        return 1
    last = out_rungs[-1]
    first = out_rungs[0]
    g_precision = last["retrieval_precision"] >= args.min_precision
    g_separation = last["return_path_separation"] >= args.min_separation
    # Two readings of the same worry. Keys crowding is the warning; a slot filed under an
    # address its own text never names is the damage, and only the second one is a gate.
    g_crowding = last["foreign_member_rate"] <= args.max_foreign
    g_ties = last["families_natural"]["tie"] > 0.0
    # Not "did it stay the same" but "did it stay usable": a metric may sag with N and still
    # clear its floor, and a metric that sags steeply while clearing it is the interesting case.
    g_mind_constant = all(r["mind_params"] == mind_params for r in out_rungs)

    overall = ("SCALE_OK" if all((g_precision, g_separation, g_crowding, g_ties))
               else "SCALE_PARTIAL" if (g_precision and g_ties) else "SCALE_NO")

    out = {
        "stage": 283, "overall": overall, "seed": SEED, "smoke": args.smoke,
        "corpus": str(args.corpus), "run_tag": args.run_tag,
        "addr_key": args.addr_key,
        "address": {"tau": args.address_tau, "overlap": args.address_overlap,
                    "soft_match": args.soft_match, "min_mentions": args.min_mentions},
        "retrieval": {"topk": args.topk, "hop": args.hop, "hop_min": args.hop_min,
                      "k_gap": args.k_gap, "subject_filter": args.subject_filter},
        "thresholds": {"min_precision": args.min_precision,
                       "min_separation": args.min_separation,
                       "max_crowding": args.max_crowding,
                       "max_foreign": args.max_foreign},
        "mind_params": mind_params,
        "gates": {
            "G_precision_holds": g_precision,
            "G_return_path_separates": g_separation,
            "G_addresses_stay_distinct": g_crowding,
            "G_k_covers_the_address": last["k_covers_address"] >= 0.90,
            "G_ties_exist_naturally": g_ties,
            "G_mind_does_not_grow": g_mind_constant,
        },
        "slopes": {
            "slots": [r["n_slots"] for r in out_rungs],
            "precision": [r["retrieval_precision"] for r in out_rungs],
            "return_path_separation": [r["return_path_separation"] for r in out_rungs],
            "address_crowding": [r["address_crowding"] for r in out_rungs],
            "member_crowding": [r["member_crowding"] for r in out_rungs],
            "two_channel_crowding": [r["two_channel_crowding"] for r in out_rungs],
            "foreign_member_rate": [r["foreign_member_rate"] for r in out_rungs],
            "distinct_anchors": [r["distinct_anchors"] for r in out_rungs],
            "anchor_only_crowding": [r["anchor_only_crowding"] for r in out_rungs],
            # How fast anchors arrive per character of corpus. Below 1 the corpus is running
            # out of new names, which is the limit that fixes itself; at or near 1 it is not,
            # and the space is the only thing that can give.
            "anchor_growth_exponent": (
                math.log(max(1, last["distinct_anchors"]) / max(1, first["distinct_anchors"]))
                / math.log(last["chars"] / max(1, first["chars"]))
                if len(out_rungs) > 1 and last["chars"] != first["chars"] else float("nan")),
            "k_covers_address": [r["k_covers_address"] for r in out_rungs],
            "tie_share": [r["families_natural"]["tie"] for r in out_rungs],
            "slots_growth": last["n_slots"] / max(1, first["n_slots"]),
        },
        "rungs": out_rungs,
        "fp_version": s271.fp_version(),
        "reference_280": {"n_addresses": 187, "n_slots": 530,
                          "retrieval_precision_train": 0.8031194295900178,
                          "held_out_reward": 0.7043859649122806},
        "note": (
            "The tape side of scale, with no mind attached. 280's numbers describe an index of "
            "530 slots, and the concept's invariant is that the mind does not grow with the "
            "knowledge - so if anything breaks with N it breaks in the addressing. Four things "
            "are measured at each rung and none of them needs training: how much of a retrieved "
            "list belongs to the address it was retrieved for; whether the return path still "
            "tells a settled value from a value taken elsewhere in the tape, which is the check "
            "meant to replace G_answer_is_slot once an answer stops being a slot index; how "
            "close the nearest other address key gets, since distinct entities crowd as the "
            "tape fills; and what the family mix is when nothing quotas it, because 280's tie "
            "rate is an artefact of --min-per-family and not a fact about text. Slopes are "
            "reported beside the gates: a metric that clears its floor while falling steeply is "
            "the case that decides whether this scales or merely has not broken yet."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(), "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"stage283_decision{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(json.dumps({"overall": overall, "gates": out["gates"], "slopes": out["slopes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
