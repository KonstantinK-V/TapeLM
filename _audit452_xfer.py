"""452: POS transfer + NEG frozen fail + retrain."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit449_adv import build, filter_cands
from _audit451_learn import episode_start, finish, greedy_pick, options

OUT = Path("results/_stage452_xfer.json")


def _pad(k):
    return " " + " ".join(f"z{k}x{j}" for j in range(20))


def _tok(rng, used):
    while True:
        w = "w" + "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
        if w not in used:
            used.add(w)
            return w


def _lex(rng):
    u = set()
    keys = (
        "APPLES", "ORANGES", "SWEET", "SOUR", "FRESH", "STALE",
        "CRISP", "SOFT", "RIPE", "RAW", "DRY", "MUSH", "WET", "DAMP",
        "KEYA", "UNLOCK", "TAG", "FOG", "PEARS", "PLUMS",
        "u1", "u2", "u3", "u4",
        "red", "cat", "sat", "on", "the", "mat",
        "kids", "like", "today", "yes",
        "barns", "store", "fruit", "now",
        "aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj", "kk", "ll",
        "clerk", "wrote", "board", "desk", "shows", "label", "here",
        "vault", "keeps", "safe", "xx", "yy",
        "blue", "dog", "lay", "in", "sun",
        "n1", "n2", "n3", "n4",
    )
    return {k: _tok(rng, u) for k in keys}


def world(rng, *, neg=False):
    n = _lex(rng)
    ctx = [n["red"], n["cat"], n["sat"], n["on"], n["the"], n["mat"]]
    rng.shuffle(ctx)
    p = [f"{ctx[0]} {ctx[1]} {ctx[2]} {n['APPLES']} {ctx[3]} {ctx[4]} {ctx[5]}" + _pad(i) for i in range(3)]
    p.append(f"{ctx[0]} {ctx[1]} {ctx[2]} {n['ORANGES']} {ctx[3]} {ctx[4]} {ctx[5]}" + _pad(3))
    r = [f"{n['kids']} {n['like']} {n['SWEET']} {n['APPLES']} {n['today']} {n['yes']}" + _pad(20 + i) for i in range(3)]
    r.append(f"{n['kids']} {n['like']} {n['SOUR']} {n['APPLES']} {n['today']} {n['yes']}" + _pad(23))
    t = [f"{n['barns']} {n['store']} {n['FRESH']} {n['SWEET']} {n['fruit']} {n['now']}" + _pad(30 + i) for i in range(3)]
    t.append(f"{n['barns']} {n['store']} {n['STALE']} {n['SWEET']} {n['fruit']} {n['now']}" + _pad(33))
    c1 = [f"{n['aa']} {n['bb']} {n['cc']} {n['CRISP']} {n['FRESH']} {n['KEYA']} {n['TAG']}" + _pad(50 + i) for i in range(3)]
    c1.append(f"{n['aa']} {n['bb']} {n['cc']} {n['SOFT']} {n['FRESH']} {n['KEYA']} {n['TAG']}" + _pad(53))
    c2 = [f"{n['dd']} {n['ee']} {n['ff']} {n['RIPE']} {n['FRESH']} {n['KEYA']} {n['u1']}" + _pad(60 + i) for i in range(3)]
    c2.append(f"{n['dd']} {n['ee']} {n['ff']} {n['RAW']} {n['FRESH']} {n['KEYA']} {n['u1']}" + _pad(63))
    c3 = [f"{n['gg']} {n['hh']} {n['ii']} {n['DRY']} {n['FRESH']} {n['KEYA']} {n['u2']}" + _pad(70 + i) for i in range(3)]
    c3.append(f"{n['gg']} {n['hh']} {n['ii']} {n['MUSH']} {n['FRESH']} {n['KEYA']} {n['u2']}" + _pad(73))
    c4 = [f"{n['jj']} {n['kk']} {n['ll']} {n['WET']} {n['FRESH']} {n['u3']} {n['u4']}" + _pad(80 + i) for i in range(3)]
    c4.append(f"{n['jj']} {n['kk']} {n['ll']} {n['DAMP']} {n['FRESH']} {n['u3']} {n['u4']}" + _pad(83))
    noise = [f"{n['n1']} {n['n2']} {n['n3']} {n['FOG']} {n['n4']} {n['yes']}" + _pad(200 + i) for i in range(2)]
    hA = [f"{n['clerk']} {n['wrote']} {n['KEYA']} {n['red']} {n['board']} {n['now']}" + _pad(90 + i) for i in range(3)]
    hA.append(f"{n['clerk']} {n['wrote']} {n['FOG']} {n['red']} {n['board']} {n['now']}" + _pad(93))
    hB = [f"{n['desk']} {n['shows']} {n['UNLOCK']} {n['cat']} {n['label']} {n['here']}" + _pad(100 + i) for i in range(3)]
    hB.append(f"{n['desk']} {n['shows']} {n['FOG']} {n['cat']} {n['label']} {n['here']}" + _pad(103))
    opener = n["KEYA"] if neg else n["UNLOCK"]
    hT = [f"{n['vault']} {n['keeps']} {n['TAG']} {opener} {n['safe']} {n['here']}" + _pad(110 + i) for i in range(3)]
    hT.append(f"{n['vault']} {n['keeps']} {n['FOG']} {opener} {n['safe']} {n['here']}" + _pad(113))
    q = []
    for i, f in enumerate((n["PEARS"], n["PLUMS"], n["PEARS"], n["PLUMS"])):
        q.append(f"{n['blue']} {n['dog']} {n['lay']} {f} {n['in']} {n['sun']}" + _pad(10 + i))
    return p + r + t + c1 + c2 + c3 + c4 + hA + hB + hT + q + noise, n


def train_table(rng, n_worlds, *, neg):
    win, tot = defaultdict(int), defaultdict(int)
    for _ in range(n_worlds):
        lines, names = world(rng, neg=neg)
        g = build(lines)
        if g is None:
            continue
        for cands, visited, used_k, order, s in episode_start(g, rng, names["APPLES"]):
            opts = options(cands, order, g["by_key"], visited, used_k,
                           g["slots_at"], g["value"], rng)
            for rec in opts:
                k, H, pinH, vis1, used1 = rec
                n_hit, d1, ok = finish(
                    H, pinH, cands, set(), order, g["by_key"], vis1, used1,
                    g["slots_at"], g["value"], g["place_keys"], rng, names["CRISP"])
                tot[(n_hit, d1)] += 1
                win[(n_hit, d1)] += int(ok)
    return {sig: win[sig] / tot[sig] for sig in tot if tot[sig]}, len(tot)


def learned_pick(opts, cands, extra, place_keys, value, rate):
    best, br = None, -1.0
    for rec in opts:
        pinH = rec[2]
        n_hit = sum(1 for p in cands if value[pinH] in place_keys[p])
        d1 = len(filter_cands(cands, extra | {value[pinH]}, place_keys))
        r = rate.get((n_hit, d1), 0.0)
        if r > br:
            br, best = r, rec
    return best if br > 0 else None


def eval_table(rng, n_worlds, rate, *, neg):
    n_l = n_g = n_ep = 0
    for _ in range(n_worlds):
        lines, names = world(rng, neg=neg)
        g = build(lines)
        if g is None:
            continue
        for cands, visited, used_k, order, s in episode_start(g, rng, names["APPLES"]):
            opts = options(cands, order, g["by_key"], visited, used_k,
                           g["slots_at"], g["value"], rng)
            if len(opts) < 2:
                continue
            n_ep += 1
            rec = learned_pick(opts, cands, set(), g["place_keys"], g["value"], rate)
            if rec is not None:
                k, H, pinH, vis1, used1 = rec
                _, _, ok = finish(
                    H, pinH, cands, set(), order, g["by_key"], vis1, used1,
                    g["slots_at"], g["value"], g["place_keys"], rng, names["CRISP"])
                n_l += int(ok)
            grec = greedy_pick(opts, cands, set(), g["place_keys"], g["value"])
            if grec is not None:
                k, H, pinH, vis1, used1 = grec
                _, _, ok = finish(
                    H, pinH, cands, set(), order, g["by_key"], vis1, used1,
                    g["slots_at"], g["value"], g["place_keys"], rng, names["CRISP"])
                n_g += int(ok)
    return n_l / max(n_ep, 1), n_g / max(n_ep, 1), n_ep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train", type=int, default=20)
    ap.add_argument("--test", type=int, default=10)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pos_rate, n_sig = train_table(rng, args.train, neg=False)
    pos_l, pos_g, pos_ep = eval_table(rng, args.test, pos_rate, neg=False)
    neg_frozen, _, _ = eval_table(rng, args.test, pos_rate, neg=True)
    neg_rate, n_sig_n = train_table(rng, args.train, neg=True)
    neg_l, neg_g, neg_ep = eval_table(rng, args.test, neg_rate, neg=True)
    void = (pos_ep == 0) or (neg_ep == 0) or (n_sig < 2)
    gate = ((not void) and (pos_l == 1.0) and (pos_g == 0.0)
            and (neg_frozen == 0.0) and (neg_l == 1.0))
    rec = dict(
        seed=args.seed, void=bool(void), gate=bool(gate),
        pos_learned=pos_l, pos_greedy=pos_g, pos_ep=pos_ep, pos_sig=n_sig,
        pos_rate={f"{a},{b}": round(v, 3) for (a, b), v in pos_rate.items()},
        neg_frozen=neg_frozen, neg_retrain=neg_l, neg_greedy=neg_g,
        neg_ep=neg_ep, neg_sig=n_sig_n,
        neg_rate={f"{a},{b}": round(v, 3) for (a, b), v in neg_rate.items()},
    )
    print(f"POS  learned {pos_l:.2f} greedy {pos_g:.2f} ep {pos_ep} sig {n_sig} {rec['pos_rate']}")
    print(f"NEG  frozen {neg_frozen:.2f} retrain {neg_l:.2f} greedy {neg_g:.2f} ep {neg_ep}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: empty split or <2 signatures.")
    elif gate:
        print("\nGO XFER: POS table transfers; POS table fails NEG; retrain flips.")
    else:
        print("\nSTOP: transfer or negative-transfer failed.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
