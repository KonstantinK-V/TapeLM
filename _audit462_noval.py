"""462 TRACK R: drop value-count from the signature."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit449_adv import build
from _audit451_learn import options
from _audit454_cost import soon
from _audit456b_geo import starts_flat
from _audit458_keycut import FAM, filter_keys, loop_keys, teacher_keys

OUT = Path("results/_stage462_noval.json")


def sig(cands, H, pinH, g, order, visited, used_k, use_soon):
    hole = g["value"][pinH]
    d1 = len(filter_keys(cands, H, g["place_keys"], hole))
    s = soon(hole, order, g["by_key"], visited, cands, used_k,
             g["place_keys"]) if use_soon else 0
    return (d1, s)


def train(rng, n_each, use_soon):
    win, tot = defaultdict(float), defaultdict(int)
    for fn in FAM.values():
        for _ in range(n_each):
            lines, names = fn(rng)
            g = build(lines)
            if g is None:
                continue
            for cands, visited, used_k, order in starts_flat(g, rng, names["FRESH"]):
                opts = options(cands, order, g["by_key"], visited, used_k,
                               g["slots_at"], g["value"], rng)
                for rec in opts:
                    k, H, pinH, vis1, used1 = rec
                    c1 = filter_keys(cands, H, g["place_keys"], g["value"][pinH])
                    key = sig(cands, H, pinH, g, order, vis1, used1, use_soon)
                    hops, ok, _ = loop_keys(
                        g, c1, vis1, used1, list(order) + [g["value"][pinH]],
                        rng, lambda o, c, gg: teacher_keys(o, c, gg), names["CRISP"])
                    if key[0] == 1:
                        hops, ok = 0, True
                    R = (1.0 if ok else 0.0) - 0.05 * (hops + 1)
                    tot[key] += 1
                    win[key] += R
    return {k: win[k] / tot[k] for k in tot}


def make_pick(table, use_soon, order, g, visited, used_k):
    def pick(opts, cands, gg):
        best, br = None, -10 ** 9
        for rec in opts:
            H, pinH = rec[1], rec[2]
            vis1, used1 = rec[3], rec[4]
            key = sig(cands, H, pinH, gg, order, vis1, used1, use_soon)
            r = table.get(key, -1.0)
            if r > br:
                br, best = r, rec
        return best if br > -1.0 else None
    return pick


def eval_tab(rng, fn, n, table, use_soon):
    n_ok = n_ref = n_ep = hops_sum = 0
    for _ in range(n):
        lines, names = fn(rng)
        g = build(lines)
        if g is None:
            continue
        for cands, visited, used_k, order in starts_flat(g, rng, names["FRESH"]):
            n_ep += 1
            pick = make_pick(table, use_soon, order, g, visited, used_k)
            hops, ok, ref = loop_keys(g, set(cands), set(visited), set(used_k),
                                      list(order), rng, pick, names["CRISP"])
            hops_sum += hops
            n_ok += int(ok)
            n_ref += int(ref)
    return dict(
        ep=n_ep,
        pin=n_ok / max(n_ep, 1),
        refuse=n_ref / max(n_ep, 1),
        mean_hops=hops_sum / max(n_ep, 1),
    )


def match_455(yes):
    return ((yes["D1"]["pin"] == 1.0) and (yes["D1"]["mean_hops"] == 1.0)
            and (yes["D2"]["pin"] == 1.0) and (yes["D2"]["mean_hops"] == 2.0)
            and (yes["BOTH"]["pin"] == 1.0) and (yes["BOTH"]["mean_hops"] == 2.0)
            and (yes["D4"]["pin"] == 1.0) and (yes["D4"]["mean_hops"] == 4.0)
            and (yes["STOP"]["refuse"] == 1.0) and (yes["STOP"]["pin"] == 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--test", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tab_s = train(rng, args.train, True)
    tab_n = train(rng, args.train, False)
    yes = {k: eval_tab(rng, fn, args.test, tab_s, True) for k, fn in FAM.items()}
    no = {k: eval_tab(rng, fn, args.test, tab_n, False) for k, fn in FAM.items()}
    void = any(yes[k]["ep"] < 5 for k in FAM)
    gate = (not void) and match_455(yes) and (no["BOTH"]["mean_hops"] > 2.0)
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate),
               with_soon=yes, no_soon=no,
               n_keys_soon=len(tab_s), n_keys_nosoon=len(tab_n))
    print("sig=(d1_keys, soon)  n_hit(value) dropped")
    print("with soon")
    for k in FAM:
        r = yes[k]
        print(f"  {k:4} pin {r['pin']:.2f} ref {r['refuse']:.2f} hops {r['mean_hops']:.1f}")
    print("no soon  BOTH hops", round(no["BOTH"]["mean_hops"], 2),
          "D2 pin", round(no["D2"]["pin"], 2))
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: empty.")
    elif gate:
        print("\nGO NOVAL: key-cut ranks without counting value on cands; soon still does BOTH.")
    else:
        print("\nSTOP R: n_hit was load-bearing, or soon unused on BOTH.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
