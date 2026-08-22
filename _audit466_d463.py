"""466 TRACK D: bootstrap 463 (key-cut+trap), no teacher."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit440_compose import think_place
from _audit449_adv import build
from _audit451_learn import options
from _audit456b_geo import starts_flat
from _audit458_keycut import FAM, filter_keys
from _audit463_trap import trap_of

OUT = Path("results/_stage466_d463.json")


def sig_rec(cands, H, pinH, g):
    hole = g["value"][pinH]
    d1 = len(filter_keys(cands, H, g["place_keys"], hole))
    tr = trap_of(cands, hole, g["place_keys"])
    return (d1, tr)


def eps_pick(opts, cands, g, table, rng, eps):
    if not opts:
        return None
    if rng.random() < eps:
        return rng.choice(opts)
    best, br = None, -10 ** 9
    for rec in opts:
        key = sig_rec(cands, rec[1], rec[2], g)
        r = table.get(key, 0.0)
        if r > br:
            br, best = r, rec
    if br <= 0:
        return None
    return best


def rollout(g, cands, visited, used_k, order, rng, table, eps, target):
    hops = 0
    traj = []
    while len(cands) > 1:
        opts = options(cands, order, g["by_key"], visited, used_k,
                       g["slots_at"], g["value"], rng)
        rec = eps_pick(opts, cands, g, table, rng, eps)
        if rec is None:
            return hops, False, True, traj
        k, H, pinH, vis1, used1 = rec
        key = sig_rec(cands, H, pinH, g)
        hole = g["value"][pinH]
        nxt = filter_keys(cands, H, g["place_keys"], hole)
        shrink = len(nxt) < len(cands)
        cands = nxt
        visited, used_k = vis1, used1
        order = list(order) + [hole]
        hops += 1
        traj.append((key, shrink))
        if hops > 8:
            return hops, False, True, traj
    if len(cands) != 1:
        return hops, False, True, traj
    pin = think_place(list(g["slots_at"][next(iter(cands))]), g["value"], rng)
    ok = pin is not None and g["value"][pin] == target
    return hops, ok, False, traj


def returns(traj, ok, ref):
    term = 1.0 if ok else (0.0 if ref else -1.0)
    out = []
    G = term
    for key, shrink in reversed(traj):
        G = G - 0.05 + (0.1 if shrink else 0.0)
        out.append((key, G))
    out.reverse()
    return out


def eval_greedy(rng, fn, n, table):
    n_ok = n_ref = n_ep = hops_sum = 0
    for _ in range(n):
        lines, names = fn(rng)
        g = build(lines)
        if g is None:
            continue
        for cands, visited, used_k, order in starts_flat(g, rng, names["FRESH"]):
            n_ep += 1
            hops, ok, ref, _ = rollout(
                g, set(cands), set(visited), set(used_k), list(order),
                rng, table, 0.0, names["CRISP"])
            hops_sum += hops
            n_ok += int(ok)
            n_ref += int(ref)
    return dict(
        ep=n_ep,
        pin=n_ok / max(n_ep, 1),
        refuse=n_ref / max(n_ep, 1),
        mean_hops=hops_sum / max(n_ep, 1),
    )


def train_arm(rng, n_steps, update):
    table = {}
    tot, win = defaultdict(int), defaultdict(float)
    mix = ("D1", "D2", "D4", "BOTH", "STOP")
    for i in range(n_steps):
        name = mix[i % 5]
        eps = max(0.05, 0.5 * (1.0 - i / n_steps)) if update else 0.0
        lines, names = FAM[name](rng)
        g = build(lines)
        if g is None:
            continue
        for cands, visited, used_k, order in starts_flat(g, rng, names["FRESH"]):
            hops, ok, ref, traj = rollout(
                g, set(cands), set(visited), set(used_k), list(order),
                rng, table, eps, names["CRISP"])
            if update:
                for key, G in returns(traj, ok, ref):
                    tot[key] += 1
                    win[key] += G
                    table[key] = win[key] / tot[key]
    return table


def fit(yes):
    return ((yes["D1"]["pin"] == 1.0) and (yes["D1"]["mean_hops"] == 1.0)
            and (yes["D2"]["pin"] == 1.0) and (yes["D2"]["mean_hops"] == 2.0)
            and (yes["D4"]["pin"] == 1.0) and (yes["D4"]["mean_hops"] == 4.0)
            and (yes["STOP"]["refuse"] == 1.0) and (yes["STOP"]["pin"] == 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--test", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng0 = random.Random(args.seed)
    empty = {}
    t0 = {k: eval_greedy(rng0, fn, args.test, empty) for k, fn in FAM.items()}
    learned = train_arm(random.Random(args.seed), args.steps, True)
    ctrl = train_arm(random.Random(args.seed + 1), args.steps, False)
    rng1 = random.Random(args.seed + 9)
    t1 = {k: eval_greedy(rng1, fn, args.test, learned) for k, fn in FAM.items()}
    t1c = {k: eval_greedy(rng1, fn, args.test, ctrl) for k, fn in FAM.items()}
    void = t0["D2"]["ep"] < 5
    gate = ((not void)
            and (t0["D2"]["pin"] == 0.0)
            and fit(t1)
            and (t1c["D2"]["pin"] == 0.0))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate),
               t0=t0, t1=t1, control=t1c, n_keys=len(learned))
    print("t0 (empty)")
    for k in FAM:
        r = t0[k]
        print(f"  {k:4} pin {r['pin']:.2f} ref {r['refuse']:.2f} hops {r['mean_hops']:.1f}")
    print("t1 learned")
    for k in FAM:
        r = t1[k]
        print(f"  {k:4} pin {r['pin']:.2f} ref {r['refuse']:.2f} hops {r['mean_hops']:.1f}")
    print("control D2 pin", round(t1c["D2"]["pin"], 2),
          "BOTH hops", round(t1["BOTH"]["mean_hops"], 2))
    print(f"VOID {void}  GATE_D {gate}  keys {len(learned)}")
    if void:
        print("\nVOID: empty.")
    elif gate:
        print("\nGO D463: e-bootstrap acquired key-cut+trap without teacher.")
    else:
        print("\nSTOP D: 463 policy not acquired from rollouts, or control not refuse.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
