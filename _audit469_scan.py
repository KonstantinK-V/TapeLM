"""469: scanner + address from a foreign tape. No apples."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from _audit440_compose import think_place
from _audit449_adv import build
from _audit451_learn import options
from _audit456b_geo import starts_flat
from _audit458_keycut import filter_keys, loop_keys, world_d2
from _audit463_trap import teacher_trap

OUT = Path("results/_stage469_scan.json")


def scan(g):
    n_p = n_pin = n_mix = n_df1 = 0
    addrs = []
    for P, slots in g["slots_at"].items():
        n_p += 1
        vs = [g["value"][s] for s in slots]
        if len(slots) < 2:
            n_df1 += 1
            continue
        maj, c = Counter(vs).most_common(1)[0]
        if c >= 3:
            n_pin += 1
            addrs.append(P)
        else:
            n_mix += 1
    return dict(places=n_p, pin=n_pin, mixed=n_mix, df1=n_df1, addrs=addrs)


def walk_d2(g, names, rng):
    n_ok = n_ref = n_ep = hops_sum = 0
    for cands, visited, used_k, order in starts_flat(g, rng, names["FRESH"]):
        n_ep += 1
        hops, ok, ref = loop_keys(
            g, set(cands), set(visited), set(used_k), list(order),
            rng, teacher_trap, names["CRISP"])
        hops_sum += hops
        n_ok += int(ok)
        n_ref += int(ref)
    return dict(
        ep=n_ep,
        pin=n_ok / max(n_ep, 1),
        refuse=n_ref / max(n_ep, 1),
        mean_hops=hops_sum / max(n_ep, 1),
    )


def train_QH(g, names, rng, n_ep):
    table = {}
    tot, win = defaultdict(int), defaultdict(float)
    pack = None
    for p in starts_flat(g, random.Random(0), names["FRESH"]):
        pack = p
        break
    if pack is None:
        return table
    for i in range(n_ep):
        eps = max(0.05, 0.4 * (1.0 - i / n_ep))
        cands, visited, used_k, order = pack
        cands, visited, used_k, order = set(cands), set(visited), set(used_k), list(order)
        hops, traj = 0, []
        while len(cands) > 1:
            opts = options(cands, order, g["by_key"], visited, used_k,
                           g["slots_at"], g["value"], rng)
            rec = None
            if opts and rng.random() < eps:
                rec = rng.choice(opts)
            elif opts:
                best, br = None, -1e9
                for r in opts:
                    qv = table.get(r[1], 0.0)
                    if qv > br:
                        br, best = qv, r
                rec = best if br > 0 else None
            if rec is None:
                break
            H, pinH = rec[1], rec[2]
            hole = g["value"][pinH]
            cands = filter_keys(cands, H, g["place_keys"], hole)
            visited, used_k = rec[3], rec[4]
            order = list(order) + [hole]
            traj.append(H)
            hops += 1
            if hops > 8:
                break
        ok = False
        if len(cands) == 1:
            pin = think_place(list(g["slots_at"][next(iter(cands))]), g["value"], rng)
            ok = pin is not None and g["value"][pin] == names["CRISP"]
        G = (1.0 if ok else 0.0) - 0.05 * max(hops, 1)
        for H in traj:
            tot[H] += 1
            win[H] += G
            table[H] = win[H] / tot[H]
    return table


def eval_QH(g, names, rng, table):
    n_ok = n_ep = 0
    for cands, visited, used_k, order in starts_flat(g, rng, names["FRESH"]):
        n_ep += 1
        hops, ok, ref = loop_keys(
            g, set(cands), set(visited), set(used_k), list(order),
            rng,
            lambda o, c, gg: (
                max(o, key=lambda r: table.get(r[1], -1.0))
                if o and max(table.get(r[1], -1.0) for r in o) > 0 else None
            ),
            names["CRISP"])
        n_ok += int(ok)
    return n_ok / max(n_ep, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    lines_a, na = world_d2(random.Random(args.seed))
    lines_b, nb = world_d2(random.Random(args.seed + 99))
    ga, gb = build(lines_a), build(lines_b)
    sa, sb = scan(ga), scan(gb)
    wa = walk_d2(ga, na, random.Random(1))
    wb = walk_d2(gb, nb, random.Random(1))
    qh = train_QH(ga, na, random.Random(args.seed), 200)
    qb = eval_QH(gb, nb, random.Random(2), qh)
    qa = eval_QH(ga, na, random.Random(2), qh)
    inter = set(sa["addrs"]) & set(sb["addrs"])
    tok_ok = na["CRISP"] != nb["CRISP"]
    void = (sa["pin"] < 1) or (sb["pin"] < 1) or (not tok_ok)
    gate = ((not void)
            and (len(inter) == 0)
            and (wa["pin"] == 1.0) and (wa["mean_hops"] == 2.0)
            and (wb["pin"] == 1.0) and (wb["mean_hops"] == 2.0)
            and (qb == 0.0))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate),
               scan_a={k: sa[k] for k in ("places", "pin", "mixed", "df1")},
               scan_b={k: sb[k] for k in ("places", "pin", "mixed", "df1")},
               walk_a=wa, walk_b=wb,
               qh_on_a=qa, qh_on_b=qb,
               crisp_equal=not tok_ok, addr_overlap=len(inter),
               n_qh=len(qh))
    print("scan A", sa["places"], "pin", sa["pin"], "mix", sa["mixed"],
          "| B", sb["places"], "pin", sb["pin"])
    print("walk A pin", wa["pin"], "hops", wa["mean_hops"],
          "| B pin", wb["pin"], "hops", wb["mean_hops"])
    print("Q[H] A", round(qa, 2), "B", round(qb, 2),
          "CRISP same", not tok_ok, "addr^", len(inter))
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: scanner empty or lex collided.")
    elif gate:
        print("\nGO SCAN: walk is the skill; B gets B-addresses; Q[H] does not ride along.")
    else:
        print("\nSTOP: walk failed on B, addresses leaked, or Q[H] transferred.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
