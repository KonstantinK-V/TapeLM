"""448: choose the SEEK that shrinks the set. Not 447's left-to-right."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit440_compose import think_place, think_slot
from _audit447_seek import (
    designed_neg, designed_pos, filter_cands, hop, next_cands,
)

OUT = Path("results/_stage448_pick.json")


def all_seeks(order, by_key, visited, cands, used_k):
    out = []
    seen = set()
    for k in order:
        if k in used_k:
            continue
        places = by_key.get(k, set()) - visited - cands
        if len(places) != 1:
            continue
        H = next(iter(places))
        if H in seen:
            continue
        seen.add(H)
        out.append((k, H))
    return out


def pick(cands, extra, order, by_key, visited, used_k, slots_at, value, place_keys, rng):
    opts = all_seeks(order, by_key, visited, cands, used_k)
    best = []
    for k, H in opts:
        pinH = think_place(list(slots_at[H]), value, rng)
        if pinH is None:
            continue
        newc = filter_cands(cands, extra | {value[pinH]}, place_keys)
        score = len(newc)
        if score < len(cands):
            best.append((score, k, H, pinH, newc))
    if not best:
        return None, opts, None
    best.sort(key=lambda x: x[0])
    s0 = best[0][0]
    top = [x for x in best if x[0] == s0]
    targets = {next(iter(x[4])) if len(x[4]) == 1 else None for x in top}
    if len(targets) != 1 or None in targets:
        return None, opts, None
    return top[0], opts, top[0][4]


def run(lines, rng):
    keep, toks, owner = tframes.frame_keep(lines, 3, 2)
    if not keep:
        return None
    place, value, line, keys, korder = [], [], [], [], []
    for (w, left, right), ps in keep:
        name = f"{' '.join(left)}|{' '.join(right)}"
        ks = {x for x in list(left) + list(right) if x}
        ordk = list(left) + list(right)
        for i in ps:
            place.append(name)
            value.append(toks[i])
            line.append(owner[i])
            keys.append(ks)
            korder.append(ordk)
    n = len(place)
    slots_at = defaultdict(list)
    place_keys, place_ord = {}, {}
    for s in range(n):
        slots_at[place[s]].append(s)
        place_keys[place[s]] = keys[s]
        place_ord[place[s]] = korder[s]
    by_key = defaultdict(set)
    for s in range(n):
        for k in keys[s]:
            by_key[k].add(place[s])

    n_p = n_q = n1 = n2 = n3 = nq = n_h3 = n_two = n_ref = n_crisp = n_ripe = 0
    n_seek = n_choice = 0
    for s in range(n):
        v = value[s]
        pin1 = think_slot(s, slots_at, place, value, line, rng)
        if v in ("PEARS", "PLUMS"):
            n_q += 1
            nq += int(pin1 is None)
            continue
        if v != "APPLES":
            continue
        n_p += 1
        if pin1 is None:
            continue
        v1 = value[pin1]
        n1 += int(v1 == "APPLES")
        pin2, _, R = hop(v1, place[s], by_key, slots_at, value, rng)
        if pin2 is None:
            continue
        v2 = value[pin2]
        n2 += int(v2 == "SWEET")
        pin3, _, T = hop(v2, R, by_key, slots_at, value, rng)
        if pin3 is None:
            continue
        n_h3 += 1
        v3 = value[pin3]
        n3 += int(v3 == "FRESH")
        cands = next_cands(v3, T, by_key)
        if len(cands) == 2:
            n_two += 1
        if len(cands) != 2:
            n_ref += 1
            continue
        visited = {place[s], R, T}
        used_k = {v1, v2, v3}
        extra = set()
        order = list(place_ord[place[s]])
        chosen, opts, newc = pick(
            cands, extra, order, by_key, visited, used_k,
            slots_at, value, place_keys, rng)
        n_choice += len(opts)
        if chosen is None:
            n_ref += 1
            continue
        n_seek += 1
        _, _, _, _, newc = chosen
        if len(newc) != 1:
            n_ref += 1
            continue
        U = next(iter(newc))
        pin4 = think_place(list(slots_at[U]), value, rng)
        if pin4 is None:
            n_ref += 1
            continue
        n_crisp += int(value[pin4] == "CRISP")
        n_ripe += int(value[pin4] == "RIPE")
    return dict(
        n_p=n_p, n_q=n_q, n_places=len(slots_at),
        h1=n1 / max(n_p, 1), h2=n2 / max(n_p, 1), h3=n3 / max(n_p, 1),
        hop3=n_h3 / max(n_p, 1),
        two=n_two / max(n_h3, 1),
        mean_choice=n_choice / max(n_p, 1),
        mean_seek=n_seek / max(n_p, 1),
        refuse=n_ref / max(n_h3, 1),
        crisp=n_crisp / max(n_p, 1), ripe=n_ripe / max(n_p, 1),
        q=nq / max(n_q, 1),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    pos = run(designed_pos(), random.Random(args.seed))
    neg = run(designed_neg(), random.Random(args.seed + 1))
    if pos is None or neg is None:
        print("no tape")
        return 1
    void = (pos["two"] < 1.0) or (neg["two"] < 1.0) or (pos["mean_choice"] < 2.0)
    gate = ((not void)
            and (pos["mean_seek"] == 1.0) and (pos["crisp"] == 1.0) and (pos["ripe"] == 0.0)
            and (neg["refuse"] == 1.0) and (neg["crisp"] == 0.0)
            and (pos["q"] == 1.0) and (neg["q"] == 1.0))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate), pos=pos, neg=neg)
    print(f"POS  two {pos['two']:.0f} choice {pos['mean_choice']:.0f} seek {pos['mean_seek']:.0f} "
          f"CRISP {pos['crisp']:.0f} RIPE {pos['ripe']:.0f}")
    print(f"NEG  two {neg['two']:.0f} refuse {neg['refuse']:.0f} CRISP {neg['crisp']:.0f} Q {neg['q']:.0f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: fewer than 2 seek options, or no 2-set.")
    elif gate:
        print("\nGO PICK: among seeks, take the one that shrinks; 1 seek -> CRISP. NEG refuses.")
    else:
        print("\nSTOP: did not pick a unique shrink, or NEG leaked.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
