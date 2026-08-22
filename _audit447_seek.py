"""447: refuse is SEEK, not a wall. Iterate until unique or exhausted.
POS  SEEK1 'red' → NOISE (still 2) → SEEK2 'cat' → TAG → CRISP
     GATE mean_seek==2
NEG  no unique unread → REFUSE
VOID if first seek already unique
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit440_compose import think_place, think_slot

OUT = Path("results/_stage447_seek.json")


def _pad(k):
    return " " + " ".join(f"z{k}x{j}" for j in range(20))


def _chain():
    p = ["red cat sat APPLES on the mat" + _pad(i) for i in range(3)]
    p.append("red cat sat ORANGES on the mat" + _pad(3))
    r = ["kids like SWEET APPLES today yes" + _pad(20 + i) for i in range(3)]
    r.append("kids like SOUR APPLES today yes" + _pad(23))
    t = ["barns store FRESH SWEET fruit now" + _pad(30 + i) for i in range(3)]
    t.append("barns store STALE SWEET fruit now" + _pad(33))
    q = []
    for i, f in enumerate(("PEARS", "PLUMS", "PEARS", "PLUMS")):
        q.append(f"blue dog lay {f} in the sun" + _pad(10 + i))
    return p, r, t, q


def designed_pos():
    p, r, t, q = _chain()
    v = ["crates mark xx CRISP FRESH TAG NOISE" + _pad(50 + i) for i in range(3)]
    v.append("crates mark xx SOFT FRESH TAG NOISE" + _pad(53))
    w = ["shops sell yy RIPE FRESH NOISE gone" + _pad(60 + i) for i in range(3)]
    w.append("shops sell yy RAW FRESH NOISE gone" + _pad(63))
    h1 = ["clerk wrote NOISE red board now" + _pad(70 + i) for i in range(3)]
    h1.append("clerk wrote FOG red board now" + _pad(73))
    h2 = ["desk shows TAG cat label here" + _pad(80 + i) for i in range(3)]
    h2.append("desk shows FOG cat label here" + _pad(83))
    return p + r + t + v + w + h1 + h2 + q


def designed_neg():
    p, r, t, q = _chain()
    v = ["crates mark CRISP FRESH red here" + _pad(50 + i) for i in range(3)]
    v.append("crates mark SOFT FRESH red here" + _pad(53))
    w = ["shops sell RIPE FRESH red gone" + _pad(60 + i) for i in range(3)]
    w.append("shops sell RAW FRESH red gone" + _pad(63))
    return p + r + t + v + w + q


def next_cands(got, place_of, by_key):
    return by_key.get(got, set()) - {place_of}


def hop(got, cur, by_key, slots_at, value, rng):
    c = next_cands(got, cur, by_key)
    if len(c) != 1:
        return None, c, None
    R = next(iter(c))
    pin = think_place(list(slots_at[R]), value, rng)
    return pin, c, R


def filter_cands(cands, extra, place_keys):
    out = set(cands)
    for k in extra:
        hit = {p for p in out if k in place_keys[p]}
        if len(hit) == 1:
            out = hit
        elif len(hit) == 0:
            return set()
    return out


def next_seek(order, by_key, visited, cands, used_k):
    for k in order:
        if k in used_k:
            continue
        places = by_key.get(k, set()) - visited - cands
        if len(places) == 1:
            return k, next(iter(places))
    return None, None


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
    place_keys = {}
    place_ord = {}
    for s in range(n):
        slots_at[place[s]].append(s)
        place_keys[place[s]] = keys[s]
        place_ord[place[s]] = korder[s]
    by_key = defaultdict(set)
    for s in range(n):
        for k in keys[s]:
            by_key[k].add(place[s])
    n_p = n_q = n1 = n2 = n3 = nq = n_h3 = n_two = n_ref = n_crisp = n_ripe = 0
    n_seek = n_seek1_still = 0
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
        seeks = 0
        pinned = None
        while len(cands) > 1:
            k, H = next_seek(order, by_key, visited, cands, used_k)
            if H is None:
                break
            used_k.add(k)
            pinH = think_place(list(slots_at[H]), value, rng)
            if pinH is None:
                break
            seeks += 1
            visited.add(H)
            extra.add(value[pinH])
            cands = filter_cands(cands, extra, place_keys)
            if seeks == 1 and len(cands) > 1:
                n_seek1_still += 1
            if len(cands) == 1:
                U = next(iter(cands))
                pinned = think_place(list(slots_at[U]), value, rng)
                break
        n_seek += seeks
        if pinned is None:
            n_ref += 1
            continue
        n_crisp += int(value[pinned] == "CRISP")
        n_ripe += int(value[pinned] == "RIPE")
    return dict(
        n_p=n_p, n_q=n_q, n_places=len(slots_at),
        h1=n1 / max(n_p, 1), h2=n2 / max(n_p, 1), h3=n3 / max(n_p, 1),
        hop3=n_h3 / max(n_p, 1), two=n_two / max(n_h3, 1),
        mean_seek=n_seek / max(n_p, 1),
        first_seek_unresolved=n_seek1_still / max(n_p, 1),
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
    void = (pos["two"] < 1.0) or (neg["two"] < 1.0) or (pos["first_seek_unresolved"] < 1.0)
    gate = ((not void)
            and (pos["mean_seek"] == 2.0) and (pos["crisp"] == 1.0) and (pos["ripe"] == 0.0)
            and (neg["refuse"] == 1.0) and (neg["crisp"] == 0.0)
            and (pos["q"] == 1.0) and (neg["q"] == 1.0))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate), pos=pos, neg=neg)
    print(f"POS  two {pos['two']:.0f} seek {pos['mean_seek']:.1f} still1 {pos['first_seek_unresolved']:.0f} "
          f"CRISP {pos['crisp']:.0f} RIPE {pos['ripe']:.0f}")
    print(f"NEG  two {neg['two']:.0f} refuse {neg['refuse']:.0f} CRISP {neg['crisp']:.0f} Q {neg['q']:.0f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: first seek already resolved, or no 2-set. Iteration untested.")
    elif gate:
        print("\nGO SEEK: refuse->read NOISE (still 2)->read TAG->CRISP. NEG exhausts -> refuse.")
    else:
        print("\nSTOP: POS did not need 2 seeks, or NEG picked.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
