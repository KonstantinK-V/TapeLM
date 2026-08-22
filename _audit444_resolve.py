"""444: 2-cand set + extra read (query-frame keys). Not a scorer."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit440_compose import think_place, think_slot

OUT = Path("results/_stage444_resolve.json")


def _pad(k):
    return " " + " ".join(f"z{k}x{j}" for j in range(20))


def designed():
    p = ["red cat sat APPLES on the mat" + _pad(i) for i in range(3)]
    p.append("red cat sat ORANGES on the mat" + _pad(3))
    r = ["kids like SWEET APPLES today yes" + _pad(20 + i) for i in range(3)]
    r.append("kids like SOUR APPLES today yes" + _pad(23))
    t = ["barns store FRESH SWEET fruit now" + _pad(30 + i) for i in range(3)]
    t.append("barns store STALE SWEET fruit now" + _pad(33))
    v = ["crates mark CRISP FRESH mat here" + _pad(50 + i) for i in range(3)]
    v.append("crates mark SOFT FRESH mat here" + _pad(53))
    w = ["shops sell RIPE FRESH witems gone" + _pad(60 + i) for i in range(3)]
    w.append("shops sell RAW FRESH witems gone" + _pad(63))
    q = []
    for i, f in enumerate(("PEARS", "PLUMS", "PEARS", "PLUMS")):
        q.append(f"blue dog lay {f} in the sun" + _pad(10 + i))
    return p + r + t + v + w + q


def next_cands(got, place_of, by_key):
    return by_key.get(got, set()) - {place_of}


def hop(got, cur_place, by_key, slots_at, value, rng):
    c = next_cands(got, cur_place, by_key)
    if len(c) != 1:
        return None, c, None
    R = next(iter(c))
    pin = think_place(list(slots_at[R]), value, rng)
    return pin, c, R


def measure(lines, rng):
    keep, toks, owner = tframes.frame_keep(lines, 3, 2)
    if not keep:
        return None
    place, value, line, keys = [], [], [], []
    for (w, left, right), ps in keep:
        name = f"{' '.join(left)}|{' '.join(right)}"
        ks = {x for x in list(left) + list(right) if x}
        for i in ps:
            place.append(name)
            value.append(toks[i])
            line.append(owner[i])
            keys.append(ks)
    n = len(place)
    slots_at = defaultdict(list)
    place_keys = {}
    for s in range(n):
        slots_at[place[s]].append(s)
        place_keys[place[s]] = keys[s]
    by_key = defaultdict(set)
    for s in range(n):
        for k in keys[s]:
            by_key[k].add(place[s])
    n_p = n_q = n1 = n2 = n3 = nq = 0
    n_h2 = n_h3 = n_two = n_ref = n_ok = n_ripe = 0
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
        n_h2 += 1
        v2 = value[pin2]
        n2 += int(v2 == "SWEET")
        pin3, _, T = hop(v2, R, by_key, slots_at, value, rng)
        if pin3 is None:
            continue
        n_h3 += 1
        v3 = value[pin3]
        n3 += int(v3 == "FRESH")
        c3 = next_cands(v3, T, by_key)
        if len(c3) == 2:
            n_two += 1
        if len(c3) != 2:
            n_ref += 1
            continue
        extra = place_keys[place[s]]
        hit = {p for p in c3 if place_keys[p] & extra}
        if len(hit) != 1:
            n_ref += 1
            continue
        U = next(iter(hit))
        pin4 = think_place(list(slots_at[U]), value, rng)
        if pin4 is None:
            n_ref += 1
            continue
        n_ok += int(value[pin4] == "CRISP")
        n_ripe += int(value[pin4] == "RIPE")
    return dict(
        n=n, n_p=n_p, n_q=n_q, n_places=len(slots_at),
        h1_apples=n1 / max(n_p, 1),
        h2_sweet=n2 / max(n_p, 1),
        h3_fresh=n3 / max(n_p, 1),
        hop2_rate=n_h2 / max(n_p, 1),
        hop3_rate=n_h3 / max(n_p, 1),
        two_cands=n_two / max(n_h3, 1),
        refuse_no_unique=n_ref / max(n_h3, 1),
        resolved_crisp=n_ok / max(n_p, 1),
        picked_ripe=n_ripe / max(n_p, 1),
        refuse_q=nq / max(n_q, 1),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rep = measure(designed(), rng)
    if rep is None:
        print("no tape")
        return 1
    void = (rep["n_p"] == 0) or (rep["two_cands"] < 1.0) or (rep["hop3_rate"] < 1.0)
    gate = ((not void)
            and (rep["h1_apples"] == 1.0)
            and (rep["h2_sweet"] == 1.0)
            and (rep["h3_fresh"] == 1.0)
            and (rep["resolved_crisp"] == 1.0)
            and (rep["picked_ripe"] == 0.0)
            and (rep["refuse_q"] == 1.0))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate), **rep)
    print(f"P {rep['n_p']} Q {rep['n_q']} places {rep['n_places']}  two {rep['two_cands']:.2f}")
    print(f"h1-3 {rep['h1_apples']:.0f}/{rep['h2_sweet']:.0f}/{rep['h3_fresh']:.0f}  "
          f"CRISP {rep['resolved_crisp']:.2f} RIPE {rep['picked_ripe']:.2f}  Q {rep['refuse_q']:.2f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: no 2-set after FRESH.")
    elif gate:
        print("\nGO RESOLVE: query-frame key picks CRISP, not RIPE.")
    else:
        print("\nSTOP: extra did not resolve, or picked RIPE.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
