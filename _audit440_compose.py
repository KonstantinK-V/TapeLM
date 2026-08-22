"""440: composition world. Two 436 operations, not a new scorer."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

OUT = Path("results/_stage440_compose.json")
CAP = 8


def _pad(k):
    return " " + " ".join(f"z{k}x{j}" for j in range(20))


def designed():
    p = ["red cat sat APPLES on the mat" + _pad(i) for i in range(3)]
    p.append("red cat sat ORANGES on the mat" + _pad(3))
    r = ["kids like SWEET APPLES today yes" + _pad(20 + i) for i in range(3)]
    r.append("kids like SOUR APPLES today yes" + _pad(23))
    q = []
    for i, f in enumerate(("PEARS", "PLUMS", "PEARS", "PLUMS")):
        q.append(f"blue dog lay {f} in the sun" + _pad(10 + i))
    return p + r + q


def think_slot(s, slots_at, place, value, line, rng):
    p, li, v = place[s], line[s], value[s]
    foreign = [t for t in slots_at[p] if t != s and line[t] != li]
    rng.shuffle(foreign)
    offer = foreign[:CAP]
    if not offer:
        return None
    maj_v = Counter(value[t] for t in offer).most_common(1)[0][0]
    if maj_v != v:
        return None
    t = next(x for x in offer if value[x] == maj_v)
    working = {("work", 0): t}
    if working[("work", 0)] != t:
        return None
    return t


def think_place(slots, value, rng):
    if len(slots) < 2:
        return None
    rng.shuffle(slots)
    offer = slots[:CAP]
    maj_v, n_maj = Counter(value[t] for t in offer).most_common(1)[0]
    if n_maj * 2 <= len(offer):
        return None
    t = next(x for x in offer if value[x] == maj_v)
    working = {("work", 1): t}
    if working[("work", 1)] != t:
        return None
    return t


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
    for s in range(n):
        slots_at[place[s]].append(s)
    by_key = defaultdict(set)
    for s in range(n):
        for k in keys[s]:
            by_key[k].add(place[s])
    n_p = n_q = n_comp = n_ctrl = n_qref = n_hop = 0
    for s in range(n):
        v = value[s]
        pin1 = think_slot(s, slots_at, place, value, line, rng)
        if v in ("PEARS", "PLUMS"):
            n_q += 1
            n_qref += int(pin1 is None)
            continue
        if v != "APPLES":
            continue
        n_p += 1
        if pin1 is None:
            continue
        n_ctrl += int(value[pin1] == "APPLES")
        got = value[pin1]
        cands = by_key.get(got, set()) - {place[s]}
        if len(cands) != 1:
            continue
        R = next(iter(cands))
        pin2 = think_place(list(slots_at[R]), value, rng)
        if pin2 is None:
            continue
        n_hop += 1
        n_comp += int(value[pin2] == "SWEET")
    return dict(
        n=n, n_p=n_p, n_q=n_q,
        ctrl_apples=n_ctrl / max(n_p, 1),
        composed_sweet=n_comp / max(n_p, 1),
        refuse_q=n_qref / max(n_q, 1),
        hop2_rate=n_hop / max(n_p, 1),
        n_places=len(slots_at),
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
    void = (rep["n_p"] == 0) or (rep["n_q"] == 0)
    gate = ((not void)
            and (rep["composed_sweet"] == 1.0)
            and (rep["ctrl_apples"] == 1.0)
            and (rep["refuse_q"] == 1.0)
            and (rep["hop2_rate"] == 1.0))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate), **rep)
    print(f"P {rep['n_p']} Q {rep['n_q']} places {rep['n_places']}")
    print(f"436-only APPLES {rep['ctrl_apples']:.2f}  "
          f"composed SWEET {rep['composed_sweet']:.2f}  "
          f"Q refuse {rep['refuse_q']:.2f}  hop2 {rep['hop2_rate']:.2f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: P or Q missing. Composition has no world.")
    elif gate:
        print("\nGO COMPOSE: hop1 pins APPLES; hop2 via APPLES-as-address reads SWEET.")
    else:
        print("\nSTOP: operations did not compose on a world where they should.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
