"""441: hop1→hop2→hop3. Same 440 ops. Break after each hop is a control."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit440_compose import think_place, think_slot

OUT = Path("results/_stage441_hop3.json")


def _pad(k):
    return " " + " ".join(f"z{k}x{j}" for j in range(20))


def designed():
    p = ["red cat sat APPLES on the mat" + _pad(i) for i in range(3)]
    p.append("red cat sat ORANGES on the mat" + _pad(3))
    r = ["kids like SWEET APPLES today yes" + _pad(20 + i) for i in range(3)]
    r.append("kids like SOUR APPLES today yes" + _pad(23))
    t = ["barns store FRESH SWEET fruit now" + _pad(30 + i) for i in range(3)]
    t.append("barns store STALE SWEET fruit now" + _pad(33))
    q = []
    for i, f in enumerate(("PEARS", "PLUMS", "PEARS", "PLUMS")):
        q.append(f"blue dog lay {f} in the sun" + _pad(10 + i))
    return p + r + t + q


def next_place(got, cur, by_key, place_of):
    cands = by_key.get(got, set()) - {place_of}
    if len(cands) != 1:
        return None
    return next(iter(cands))


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
    n_p = n_q = n1 = n2 = n3 = nq = n_h2 = n_h3 = 0
    diff12 = diff23 = 0
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
        R = next_place(v1, s, by_key, place[s])
        if R is None:
            continue
        pin2 = think_place(list(slots_at[R]), value, rng)
        if pin2 is None:
            continue
        n_h2 += 1
        v2 = value[pin2]
        n2 += int(v2 == "SWEET")
        diff12 += int(pin2 != pin1 and v2 != v1)
        T = next_place(v2, s, by_key, R)
        if T is None:
            continue
        pin3 = think_place(list(slots_at[T]), value, rng)
        if pin3 is None:
            continue
        n_h3 += 1
        v3 = value[pin3]
        n3 += int(v3 == "FRESH")
        diff23 += int(pin3 != pin2 and v3 != v2)
    return dict(
        n=n, n_p=n_p, n_q=n_q, n_places=len(slots_at),
        h1_apples=n1 / max(n_p, 1),
        h2_sweet=n2 / max(n_p, 1),
        h3_fresh=n3 / max(n_p, 1),
        refuse_q=nq / max(n_q, 1),
        hop2_rate=n_h2 / max(n_p, 1),
        hop3_rate=n_h3 / max(n_p, 1),
        hop_changes_12=diff12 / max(n_h2, 1),
        hop_changes_23=diff23 / max(n_h3, 1),
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
            and (rep["h1_apples"] == 1.0)
            and (rep["h2_sweet"] == 1.0)
            and (rep["h3_fresh"] == 1.0)
            and (rep["refuse_q"] == 1.0)
            and (rep["hop2_rate"] == 1.0)
            and (rep["hop3_rate"] == 1.0)
            and (rep["hop_changes_12"] == 1.0)
            and (rep["hop_changes_23"] == 1.0))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate), **rep)
    print(f"P {rep['n_p']} Q {rep['n_q']} places {rep['n_places']}")
    print(f"h1 APPLES {rep['h1_apples']:.2f}  h2 SWEET {rep['h2_sweet']:.2f}  "
          f"h3 FRESH {rep['h3_fresh']:.2f}")
    print(f"Q refuse {rep['refuse_q']:.2f}  hop2 {rep['hop2_rate']:.2f}  "
          f"hop3 {rep['hop3_rate']:.2f}  d12 {rep['hop_changes_12']:.2f}  "
          f"d23 {rep['hop_changes_23']:.2f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: P or Q missing.")
    elif gate:
        print("\nGO HOP3: APPLES -> SWEET -> FRESH, each hop a new cell. "
              "Q breaks the chain at hop1.")
    else:
        print("\nSTOP: chain did not hold on a world where each hop is written.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
