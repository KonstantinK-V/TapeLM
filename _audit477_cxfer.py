"""477 C: hop3 compose (441) on two lex tapes. No Q. Result = next address."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit440_compose import think_place, think_slot

OUT = Path("results/_stage477_cxfer.json")


def _pad(k):
    return " " + " ".join(f"z{k}x{j}" for j in range(20))


def _tok(rng, used):
    while True:
        w = "w" + "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        if w not in used:
            used.add(w)
            return w


def world(rng):
    u = set()
    n = {k: _tok(rng, u) for k in (
        "APPLES", "ORANGES", "SWEET", "SOUR", "FRESH", "STALE", "PEARS", "PLUMS")}
    p = [f"red cat sat {n['APPLES']} on the mat" + _pad(i) for i in range(3)]
    p.append(f"red cat sat {n['ORANGES']} on the mat" + _pad(3))
    r = [f"kids like {n['SWEET']} {n['APPLES']} today yes" + _pad(20 + i) for i in range(3)]
    r.append(f"kids like {n['SOUR']} {n['APPLES']} today yes" + _pad(23))
    t = [f"barns store {n['FRESH']} {n['SWEET']} fruit now" + _pad(30 + i) for i in range(3)]
    t.append(f"barns store {n['STALE']} {n['SWEET']} fruit now" + _pad(33))
    q = []
    for i, f in enumerate((n["PEARS"], n["PLUMS"], n["PEARS"], n["PLUMS"])):
        q.append(f"blue dog lay {f} in the sun" + _pad(10 + i))
    return p + r + t + q, n


def next_place(got, cur_place, by_key):
    cands = by_key.get(got, set()) - {cur_place}
    if len(cands) != 1:
        return None
    return next(iter(cands))


def measure(lines, names, rng):
    keep, toks, owner = tframes.frame_keep(lines, 3, 2)
    if not keep:
        return None
    place, value, line, keys = [], [], [], []
    for (w, left, right), ps in keep:
        pname = f"{' '.join(left)}|{' '.join(right)}"
        ks = {x for x in list(left) + list(right) if x}
        for i in ps:
            place.append(pname)
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
    A, Sw, F = names["APPLES"], names["SWEET"], names["FRESH"]
    for s in range(n):
        v = value[s]
        pin1 = think_slot(s, slots_at, place, value, line, rng)
        if v in (names["PEARS"], names["PLUMS"]):
            n_q += 1
            nq += int(pin1 is None)
            continue
        if v != A:
            continue
        n_p += 1
        if pin1 is None:
            continue
        v1 = value[pin1]
        n1 += int(v1 == A)
        R = next_place(v1, place[s], by_key)
        if R is None:
            continue
        pin2 = think_place(list(slots_at[R]), value, rng)
        if pin2 is None:
            continue
        n_h2 += 1
        v2 = value[pin2]
        n2 += int(v2 == Sw)
        diff12 += int(pin2 != pin1 and v2 != v1)
        T = next_place(v2, R, by_key)
        if T is None:
            continue
        pin3 = think_place(list(slots_at[T]), value, rng)
        if pin3 is None:
            continue
        n_h3 += 1
        v3 = value[pin3]
        n3 += int(v3 == F)
        diff23 += int(pin3 != pin2 and v3 != v2)
    return dict(
        n_p=n_p, n_q=n_q, n_places=len(slots_at),
        h1=n1 / max(n_p, 1), h2=n2 / max(n_p, 1), h3=n3 / max(n_p, 1),
        refuse_q=nq / max(n_q, 1),
        hop2=n_h2 / max(n_p, 1), hop3=n_h3 / max(n_p, 1),
        d12=diff12 / max(n_h2, 1), d23=diff23 / max(n_h3, 1),
        apples=A, fresh=F,
    )


def ok_tape(r):
    return (r["h1"] == 1.0 and r["h2"] == 1.0 and r["h3"] == 1.0
            and r["refuse_q"] == 1.0 and r["hop2"] == 1.0 and r["hop3"] == 1.0
            and r["d12"] == 1.0 and r["d23"] == 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    la, na = world(random.Random(args.seed))
    lb, nb = world(random.Random(args.seed + 99))
    a = measure(la, na, random.Random(args.seed))
    b = measure(lb, nb, random.Random(args.seed + 1))
    if a is None or b is None:
        print("no tape")
        return 1
    void = a["n_p"] == 0 or a["n_q"] == 0 or b["n_p"] == 0 or b["n_q"] == 0
    names_diff = na["APPLES"] != nb["APPLES"] and na["FRESH"] != nb["FRESH"]
    gate = ((not void) and ok_tape(a) and ok_tape(b) and names_diff)
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate),
               A=a, B=b, names_diff=names_diff)
    print("A h1/h2/h3", round(a["h1"], 2), round(a["h2"], 2), round(a["h3"], 2),
          "Q", round(a["refuse_q"], 2))
    print("B h1/h2/h3", round(b["h1"], 2), round(b["h2"], 2), round(b["h3"], 2),
          "Q", round(b["refuse_q"], 2))
    print("disjoint names", names_diff)
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: P or Q missing.")
    elif gate:
        print("\nGO C: pin->address->pin x3 on foreign lex. No Q. Result is the next walk.")
    else:
        print("\nSTOP: compose did not transfer as a procedure.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
