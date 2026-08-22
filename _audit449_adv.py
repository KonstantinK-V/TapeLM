"""449: greedy 4→3 refuse vs 2-step UNLOCK→TAG CRISP."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit440_compose import think_place, think_slot
from _audit447_seek import hop, next_cands
from _audit448_pick import all_seeks

OUT = Path("results/_stage449_adv.json")


def _pad(k):
    return " " + " ".join(f"z{k}x{j}" for j in range(20))


def designed():
    p = ["red cat sat APPLES on the mat" + _pad(i) for i in range(3)]
    p.append("red cat sat ORANGES on the mat" + _pad(3))
    r = ["kids like SWEET APPLES today yes" + _pad(20 + i) for i in range(3)]
    r.append("kids like SOUR APPLES today yes" + _pad(23))
    t = ["barns store FRESH SWEET fruit now" + _pad(30 + i) for i in range(3)]
    t.append("barns store STALE SWEET fruit now" + _pad(33))
    c1 = ["aa bb cc CRISP FRESH KEYA TAG" + _pad(50 + i) for i in range(3)]
    c1.append("aa bb cc SOFT FRESH KEYA TAG" + _pad(53))
    c2 = ["dd ee ff RIPE FRESH KEYA u1" + _pad(60 + i) for i in range(3)]
    c2.append("dd ee ff RAW FRESH KEYA u1" + _pad(63))
    c3 = ["gg hh ii DRY FRESH KEYA u2" + _pad(70 + i) for i in range(3)]
    c3.append("gg hh ii MUSH FRESH KEYA u2" + _pad(73))
    c4 = ["jj kk ll WET FRESH u3 u4" + _pad(80 + i) for i in range(3)]
    c4.append("jj kk ll DAMP FRESH u3 u4" + _pad(83))
    hA = ["clerk wrote KEYA red board now" + _pad(90 + i) for i in range(3)]
    hA.append("clerk wrote FOG red board now" + _pad(93))
    hB = ["desk shows UNLOCK cat label here" + _pad(100 + i) for i in range(3)]
    hB.append("desk shows FOG cat label here" + _pad(103))
    hT = ["vault keeps TAG UNLOCK safe here" + _pad(110 + i) for i in range(3)]
    hT.append("vault keeps FOG UNLOCK safe here" + _pad(113))
    q = []
    for i, f in enumerate(("PEARS", "PLUMS", "PEARS", "PLUMS")):
        q.append(f"blue dog lay {f} in the sun" + _pad(10 + i))
    return p + r + t + c1 + c2 + c3 + c4 + hA + hB + hT + q


def filter_cands(cands, extra, place_keys):
    out = set(cands)
    for k in extra:
        hit = {p for p in out if k in place_keys[p]}
        if len(hit) == 0:
            continue
        if len(hit) < len(out):
            out = hit
    return out


def build(lines):
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
    slots_at = defaultdict(list)
    place_keys, place_ord = {}, {}
    for s in range(len(place)):
        slots_at[place[s]].append(s)
        place_keys[place[s]] = keys[s]
        place_ord[place[s]] = korder[s]
    by_key = defaultdict(set)
    for s in range(len(place)):
        for k in keys[s]:
            by_key[k].add(place[s])
    return dict(place=place, value=value, line=line, n=len(place),
                slots_at=slots_at, place_keys=place_keys, place_ord=place_ord,
                by_key=by_key)


def greedy(cands, extra, order, by_key, visited, used_k, slots_at, value, place_keys, rng):
    opts = all_seeks(order, by_key, visited, cands, used_k)
    best = []
    for k, H in opts:
        pinH = think_place(list(slots_at[H]), value, rng)
        if pinH is None:
            continue
        newc = filter_cands(cands, extra | {value[pinH]}, place_keys)
        if len(newc) < len(cands):
            best.append((len(newc), value[pinH], newc))
    if not best:
        return None, opts
    best.sort(key=lambda x: x[0])
    return best[0][2], opts


def deep(cands, extra, order, by_key, visited, used_k, slots_at, value, place_keys, rng):
    opts = all_seeks(order, by_key, visited, cands, used_k)
    win = []
    for k, H in opts:
        pinH = think_place(list(slots_at[H]), value, rng)
        if pinH is None:
            continue
        extra1 = extra | {value[pinH]}
        c1 = filter_cands(cands, extra1, place_keys)
        vis1 = set(visited) | {H}
        used1 = set(used_k) | {k}
        if len(c1) == 1:
            win.append((1, c1))
            continue
        opts2 = all_seeks(order + [value[pinH]], by_key, vis1, c1, used1)
        for k2, H2 in opts2:
            pin2 = think_place(list(slots_at[H2]), value, rng)
            if pin2 is None:
                continue
            c2 = filter_cands(c1, extra1 | {value[pin2]}, place_keys)
            if len(c2) == 1:
                win.append((2, c2))
                break
    targets = {next(iter(c)) for nstep, c in win if len(c) == 1}
    if len(targets) != 1:
        return None, opts
    return next(iter(t for nstep, t in win if len(t) == 1 and next(iter(t)) in targets)), opts


def run(lines, rng):
    g = build(lines)
    if g is None:
        return None
    place, value, line = g["place"], g["value"], g["line"]
    slots_at, place_keys, place_ord = g["slots_at"], g["place_keys"], g["place_ord"]
    by_key = g["by_key"]
    n = g["n"]
    n_p = n_q = n_h3 = nq = n_g_ok = n_g_ref = n_d_ok = n_d_ripe = n_four = 0
    sizes = []
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
        pin2, _, R = hop(value[pin1], place[s], by_key, slots_at, value, rng)
        if pin2 is None:
            continue
        pin3, _, T = hop(value[pin2], R, by_key, slots_at, value, rng)
        if pin3 is None:
            continue
        n_h3 += 1
        cands = next_cands(value[pin3], T, by_key)
        sizes.append(len(cands))
        n_four += int(len(cands) == 4)
        visited = {place[s], R, T}
        used_k = {value[pin1], value[pin2], value[pin3]}
        order = list(place_ord[place[s]])
        gnew, _ = greedy(cands, set(), order, by_key, visited, used_k,
                         slots_at, value, place_keys, rng)
        if gnew is None or len(gnew) != 1:
            n_g_ref += 1
        else:
            pin = think_place(list(slots_at[next(iter(gnew))]), value, rng)
            n_g_ok += int(pin is not None and value[pin] == "CRISP")
        dnew, _ = deep(cands, set(), order, by_key, visited, used_k,
                       slots_at, value, place_keys, rng)
        if dnew is not None and len(dnew) == 1:
            pin = think_place(list(slots_at[next(iter(dnew))]), value, rng)
            if pin is not None:
                n_d_ok += int(value[pin] == "CRISP")
                n_d_ripe += int(value[pin] == "RIPE")
    return dict(
        n_p=n_p, n_q=n_q, hop3=n_h3 / max(n_p, 1),
        four=n_four / max(n_h3, 1),
        mean_cands=(sum(sizes) / len(sizes)) if sizes else 0,
        greedy_crisp=n_g_ok / max(n_p, 1),
        greedy_refuse=n_g_ref / max(n_h3, 1),
        deep_crisp=n_d_ok / max(n_p, 1),
        deep_ripe=n_d_ripe / max(n_p, 1),
        q=nq / max(n_q, 1),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rep = run(designed(), random.Random(args.seed))
    if rep is None:
        print("no tape")
        return 1
    void = (rep["four"] < 1.0) or (rep["greedy_crisp"] == 1.0)
    gate = ((not void) and (rep["greedy_refuse"] == 1.0)
            and (rep["deep_crisp"] == 1.0) and (rep["deep_ripe"] == 0.0)
            and (rep["q"] == 1.0))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate), **rep)
    print(f"cands {rep['mean_cands']:.1f} four {rep['four']:.0f}")
    print(f"greedy CRISP {rep['greedy_crisp']:.0f} refuse {rep['greedy_refuse']:.0f}")
    print(f"deep   CRISP {rep['deep_crisp']:.0f} RIPE {rep['deep_ripe']:.0f} Q {rep['q']:.0f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not 4 cands, or greedy already unique.")
    elif gate:
        print("\nGO ADV: greedy 4->3 refuse; UNLOCK->TAG pins CRISP.")
    else:
        print("\nSTOP: greedy lucked CRISP, or deep failed.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
