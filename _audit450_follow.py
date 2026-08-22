"""450: choose SEEK by newly opened addresses. No hop2 value peek. No Φ."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _audit440_compose import think_place, think_slot
from _audit447_seek import hop, next_cands
from _audit448_pick import all_seeks
from _audit449_adv import build, designed, filter_cands, greedy

OUT = Path("results/_stage450_follow.json")


def opened(vH, order, by_key, visited, cands, used_k):
    before = {H for _, H in all_seeks(order, by_key, visited, cands, used_k)}
    after = all_seeks(order + [vH], by_key, visited, cands, used_k)
    return [(k, H) for k, H in after if H not in before]


def n_follow(vH, order, by_key, visited, cands, used_k):
    return len(opened(vH, order, by_key, visited, cands, used_k))


def pick_follow(cands, extra, order, by_key, visited, used_k, slots_at, value, rng):
    opts = all_seeks(order, by_key, visited, cands, used_k)
    scored = []
    for k, H in opts:
        pinH = think_place(list(slots_at[H]), value, rng)
        if pinH is None:
            continue
        vis1 = set(visited) | {H}
        used1 = set(used_k) | {k}
        nf = n_follow(value[pinH], order, by_key, vis1, cands, used1)
        scored.append((nf, k, H, pinH, vis1, used1))
    if not scored:
        return None, opts, None
    m = max(x[0] for x in scored)
    if m < 1:
        return None, opts, None
    top = [x for x in scored if x[0] == m]
    if len({x[2] for x in top}) != 1:
        return None, opts, None
    return top[0], opts, m


def run(lines, rng):
    g = build(lines)
    if g is None:
        return None
    place, value, line = g["place"], g["value"], g["line"]
    slots_at, place_keys, place_ord = g["slots_at"], g["place_keys"], g["place_ord"]
    by_key = g["by_key"]
    n = g["n"]
    n_p = n_q = n_h3 = nq = n_four = 0
    n_g_ok = n_g_ref = n_f_ok = n_f_ripe = n_unl = 0
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
        n_four += int(len(cands) == 4)
        visited = {place[s], R, T}
        used_k = {value[pin1], value[pin2], value[pin3]}
        order = list(place_ord[place[s]])
        extra = set()
        gnew, _ = greedy(cands, extra, order, by_key, visited, used_k,
                         slots_at, value, place_keys, rng)
        if gnew is None or len(gnew) != 1:
            n_g_ref += 1
        else:
            pin = think_place(list(slots_at[next(iter(gnew))]), value, rng)
            n_g_ok += int(pin is not None and value[pin] == "CRISP")
        chosen, _, m = pick_follow(
            cands, extra, order, by_key, visited, used_k, slots_at, value, rng)
        if chosen is None:
            continue
        nf, k, H, pinH, vis1, used1 = chosen
        n_unl += int(value[pinH] == "UNLOCK")
        extra1 = extra | {value[pinH]}
        c1 = filter_cands(cands, extra1, place_keys)
        opts2 = opened(value[pinH], order, by_key, vis1, c1, used1)
        if len(opts2) != 1:
            continue
        k2, H2 = opts2[0]
        pin2b = think_place(list(slots_at[H2]), value, rng)
        if pin2b is None:
            continue
        c2 = filter_cands(c1, extra1 | {value[pin2b]}, place_keys)
        if len(c2) != 1:
            continue
        pin4 = think_place(list(slots_at[next(iter(c2))]), value, rng)
        if pin4 is None:
            continue
        n_f_ok += int(value[pin4] == "CRISP")
        n_f_ripe += int(value[pin4] == "RIPE")
    return dict(
        n_p=n_p, n_q=n_q, hop3=n_h3 / max(n_p, 1),
        four=n_four / max(n_h3, 1),
        greedy_crisp=n_g_ok / max(n_p, 1),
        greedy_refuse=n_g_ref / max(n_h3, 1),
        follow_crisp=n_f_ok / max(n_p, 1),
        follow_ripe=n_f_ripe / max(n_p, 1),
        picked_unlock=n_unl / max(n_p, 1),
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
    void = (rep["four"] < 1.0) or (rep["picked_unlock"] < 1.0)
    gate = ((not void)
            and (rep["greedy_crisp"] == 0.0) and (rep["greedy_refuse"] == 1.0)
            and (rep["follow_crisp"] == 1.0) and (rep["follow_ripe"] == 0.0)
            and (rep["q"] == 1.0))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate), **rep)
    print(f"four {rep['four']:.0f}  greedy CRISP {rep['greedy_crisp']:.0f} refuse {rep['greedy_refuse']:.0f}")
    print(f"follow CRISP {rep['follow_crisp']:.0f} RIPE {rep['follow_ripe']:.0f} "
          f"picked UNLOCK {rep['picked_unlock']:.0f} Q {rep['q']:.0f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: n_follow did not unique-pick UNLOCK.")
    elif gate:
        print("\nGO FOLLOW: hop1 by next-address count, not peek of TAG. Greedy still refuses.")
    else:
        print("\nSTOP: follow missed CRISP, or greedy already had it.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
