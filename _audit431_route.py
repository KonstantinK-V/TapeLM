"""431: structural ROUTE to foreign evidence. Torch-free. Not a policy.

Graph vertices = places. Edges = geometry only (no values):
  LINE   two places sit on the same line
  NEIGH  adjacent positions on the SAME line
  FRAME  |frame ∩ frame| ≥ 2
Start  = {place_A, place_B}
Target = places of 425 eA(true)∪eB(true) MINUS start
GATE   mean(dist_struct − dist_random) < −0.5   not GO policy
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path

import _tape_frames as tframes
import _audit425_hang as H425

OUT = Path("results/_stage431_route.json")
WIKI = Path("data/_wikitext103_train.txt")
K = 8
DELTA = 0.5
N_RAND = 5


def frame_set(name):
    if "|" not in name:
        return set(name.split())
    left, right = name.split("|", 1)
    return {w for w in left.split() + right.split() if w}


def bfs(starts, goals, adj, inf):
    starts, goals = set(starts), set(goals)
    if not starts or not goals:
        return inf
    if starts & goals:
        return 0
    seen = set(starts)
    q = deque((s, 0) for s in starts)
    while q:
        u, d = q.popleft()
        for v in adj.get(u, ()):
            if v in seen:
                continue
            if v in goals:
                return d + 1
            seen.add(v)
            q.append((v, d + 1))
    return inf


def measure(lines, args, rng):
    keep, toks, owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    if args.addresses and len(keep) > args.addresses:
        keep = rng.sample(keep, args.addresses)
    if not keep:
        return None
    place, value, line, pos = [], [], [], []
    for (w, left, right), ps in keep:
        name = f"{' '.join(left)}|{' '.join(right)}"
        for i in ps:
            place.append(name)
            value.append(toks[i])
            line.append(owner[i])
            pos.append(i)
    n = len(place)
    places = sorted(set(place))
    inf = len(places) + 1
    slots_at = defaultdict(list)
    on_line = defaultdict(list)
    lines_of_place = defaultdict(set)
    by_val = defaultdict(list)
    for s in range(n):
        slots_at[place[s]].append(s)
        on_line[line[s]].append(s)
        lines_of_place[place[s]].add(line[s])
        by_val[value[s]].append(s)
    adj = defaultdict(set)
    n_line = n_neigh = n_frame = 0

    def link(u, v):
        if u == v or v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        return True

    for ss in on_line.values():
        ps = list({place[s] for s in ss})
        for i, u in enumerate(ps):
            for v in ps[i + 1:]:
                if link(u, v):
                    n_line += 1
        order = sorted(ss, key=lambda s: pos[s])
        for a, b in zip(order, order[1:]):
            if pos[b] - pos[a] == 1:
                if link(place[a], place[b]):
                    n_neigh += 1
    frames = {p: frame_set(p) for p in places}
    for i, u in enumerate(places):
        fu = frames[u]
        if len(fu) < 2:
            continue
        for v in places[i + 1:]:
            if len(fu & frames[v]) >= 2:
                if link(u, v):
                    n_frame += 1
    qs = []
    for li, ss in on_line.items():
        if len(ss) < 2:
            continue
        cand = [(a, b) for ai, a in enumerate(ss) for b in ss[ai + 1:]
                if place[a] != place[b] and abs(pos[a] - pos[b]) > args.frame_max]
        if not cand:
            continue
        rng.shuffle(cand)
        qs.extend(cand[:args.pairs_per_line])
    rng.shuffle(qs)
    qs = qs[:args.max_questions]
    k = args.k
    line_map = defaultdict(lambda: defaultdict(list))
    for s in range(n):
        line_map[line[s]][place[s]].append(value[s])
    ds, dr, dm, tsize = [], [], [], []
    n_off = n_co = n_foreign = n_reach = 0
    for a, b in qs:
        pa, pb, va, vb = place[a], place[b], value[a], value[b]
        ca = Counter(value[t] for t in slots_at[pa] if t != a)
        cb = Counter(value[t] for t in slots_at[pb] if t != b)
        if not ((va in ca) and (vb in cb)):
            continue
        n_off += 1
        marg = (ca.most_common(1)[0][0], cb.most_common(1)[0][0])
        right = marg == (va, vb)
        seen = False
        for lj in (lines_of_place[pa] & lines_of_place[pb]):
            if lj == line[a]:
                continue
            m = line_map[lj]
            if va in m.get(pa, ()) and vb in m.get(pb, ()):
                seen = True
                break
        comp = (not right) and (not seen)
        if not comp:
            continue
        n_co += 1
        eA = H425.evidence(va, by_val, line, line[a], {a, b}, k)
        eB = H425.evidence(vb, by_val, line, line[a], {a, b}, k)
        T = {place[s] for s in eA + eB} - {pa, pb}
        if not T:
            continue
        n_foreign += 1
        start = {pa, pb}
        d_s = bfs(start, T, adj, inf)
        n_reach += int(d_s < inf)
        tsize.append(len(T))
        others = [p for p in places if p not in start]
        d_r_acc = 0.0
        for _ in range(N_RAND):
            if len(others) >= len(T):
                rnd = set(rng.sample(others, len(T)))
            else:
                rnd = set(others)
            d_r_acc += bfs(start, rnd, adj, inf)
        d_r = d_r_acc / N_RAND
        eM0 = H425.evidence(marg[0], by_val, line, line[a], {a, b}, k)
        eM1 = H425.evidence(marg[1], by_val, line, line[a], {a, b}, k)
        Tm = {place[s] for s in eM0 + eM1} - {pa, pb}
        d_m = bfs(start, Tm, adj, inf) if Tm else inf
        ds.append(d_s)
        dr.append(d_r)
        dm.append(d_m)

    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    return {
        "slots": n, "n_places": len(places),
        "n_edges": sum(len(v) for v in adj.values()) // 2,
        "edges_line": n_line, "edges_neigh": n_neigh, "edges_frame": n_frame,
        "n_offered": n_off, "n_comp_only": n_co, "n_foreign": n_foreign,
        "reach_any_target": n_reach / max(n_foreign, 1),
        "target_size": mean(tsize),
        "dist_struct": mean(ds), "dist_random": mean(dr), "dist_maj": mean(dm),
        "delta_vs_random": mean(ds) - mean(dr) if ds else 0.0,
        "delta_vs_maj": mean(ds) - mean(dm) if ds else 0.0,
        "inf": inf, "delta_bar": DELTA, "k": k, "working_cells": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--addresses", type=int, default=1500)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--pairs-per-line", type=int, default=2)
    ap.add_argument("--max-questions", type=int, default=20000)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default=str(WIKI))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    if args.window_lines and args.window_lines < len(lines):
        s0 = rng.randrange(len(lines) - args.window_lines)
        lines = lines[s0:s0 + args.window_lines]
    rep = measure(lines, args, rng)
    if rep is None:
        print("no tape")
        return 1
    rep["seed"] = args.seed
    void = (rep["n_foreign"] < 30) or (rep["reach_any_target"] <= 0.05)
    gate = (not void) and (rep["delta_vs_random"] < -DELTA)
    rep["void"], rep["gate"] = bool(void), bool(gate)
    print(f"places {rep['n_places']}  edges {rep['n_edges']}  "
          f"line {rep['edges_line']} neigh {rep['edges_neigh']} frame {rep['edges_frame']}")
    print(f"foreign {rep['n_foreign']}/{rep['n_comp_only']}  "
          f"reach_any_target {rep['reach_any_target']:.3f}  target_size {rep['target_size']:.2f}")
    print(f"dist  struct {rep['dist_struct']:.3f}  random {rep['dist_random']:.3f}  "
          f"maj {rep['dist_maj']:.3f}  Δrand {rep['delta_vs_random']:+.3f}")
    print(f"VOID {rep['void']}  GATE {rep['gate']}")
    if void:
        print("\nVOID: no foreign evidence to walk to, or graph never reaches it. "
              "Policy has nothing to eat. 424 is not guilty.")
    elif gate:
        print("\nROUTE EXISTS: structural hops beat random. Not GO policy.")
    else:
        print("\nNO ROUTE: graph reaches somewhere, not teacher evidence. "
              "№2 closed. Do not train a stepper.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
