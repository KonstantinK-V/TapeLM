"""429: Phi on the 428 graph. Ceiling first, net only if algebra is blind.

Features (no letters, no pair_seen, no question line):
  hang_428 (2<=df<=5), jaccard, cosine, log|eA|, log|eB|, log n_cross
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

import _tape_frames as tframes
import _audit425_hang as H425
import _audit428_hang as H428

OUT = Path("results/_stage429_hang.json")
WIKI = Path("data/_wikitext103_train.txt")
FEAT = 6
K = 8


def union_ctx(slots, value, line, line_toks, other):
    b = set()
    for s in slots:
        b.update(H425.ctx_of(s, value, line, line_toks, other))
    return b


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def cosine(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def pair_feats(va, vb, a, b, by_val, line, value, line_toks, df, k):
    eA = H425.evidence(va, by_val, line, line[a], {a, b}, k)
    eB = H425.evidence(vb, by_val, line, line[a], {a, b}, k)
    h = H428.hang_of(va, vb, a, b, by_val, line, value, line_toks, df, 0, k)
    ua = union_ctx(eA, value, line, line_toks, vb)
    ub = union_ctx(eB, value, line, line_toks, va)
    n_cross = 0
    for i in eA:
        for j in eB:
            if line[i] != line[j]:
                n_cross += 1
    return [
        h,
        jaccard(ua, ub),
        cosine(ua, ub),
        math.log1p(len(eA)) / 3.0,
        math.log1p(len(eB)) / 3.0,
        math.log1p(n_cross) / 4.0,
    ]


class HangNet(nn.Module if nn is not None else object):
    def __init__(self):
        if nn is None:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEAT, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def scores(self, xs):
        x = torch.tensor(xs, dtype=torch.float32)
        return self.net(x).squeeze(-1)


def build(lines, args, rng):
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
    slots_at = defaultdict(list)
    on_line = defaultdict(list)
    lines_of_place = defaultdict(set)
    by_val = defaultdict(list)
    for s in range(n):
        slots_at[place[s]].append(s)
        on_line[line[s]].append(s)
        lines_of_place[place[s]].add(line[s])
        by_val[value[s]].append(s)
    line_map = defaultdict(lambda: defaultdict(list))
    for s in range(n):
        line_map[line[s]][place[s]].append(value[s])
    line_toks = defaultdict(list)
    for i, tok in enumerate(toks):
        line_toks[owner[i]].append(tok)
    df = Counter()
    for ws in line_toks.values():
        for w in set(ws):
            df[w] += 1
    tape = dict(place=place, value=value, line=line, slots_at=slots_at,
                on_line=on_line, lines_of_place=lines_of_place, by_val=by_val,
                line_map=line_map, line_toks=line_toks, df=df)
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
    rows = []
    k = args.k
    for a, b in qs:
        pa, pb, va, vb = place[a], place[b], value[a], value[b]
        ca = Counter(value[t] for t in slots_at[pa] if t != a)
        cb = Counter(value[t] for t in slots_at[pb] if t != b)
        if not ((va in ca) and (vb in cb)):
            continue
        marg = (ca.most_common(1)[0][0], cb.most_common(1)[0][0])
        right = marg == (va, vb)
        seen = False
        common = lines_of_place[pa] & lines_of_place[pb]
        for lj in common:
            if lj == line[a]:
                continue
            m = line_map[lj]
            if va in m.get(pa, ()) and vb in m.get(pb, ()):
                seen = True
                break
        comp = (not right) and (not seen)
        keys_a, keys_b = list(ca), list(cb)
        ft = pair_feats(va, vb, a, b, by_val, line, value, line_toks, df, k)
        fm = pair_feats(marg[0], marg[1], a, b, by_val, line, value, line_toks, df, k)
        decoys = []
        for _ in range(16):
            if len(decoys) >= 2:
                break
            ra, rb = rng.choice(keys_a), rng.choice(keys_b)
            if (ra, rb) in ((va, vb), marg):
                continue
            decoys.append((ra, rb))
        while len(decoys) < 2:
            decoys.append(marg)
        fd = [pair_feats(ra, rb, a, b, by_val, line, value, line_toks, df, k)
              for ra, rb in decoys]
        rows.append(dict(true=(va, vb), marg=marg, ft=ft, fm=fm, fd=fd,
                         comp=comp, keys_a=keys_a, keys_b=keys_b))
    tape["rows"] = rows
    return tape


def d_of(rows, ix):
    xs = [r["ft"][ix] - r["fm"][ix] for r in rows]
    return sum(xs) / max(len(xs), 1)


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
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--exam", type=float, default=0.35)
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
    tape = build(lines, args, rng)
    if tape is None or not tape["rows"]:
        print("no tape")
        return 1
    rows = tape["rows"]
    co = [r for r in rows if r["comp"]]
    names = ("hang428", "jaccard", "cosine")
    ceil = {nm: d_of(co, i) for i, nm in enumerate(names)} if co else {nm: 0.0 for nm in names}
    best = max(ceil.values()) if ceil else 0.0
    print(f"offered {len(rows)}  comp_only {len(co)}")
    print("CEILING  " + "  ".join(f"{nm} {ceil[nm]:+.4f}" for nm in names)
          + f"  best {best:+.4f}")
    rep = {
        "seed": args.seed, "n_offered": len(rows), "n_comp_only": len(co),
        "ceil": ceil, "best_algebra": best, "k": args.k, "working_cells": 0,
        "feat": "hang428,jaccard,cosine,log_eA,log_eB,log_cross",
    }
    if best > 0.05:
        rep["void"] = False
        rep["go_algebra"] = True
        rep["gate"] = False
        print("\nGO ALGEBRA: a mix of these already ranks. No net (38.3).")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
        prev[str(args.seed)] = rep
        Path(args.out).write_text(json.dumps(prev, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
        return 0
    if len(co) < 20:
        rep["void"] = True
        rep["go_algebra"] = False
        rep["gate"] = False
        print("\nVOID: too few comp_only to train or exam. Not a gap for Phi.")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
        prev[str(args.seed)] = rep
        Path(args.out).write_text(json.dumps(prev, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
        return 0

    if torch is None:
        print("need torch to train; ceiling already printed")
        return 1
    rng.shuffle(rows)
    n_ex = max(8, int(len(rows) * args.exam))
    exam, train_rows = rows[:n_ex], rows[n_ex:]
    exam_co = [r for r in exam if r["comp"]]
    torch.manual_seed(args.seed)
    net = HangNet()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for _ in range(args.steps):
        r = train_rows[rng.randrange(len(train_rows))]
        xs = [r["ft"], r["fm"]] + r["fd"]
        order = list(range(4))
        rng.shuffle(order)
        xs = [xs[i] for i in order]
        y = order.index(0)
        logits = net.scores(xs)
        logp = torch.log_softmax(logits, dim=0)
        loss = -logp[y]
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    n_gt = dsum = 0
    with torch.no_grad():
        for r in exam_co:
            st = float(net.scores([r["ft"]]).reshape(-1)[0])
            sm = float(net.scores([r["fm"]]).reshape(-1)[0])
            n_gt += int(st > sm)
            dsum += st - sm
    acc = n_gt / max(len(exam_co), 1)
    dmean = dsum / max(len(exam_co), 1)
    gate = acc > 0.55
    rep.update(dict(void=False, go_algebra=False, gate=bool(gate),
                    exam_comp=len(exam_co), mind_gt_maj=acc, d_score=dmean))
    print(f"MLP exam_comp {len(exam_co)}  P(true>maj) {acc:.3f}  d_score {dmean:+.4f}")
    print(f"VOID False  GATE {gate}")
    if gate:
        print("\nGO PHI: mind ranks the true pair over majority on this graph.")
    else:
        print("\nSTOP PHI: algebra was pair-sensitive and blind; the net is too. "
              "This graph+feature family is closed.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
