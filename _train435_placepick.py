"""435: CE pick among e(P). 434 offer; 432-style feats."""
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

OUT = Path("results/_stage435_placepick.json")
WIKI = Path("data/_wikitext103_train.txt")
CAP = 8
FEAT = 4


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def feats(s, t, place, line, line_toks, places_on_line, value):
    p = place[s]
    nq = places_on_line[line[s]] - {p}
    nt = places_on_line[line[t]] - {p}
    fq = set(line_toks[line[s]]) - {value[s]}
    ft = set(line_toks[line[t]]) - {value[t]}
    return [
        jaccard(nq, nt),
        jaccard(fq, ft),
        math.log1p(len(ft)) / 5.0,
        float(line[t] != line[s]),
    ]


class PickNet(nn.Module if nn is not None else object):
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
        return self.net(x).reshape(-1)


def build(lines, args, rng):
    keep, toks, owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    if args.addresses and len(keep) > args.addresses:
        keep = rng.sample(keep, args.addresses)
    if not keep:
        return None
    place, value, line = [], [], []
    for (w, left, right), ps in keep:
        name = f"{' '.join(left)}|{' '.join(right)}"
        for i in ps:
            place.append(name)
            value.append(toks[i])
            line.append(owner[i])
    n = len(place)
    slots_at = defaultdict(list)
    on_line = defaultdict(list)
    for s in range(n):
        slots_at[place[s]].append(s)
        on_line[line[s]].append(s)
    places_on_line = {li: {place[s] for s in ss} for li, ss in on_line.items()}
    line_toks = defaultdict(list)
    for i, tok in enumerate(toks):
        line_toks[owner[i]].append(tok)
    df = Counter(value)
    rows = []
    idx = list(range(n))
    rng.shuffle(idx)
    for s in idx[: args.max_questions]:
        v, p, li = value[s], place[s], line[s]
        foreign = [t for t in slots_at[p] if t != s and line[t] != li]
        rng.shuffle(foreign)
        offer = foreign[:CAP]
        teach = next((t for t in offer if value[t] == v), None)
        maj_v = (Counter(value[t] for t in offer).most_common(1)[0][0]
                 if offer else None)
        rows.append(dict(
            s=s, v=v, df=df[v], offer=offer, teach=teach,
            mixed=bool(offer) and maj_v != v,
            refuse=teach is None,
        ))
    return dict(place=place, value=value, line=line, line_toks=line_toks,
                places_on_line=places_on_line, rows=rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--addresses", type=int, default=1500)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--max-questions", type=int, default=8000)
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
    df2 = [r for r in rows if r["df"] >= 2]
    live = sum(not r["refuse"] for r in df2) / max(len(df2), 1)
    mixed = [r for r in df2 if r["mixed"] and r["teach"] is not None]
    print(f"df≥2 {len(df2)}  live {live:.3f}  mixed {len(mixed)}")
    void = (live <= 0.05) or (len(mixed) < 30)
    rep = dict(seed=args.seed, n=len(rows), n_df2=len(df2), live=live,
               n_mixed=len(mixed), void=bool(void), k=CAP, working_cells=1)
    if void:
        print("VOID: 434 offer not live here. Do not train.")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
        prev[str(args.seed)] = rep
        Path(args.out).write_text(json.dumps(prev, indent=1), encoding="utf-8")
        return 0
    if torch is None:
        print("need torch")
        return 1
    rng.shuffle(rows)
    n_ex = max(8, int(len(rows) * args.exam))
    exam, train_rows = rows[:n_ex], rows[n_ex:]
    train_m = [r for r in train_rows if r["mixed"] and r["teach"] is not None]
    exam_m = [r for r in exam if r["mixed"] and r["offer"]]
    if len(train_m) < 20 or len(exam_m) < 10:
        print("VOID: mixed split too thin after exam cut.")
        rep["void"] = True
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
        prev[str(args.seed)] = rep
        Path(args.out).write_text(json.dumps(prev, indent=1), encoding="utf-8")
        return 0
    place, value, line = tape["place"], tape["value"], tape["line"]
    line_toks, places_on_line = tape["line_toks"], tape["places_on_line"]
    torch.manual_seed(args.seed)
    net = PickNet()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for _ in range(args.steps):
        r = train_m[rng.randrange(len(train_m))]
        xs = [feats(r["s"], t, place, line, line_toks, places_on_line, value)
              for t in r["offer"]]
        y = r["offer"].index(r["teach"])
        logits = net.scores(xs)
        loss = -torch.log_softmax(logits, dim=0)[y]
        opt.zero_grad()
        loss.backward()
        opt.step()

    def mind(r):
        xs = [feats(r["s"], t, place, line, line_toks, places_on_line, value)
              for t in r["offer"]]
        with torch.no_grad():
            i = int(net.scores(xs).argmax())
        return r["offer"][i]

    def rnd(r):
        return rng.choice(r["offer"])

    def hit_of(picker, pop):
        n_ok = n_pin = hop2 = 0
        for r in pop:
            t = picker(r)
            working = {("work", 0): t}
            hop2 += int(working[("work", 0)] == t)
            n_pin += 1
            n_ok += int(value[t] == r["v"])
        return n_ok / max(n_pin, 1), hop2 / max(n_pin, 1)

    mh, hop = hit_of(mind, exam_m)
    rh, _ = hit_of(rnd, exam_m)
    d = mh - rh
    gate = d > 0.05 and hop == 1.0
    rep.update(dict(void=False, gate=bool(gate), mind_hit=mh, rnd_hit=rh,
                    d_hit=d, hop2_sees_pin=hop, exam_mixed=len(exam_m)))
    print(f"MLP mixed  mind {mh:.3f}  rnd {rh:.3f}  Δ {d:+.3f}  hop2 {hop:.0f}")
    print(f"GATE {gate}")
    if gate:
        print("\nGO PICK: mind pins a mixed e(P) cell that holds v, better than random.")
    else:
        print("\nSTOP PICK: 434 offer live; 432-style feats do not pick on w400.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
