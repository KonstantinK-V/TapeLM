"""433: walk + pin where the tape already holds the answer."""
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

OUT = Path("results/_stage433_pinwalk.json")
WIKI = Path("data/_wikitext103_train.txt")
CAP = 8
FEAT = 4


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def feats(s, t, keys, place, line, line_toks):
    kt = set(line_toks[line[t]])
    return [
        jaccard(keys[s], kt),
        float(place[t] == place[s]),
        math.log1p(len(kt)) / 5.0,
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
    line_toks = defaultdict(list)
    for i, tok in enumerate(toks):
        line_toks[owner[i]].append(tok)
    df = Counter(value)
    rows = []
    idx = list(range(n))
    rng.shuffle(idx)
    for s in idx[: args.max_questions]:
        v, li = value[s], line[s]
        bag = []
        for t in range(n):
            if line[t] == li:
                continue
            if keys[s] & set(line_toks[line[t]]):
                bag.append(t)
        rng.shuffle(bag)
        offer = bag[:CAP]
        teach = next((t for t in offer if value[t] == v), None)
        rows.append(dict(s=s, v=v, df=df[v], offer=offer, teach=teach,
                         refuse=teach is None))
    return dict(place=place, value=value, line=line, keys=keys,
                line_toks=line_toks, df=df, rows=rows, n=n)


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

    def hit_of(pick_fn, pop):
        n = n_ok = n_ref = n_pin = hop2_ok = 0
        for r in pop:
            n += 1
            t = pick_fn(r)
            working = {}
            if t is None:
                n_ref += 1
                continue
            working[("work", 0)] = t
            hop2 = working[("work", 0)]
            n_pin += 1
            hop2_ok += int(hop2 == t)
            n_ok += int(tape["value"][t] == r["v"])
        return dict(
            n=n, hit=n_ok / max(n_pin, 1), refuse=n_ref / max(n, 1),
            hop2_sees_pin=hop2_ok / max(n_pin, 1), n_pin=n_pin,
        )

    def ora(r):
        return r["teach"]

    def rnd(r):
        return rng.choice(r["offer"]) if r["offer"] else None

    def maj(r):
        if not r["offer"]:
            return None
        c = Counter(tape["value"][t] for t in r["offer"])
        want = c.most_common(1)[0][0]
        for t in r["offer"]:
            if tape["value"][t] == want:
                return t
        return r["offer"][0]

    ora_s = hit_of(ora, df2)
    rnd_s = hit_of(rnd, df2)
    maj_s = hit_of(maj, df2)
    d_ceil = ora_s["hit"] - rnd_s["hit"]
    void = live <= 0.05
    print(f"df≥2 {len(df2)}/{len(rows)}  live {live:.3f}")
    print(f"CEILING  ora {ora_s['hit']:.3f}  rnd {rnd_s['hit']:.3f}  "
          f"maj {maj_s['hit']:.3f}  ora−rnd {d_ceil:+.3f}")
    print(f"VOID {void}")
    rep = dict(
        seed=args.seed, n=len(rows), n_df2=len(df2), live=live,
        ceil_ora=ora_s["hit"], ceil_rnd=rnd_s["hit"], ceil_maj=maj_s["hit"],
        ceil_d=d_ceil, void=bool(void), k=CAP, working_cells=1,
        feat="jaccard_keys,same_place,log_line,other_line",
    )
    if void:
        print("\nVOID: window does not retrieve a cell that holds v. Nothing to pin.")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
        prev[str(args.seed)] = rep
        Path(args.out).write_text(json.dumps(prev, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
        return 0
    if d_ceil <= 0.05:
        rep["gate"] = False
        print("\nSTOP CEILING: even the oracle pin is not above random. Do not train.")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
        prev[str(args.seed)] = rep
        Path(args.out).write_text(json.dumps(prev, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
        return 0

    if torch is None:
        print("need torch to train; ceiling GO already printed")
        return 1

    rng.shuffle(rows)
    n_ex = max(8, int(len(rows) * args.exam))
    exam, train_rows = rows[:n_ex], rows[n_ex:]
    exam_df2 = [r for r in exam if r["df"] >= 2]
    exam_df1 = [r for r in exam if r["df"] == 1]
    torch.manual_seed(args.seed)
    net = PickNet()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    place, value, line, keys, line_toks = (
        tape["place"], tape["value"], tape["line"], tape["keys"], tape["line_toks"])
    refuse_x = [0.0] * FEAT
    for _ in range(args.steps):
        r = train_rows[rng.randrange(len(train_rows))]
        xs = [feats(r["s"], t, keys, place, line, line_toks) for t in r["offer"]]
        xs = xs + [refuse_x]
        y = r["offer"].index(r["teach"]) if r["teach"] is not None else len(r["offer"])
        logits = net.scores(xs)
        logp = torch.log_softmax(logits, dim=0)
        loss = -logp[y]
        opt.zero_grad()
        loss.backward()
        opt.step()

    def mind(r):
        xs = [feats(r["s"], t, keys, place, line, line_toks) for t in r["offer"]]
        xs = xs + [refuse_x]
        with torch.no_grad():
            sc = net.scores(xs)
        i = int(sc.argmax())
        if i >= len(r["offer"]):
            return None
        return r["offer"][i]

    m2 = hit_of(mind, exam_df2)
    r2 = hit_of(rnd, exam_df2)

    def refuse_rate(pop, picker):
        if not pop:
            return 0.0
        return sum(picker(r) is None for r in pop) / len(pop)

    ref1 = refuse_rate(exam_df1, mind)
    ref2 = m2["refuse"]
    d = m2["hit"] - r2["hit"]
    gate = (d > 0.05) and (ref1 > ref2) and (m2["hop2_sees_pin"] == 1.0)
    rep.update(dict(
        ceiling_go=True, gate=bool(gate),
        mind_hit=m2["hit"], rnd_hit=r2["hit"], d_hit=d,
        refuse_df1=ref1, refuse_df2=ref2,
        hop2_sees_pin=m2["hop2_sees_pin"],
        exam_df2=len(exam_df2), exam_df1=len(exam_df1),
    ))
    print(f"MLP  df≥2 hit {m2['hit']:.3f}  rnd {r2['hit']:.3f}  Δ {d:+.3f}")
    print(f"refuse  df1 {ref1:.3f}  df2 {ref2:.3f}  hop2_sees_pin {m2['hop2_sees_pin']:.0f}")
    print(f"GATE {gate}")
    if gate:
        print("\nGO: pin on a cell that holds v; hop2 stands there.")
    else:
        print("\nSTOP: ceiling was live; pick/refuse/pin did not hold.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
