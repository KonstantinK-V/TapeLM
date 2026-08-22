"""504: Famb is the world. Unique not required.

  436-const pin, |next|>=2
  extra = other tokens on the question line
  resolve if an extra sits in exactly one candidate place's keys  (444)
  control: extras from a foreign line

  DIAG hop_any: think_place on a random next vs random place

VOID  n_famb < 30
GATE  resolve > 0.05  and  resolve > shuffle + 0.05
No Q. Coverage-over-episodes is 505 if this arena is live.

    python _check504_famb.py
    python _audit504_famb.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit440_compose import think_place

OUT = Path("results/_stage504_famb.json")
WIKI = Path("data/_wikitext103_train.txt")
FALLBACK = Path("data/external_tinystories_mini.txt")
CAP = 8


def pick_corpus(explicit):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"no corpus {p}")
        kind = "wiki" if "wiki" in p.name.lower() else "given"
        return p, kind, 80 if kind == "wiki" else 20
    if WIKI.exists():
        return WIKI, "wiki", 80
    if FALLBACK.exists():
        return FALLBACK, "tinystories-fallback", 20
    raise SystemExit("no corpus")


def graph(lines, frame_max, min_fillers):
    keep, toks, owner = tframes.frame_keep(lines, frame_max, min_fillers)
    if not keep:
        return None
    place, value, line, keys = [], [], [], []
    for (w, left, right), ps in keep:
        name = f"{' '.join(left)}|{' '.join(right)}"
        ks = tuple(x for x in list(left) + list(right) if x)
        for i in ps:
            place.append(name)
            value.append(toks[i])
            line.append(owner[i])
            keys.append(ks)
    slots_at = defaultdict(list)
    for s, P in enumerate(place):
        slots_at[P].append(s)
    by_key = defaultdict(set)
    pkeys = {}
    for s, P in enumerate(place):
        pkeys.setdefault(P, set()).update(keys[s])
        for k in keys[s]:
            by_key[k].add(P)
    line_toks = defaultdict(list)
    for li, row in enumerate(lines):
        line_toks[li] = row.split()
    return dict(n=len(place), place=place, value=value, line=line, keys=keys,
                slots_at=slots_at, by_key=by_key, pkeys=pkeys, line_toks=line_toks)


def nexts(g, P, v):
    return g["by_key"].get(v, set()) - {P}


def famb_pop(g, rng, max_q):
    idx = list(range(g["n"]))
    rng.shuffle(idx)
    out = []
    for s in idx[:max_q]:
        v, P, li = g["value"][s], g["place"][s], g["line"][s]
        foreign = [t for t in g["slots_at"][P] if t != s and g["line"][t] != li]
        if not foreign:
            continue
        rng.shuffle(foreign)
        offer = foreign[:CAP]
        maj = Counter(g["value"][t] for t in offer).most_common(1)[0][0]
        if maj != v:
            continue
        c = nexts(g, P, v)
        if len(c) < 2:
            continue
        extras = [t for t in g["line_toks"][li] if t != v]
        out.append((P, v, li, c, extras))
    return out


def unique_hit(g, cands, extra):
    hit = [R for R in cands if extra in g["pkeys"][R]]
    return hit[0] if len(hit) == 1 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--max-questions", type=int, default=8000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    L = args.window_lines
    if L < len(pool):
        s0 = rng.randrange(len(pool) - L + 1)
        lines = pool[s0:s0 + L]
    else:
        lines = pool
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        print("no tape")
        return 1
    pop = famb_pop(g, random.Random(args.seed), args.max_questions)
    n = len(pop)
    n_res = n_sh = n_any = n_rnd = 0
    places = list(g["slots_at"])
    rng2 = random.Random(args.seed + 9)
    for P, v, li, cands, extras in pop:
        n_res += int(any(unique_hit(g, cands, e) is not None for e in extras))
        other = g["line_toks"][rng2.choice(list(g["line_toks"]))]
        n_sh += int(any(unique_hit(g, cands, e) is not None for e in other if e != v))
        R = rng2.choice(sorted(cands))
        n_any += int(think_place(list(g["slots_at"][R]), g["value"], rng2) is not None)
        Q = rng2.choice(places)
        n_rnd += int(think_place(list(g["slots_at"][Q]), g["value"], rng2) is not None)
    resolve = n_res / max(n, 1)
    shuffle = n_sh / max(n, 1)
    hop_any = n_any / max(n, 1)
    hop_rnd = n_rnd / max(n, 1)
    void = n < 30
    gate = (not void) and (resolve > 0.05) and (resolve > shuffle + 0.05)
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines), n_famb=n,
               resolve=resolve, shuffle=shuffle,
               hop_any=hop_any, hop_rnd=hop_rnd,
               void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  window {len(lines)}  famb {n}")
    print(f"resolve {resolve:.3f}  foreign-line {shuffle:.3f}  "
          f"hop_any {hop_any:.3f}  hop_rnd {hop_rnd:.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: almost no Famb after 436 pin.")
    elif gate:
        print("\nGO FAMB: a line extra uniquely cuts the branch. Unique hop not required.")
    else:
        print("\nSTOP: Famb exists, extra-from-line does not resolve better than a foreign line.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
