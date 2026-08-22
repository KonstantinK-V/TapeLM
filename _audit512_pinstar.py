"""512: 436 pin → walk the star of that v. Separate from 513."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit511_ring import mentions, walk

OUT = Path("results/_stage512_pinstar.json")
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
    n = len(value)
    df = Counter()
    slots_at = defaultdict(list)
    for s, P in enumerate(place):
        slots_at[P].append(s)
        df[value[s]] += 1
        for k in keys[s]:
            df[k] += 1
    return dict(n=n, place=place, value=value, line=line, keys=keys,
                slots_at=slots_at, df=df)


def const_pins(g, rng, max_q):
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
        out.append(v)
    return out


def mean_walk(g, by, vs, cache):
    rows = [walk(g, by, v, cache) for v in vs]
    n = max(len(rows), 1)
    return dict(
        n=len(rows),
        d1=sum(r["d1"] for r in rows) / n,
        d2=sum(r["d2"] for r in rows) / n,
        m1=sum(r["m1"] for r in rows) / n,
        m2=sum(r["m2"] for r in rows) / n,
    )


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
    by = mentions(g)
    pins = const_pins(g, random.Random(args.seed), args.max_questions)
    vocab = [t for t, sl in by.items() if len(sl) >= 8]
    rng2 = random.Random(args.seed + 3)
    rnd = [rng2.choice(vocab) for _ in pins] if vocab else []
    cache = {}
    pin_rep = mean_walk(g, by, pins, cache)
    rnd_rep = mean_walk(g, by, rnd, cache)
    void = pin_rep["n"] < 30
    gate = (not void) and (pin_rep["d2"] > rnd_rep["d2"]) and (
        pin_rep["m2"] - rnd_rep["m2"] > 0.05)
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines),
               pin=pin_rep, rnd=rnd_rep, void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  window {len(lines)}  pins {pin_rep['n']}")
    print(f"PIN  d1 {pin_rep['d1']:.2f} d2 {pin_rep['d2']:.2f}  "
          f"m1 {pin_rep['m1']:.3f} m2 {pin_rep['m2']:.3f}")
    print(f"RND  d1 {rnd_rep['d1']:.2f} d2 {rnd_rep['d2']:.2f}  "
          f"m1 {rnd_rep['m1']:.3f} m2 {rnd_rep['m2']:.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not enough 436 pins.")
    elif gate:
        print("\nGO PIN-STAR: a const pin's v walks a richer star than a random token.")
    else:
        print("\nSTOP: pin does not land on a better star than chance.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
