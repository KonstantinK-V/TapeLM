"""501: three agreement thresholds, same raw window. Unique-next is a counter."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes

OUT = Path("results/_stage501_agree.json")
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
    return dict(n=n, place=place, value=value, line=line, keys=keys,
                slots_at=slots_at, by_key=by_key)


def uniq_of(g, P, v):
    return len(g["by_key"].get(v, set()) - {P}) == 1


def arm_soft(g, thresh):
    n_p = n_u = 0
    for P, sl in g["slots_at"].items():
        vs = [g["value"][i] for i in sl]
        if len(sl) < 2:
            continue
        maj, c = Counter(vs).most_common(1)[0]
        if c / len(vs) < thresh:
            continue
        n_p += 1
        n_u += int(uniq_of(g, P, maj))
    return dict(n=n_p, uniq=n_u, rate=n_u / max(n_p, 1))


def arm_436(g, rng, max_q):
    n = g["n"]
    idx = list(range(n))
    rng.shuffle(idx)
    idx = idx[:max_q]
    n_c = n_u = 0
    for s in idx:
        v, P, li = g["value"][s], g["place"][s], g["line"][s]
        foreign = [t for t in g["slots_at"][P] if t != s and g["line"][t] != li]
        if len(foreign) < 1:
            continue
        rng.shuffle(foreign)
        offer = foreign[:CAP]
        maj = Counter(g["value"][t] for t in offer).most_common(1)[0][0]
        if maj != v:
            continue
        n_c += 1
        n_u += int(uniq_of(g, P, v))
    return dict(n=n_c, uniq=n_u, rate=n_u / max(n_c, 1))


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
    lines = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    if args.window_lines and args.window_lines < len(lines):
        s0 = rng.randrange(max(len(lines) - args.window_lines, 1))
        lines = lines[s0:s0 + args.window_lines]
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        print("no tape")
        return 1

    soft = arm_soft(g, 0.6)
    strict = arm_soft(g, 0.999)
    pin = arm_436(g, random.Random(args.seed), args.max_questions)
    void = pin["n"] < 30
    gate = (not void) and (pin["rate"] > 0.05)
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines), n_slots=g["n"],
               void=bool(void), gate=bool(gate),
               soft=soft, pin436=pin, strict=strict)
    print(f"corpus {kind}  window {len(lines)}  slots {g['n']}")
    print(f"{'arm':8}  n_agree   uniq   unique_next")
    print(f"{'SOFT':8}  {soft['n']:7}  {soft['uniq']:5}  {soft['rate']:.3f}")
    print(f"{'PIN436':8}  {pin['n']:7}  {pin['uniq']:5}  {pin['rate']:.3f}")
    print(f"{'STRICT':8}  {strict['n']:7}  {strict['uniq']:5}  {strict['rate']:.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: not enough 436-const pins.")
    elif gate:
        print("\nGO ARENA: unique-next among 436 pins is not empty. Compose has a population.")
    else:
        print("\nSTOP ARENA: 436 pins exist, almost none have a unique next place.")
    print("DIAG: SOFT/STRICT are counters. Night lived in SOFT; 484 compose was STRICT.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
