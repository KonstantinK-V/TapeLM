"""502: frozen 477 on the 501 PIN436 unique-next arena. No Q."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit440_compose import think_place

OUT = Path("results/_stage502_compose.json")
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


def pop_unique(g, rng, max_q):
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
        cands = g["by_key"].get(v, set()) - {P}
        if len(cands) != 1:
            continue
        out.append((P, v, next(iter(cands))))
    return out


def think_ok(g, R, rng):
    pin = think_place(list(g["slots_at"][R]), g["value"], rng)
    return pin is not None


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

    pop = pop_unique(g, random.Random(args.seed), args.max_questions)
    places = list(g["slots_at"])
    n_pop = len(pop)
    hop_ok = n_pin = n_ref = n_pr = 0
    rng2 = random.Random(args.seed + 7)
    for P, v, R in pop:
        hop_ok += int(R in g["slots_at"] and R != P)
        if think_ok(g, R, rng2):
            n_pin += 1
        else:
            n_ref += 1
        Q = rng2.choice(places)
        n_pr += int(think_ok(g, Q, rng2))

    pin_rate = n_pin / max(n_pop, 1)
    pin_rand = n_pr / max(n_pop, 1)
    hop_rate = hop_ok / max(n_pop, 1)
    void = n_pop < 30
    gate = ((not void) and (hop_rate == 1.0)
            and (pin_rate > 0.05)
            and (pin_rate > pin_rand + 0.05))
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines),
               n_pop=n_pop, hop_ok=hop_rate,
               n_pin=n_pin, n_refuse=n_ref,
               pin_rate=pin_rate, pin_rand=pin_rand,
               void=bool(void), gate=bool(gate))
    print(f"corpus {kind}  window {len(lines)}  pop {n_pop}")
    print(f"hop_ok {hop_rate:.2f}  pin {pin_rate:.3f}  refuse {n_ref/max(n_pop,1):.3f}  "
          f"pin_rand {pin_rand:.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: unique-next 436 population < 30.")
    elif gate:
        print("\nGO COMPOSE: unique hop lands; second pin beats random landing.")
    else:
        print("\nSTOP: unique address exists, but hop2 think is empty or = random place.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
