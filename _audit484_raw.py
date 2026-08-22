"""484: frozen 436/477 machine on a raw tape. No new lever, no Q."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import _tape_frames as tframes
import _audit436_constpin as M436
from _audit440_compose import think_place

OUT = Path("results/_stage484_raw.json")
WIKI = Path("data/_wikitext103_train.txt")
FALLBACK = Path("data/external_tinystories_mini.txt")


def pick_corpus(explicit):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"no corpus {p}")
        kind = "wiki" if "wiki" in p.name.lower() else "given"
        min_line = 80 if kind == "wiki" else 20
        return p, kind, min_line
    if WIKI.exists():
        return WIKI, "wiki", 80
    if FALLBACK.exists():
        return FALLBACK, "tinystories-fallback", 20
    raise SystemExit("no wiki, no tinystories")


def compose_diag(lines, args, rng):
    keep, toks, owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    if not keep:
        return dict(unique_next=0.0, n_const_c=0, n_uniq=0, hop3_if_live=0.0)
    place, value, line, keys = [], [], [], []
    for (w, left, right), ps in keep:
        name = f"{' '.join(left)}|{' '.join(right)}"
        ks = {x for x in list(left) + list(right) if x}
        for i in ps:
            place.append(name)
            value.append(toks[i])
            line.append(owner[i])
            keys.append(ks)
    slots_at = defaultdict(list)
    for s, P in enumerate(place):
        slots_at[P].append(s)
    by_key = defaultdict(set)
    for s, ks in enumerate(keys):
        for k in ks:
            by_key[k].add(place[s])
    n_c = n_u = n_h3 = 0
    seen = list(slots_at)
    rng.shuffle(seen)
    for P in seen[: args.addresses]:
        sl = slots_at[P]
        vs = [value[i] for i in sl]
        if len(set(vs)) != 1 or len(sl) < 2:
            continue
        n_c += 1
        v = vs[0]
        cands = by_key.get(v, set()) - {P}
        if len(cands) != 1:
            continue
        n_u += 1
        R = next(iter(cands))
        tape = {P: "LIVE"}
        if tape.get(P) != "LIVE":
            continue
        pin = think_place(list(slots_at[R]), value, rng)
        n_h3 += int(pin is not None)
    return dict(unique_next=n_u / max(n_c, 1),
                n_const_c=n_c, n_uniq=n_u, hop3_if_live=n_h3 / max(n_u, 1))


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
    print(f"corpus {kind}  {path}  window {len(lines)}  min_line {min_line}")
    if not lines:
        print("no lines")
        return 1
    pin = M436.measure(lines, args, random.Random(args.seed))
    if pin is None:
        print("no tape")
        return 1
    comp = compose_diag(lines, args, random.Random(args.seed + 1))
    void = pin["const_live"] <= 0.05
    gate = ((not void) and (pin["const_hit"] > 0.90)
            and (pin["n_mixed"] == 0 or pin["refuse_mixed"] > 0.90)
            and (pin["hop2_sees_pin"] == 1.0))
    rec = dict(seed=args.seed, corpus=kind, n_lines=len(lines),
               void=bool(void), gate=bool(gate), pin=pin, compose=comp)
    print(f"const_live {pin['const_live']:.3f}  mixed {pin['mixed_of_df2']:.3f}  "
          f"hit {pin['const_hit']:.2f}  refuse_m {pin['refuse_mixed']:.2f}  "
          f"hop2 {pin['hop2_sees_pin']:.0f}")
    print(f"DIAG unique_next {comp['unique_next']:.3f}  "
          f"n_const {comp['n_const_c']} uniq {comp['n_uniq']}  "
          f"hop3|LIVE {comp['hop3_if_live']:.2f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: almost no constant e(P) on this raw window.")
    elif gate:
        print("\nGO RAW: 436 think holds on raw tape. unique_next is a counter, not the gate.")
    else:
        print("\nSTOP: frozen think broke on raw tape.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
