"""591: LIVE/DEAD on v1 walk, per window. No new hop. No Q.

After miss  mark pick DEAD
After hit   mark pick LIVE
Next hole   PMI-first unique extra that is not DEAD

A  frozen PMI (590)
B  PMI + skip DEAD
C  random unique extra

GATE  B-A > 0.05  AND  n_changed >= 40
VOID  n_changed < 40   (DEAD never bites)

    python _check591_mark.py
    python _audit591_mark.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import collect, prefix_windows

OUT = Path("results/_stage591_mark.json")

DEAD, LIVE = "DEAD", "LIVE"


def pick_pmi(toks):
    return toks[0] if toks else None


def pick_skip(toks, marks):
    if not toks:
        return None
    for tok in toks:
        if marks.get(tok) != DEAD:
            return tok
    return toks[0]


def pick_rnd(toks, rng):
    return toks[rng.randrange(len(toks))] if toks else None


def walk_eps(eps, how, rng):
    marks = {}
    n = hit = changed = 0
    n2 = hit2 = 0
    for ep in eps:
        h1 = ep["h1"]
        toks, held = h1["toks"], h1["held"]
        n += 1
        if how == "pmi":
            pick = pick_pmi(toks)
        elif how == "skip":
            pick = pick_skip(toks, marks)
            if toks and pick != toks[0]:
                changed += 1
        else:
            pick = pick_rnd(toks, rng)
        ok = int(pick == held)
        hit += ok
        if pick is not None:
            marks[pick] = LIVE if ok else DEAD
        if not ok:
            continue
        h2 = ep["h2"]
        if h2 is None or not h2["toks"]:
            continue
        n2 += 1
        if how == "pmi":
            p2 = pick_pmi(h2["toks"])
        elif how == "skip":
            p2 = pick_skip(h2["toks"], marks)
        else:
            p2 = pick_rnd(h2["toks"], rng)
        ok2 = int(p2 == h2["held"])
        hit2 += ok2
        if p2 is not None:
            marks[p2] = LIVE if ok2 else DEAD
    return dict(
        n=n,
        fill=hit / n if n else 0.0,
        n_changed=changed,
        n2=n2,
        fill2=hit2 / n2 if n2 else 0.0,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=80_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-win", type=int, default=80)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=120000)
    ap.add_argument("--cap-probe", type=int, default=4)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    windows = prefix_windows(pool, args.window_lines, args.n_win)
    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"591 mark  {path}  {kind}  windows={len(windows)}", flush=True)

    acc = {k: dict(n=0, hit=0, changed=0, n2=0, hit2=0) for k in ("pmi", "skip", "rnd")}
    n_eps = 0
    for lines in windows:
        eps = collect(lines, args, rng)
        n_eps += len(eps)
        for how in acc:
            rec = walk_eps(eps, how, random.Random(args.seed + 17))
            acc[how]["n"] += rec["n"]
            acc[how]["hit"] += int(round(rec["fill"] * rec["n"]))
            acc[how]["changed"] += rec["n_changed"]
            acc[how]["n2"] += rec["n2"]
            acc[how]["hit2"] += int(round(rec["fill2"] * rec["n2"]))

    def fill(how):
        a = acc[how]
        return a["hit"] / a["n"] if a["n"] else 0.0

    fa, fb, fc = fill("pmi"), fill("skip"), fill("rnd")
    n_changed = acc["skip"]["changed"]
    void = acc["pmi"]["n"] < 40 or n_changed < 40
    d_pmi = fb - fa
    d_rnd = fb - fc
    gate = (not void) and d_pmi > 0.05 and d_rnd > 0.05
    print(
        f"n {acc['pmi']['n']}  PMI {fa:.3f}  skip {fb:.3f}  rnd {fc:.3f}  "
        f"changed {n_changed}"
    )
    print(f"skip-PMI {d_pmi:+.3f}  skip-rnd {d_rnd:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: DEAD never changes the PMI pick.")
    elif gate:
        print("GO MARK: skip-DEAD beats frozen PMI on this walk.")
    else:
        print("STOP: marks bite or not, but skip does not beat PMI.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), n_eps=n_eps,
        elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate),
        fill_pmi=fa, fill_skip=fb, fill_rnd=fc,
        d_pmi=d_pmi, d_rnd=d_rnd, n_changed=n_changed,
        n=acc["pmi"]["n"],
        n2_pmi=acc["pmi"]["n2"], n2_skip=acc["skip"]["n2"],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev[f"{args.seed}_{path.stem}"] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
