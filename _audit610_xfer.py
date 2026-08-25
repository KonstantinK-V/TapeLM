"""610: 609 k-curve on a foreign tape. No Q. Same frozen unique+PMI + mention order.

Home 609 was stories. This is the same SEARCH law on --corpus (news).
k is READ budget, not 519's k·n/df. Nothing is fitted on home.

VOID  n < 40
CLOSE REACH - F@8 <= 0.05
STILL F@8 - F@3 > 0.05
TAIL  neither
GATE  CLOSE or STILL
609/608 not retrained. 557 stays closed.

    python _check610_xfer.py
    python _audit610_xfer.py --seed 1337 --corpus data/_stage254_news.txt
    python _audit610_xfer.py --seed 8642 --corpus data/_stage254_news.txt
    python _audit610_xfer.py --seed 2890 --corpus data/_stage254_news.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import prefix_windows
from _audit606_bridge import collect
from _audit609_kcurve import KS, hit_prefix

OUT = Path("results/_stage610_xfer.json")


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
    ap.add_argument("--corpus", default="data/_stage254_news.txt")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    windows = prefix_windows(pool, args.window_lines, args.n_win)
    rng = random.Random(args.seed)
    rnd = random.Random(args.seed + 79)
    t0 = time.time()
    print(f"610 xfer  {path}  {kind}  windows={len(windows)}", flush=True)

    n = o = m = nlen = 0
    acc = {name: [0] * (len(KS) + 1) for name in ("F", "C", "A")}
    ge = {k: 0 for k in KS}
    for lines in windows:
        for row in collect(lines, args, rng):
            n += 1
            held, places, bag0 = row["held"], row["places"], row["bag0"]
            nlen += len(places)
            o += int(any(pl["extract"] == held for pl in places))
            maj = Counter(bag0).most_common(1)[0][0]
            m += int(maj == held)
            mention = list(places)
            count = sorted(places, key=lambda pl: pl["count_key"], reverse=True)
            shuf = list(places)
            rnd.shuffle(shuf)
            for k in KS:
                ge[k] += int(len(places) >= k)
            for name, order in (("F", mention), ("C", count), ("A", shuf)):
                for i, k in enumerate(KS):
                    acc[name][i] += hit_prefix(order, held, k)
                acc[name][-1] += hit_prefix(order, held, len(order))

    def r(x):
        return x / n if n else 0.0

    def col(name):
        return [r(v) for v in acc[name]]

    f, c, a = col("F"), col("C"), col("A")
    fo, fm = r(o), r(m)
    mean_n = nlen / n if n else 0.0
    i3, i6, i8 = KS.index(3), KS.index(6), KS.index(8)
    d_still = f[i8] - f[i3]
    gap8 = fo - f[i8]
    d_maj6 = f[i6] - fm
    void = n < 40
    close = (not void) and gap8 <= 0.05
    still = (not void) and d_still > 0.05
    tail = (not void) and (not close) and (not still)
    gate = close or still
    print(
        f"n {n}  mean|P| {mean_n:.1f}  REACH {fo:.3f}  bagMAJ {fm:.3f}"
    )
    print("k        " + "  ".join(f"{k:>5}" for k in KS) + "    all")
    print("F        " + "  ".join(f"{v:5.3f}" for v in f))
    print("C        " + "  ".join(f"{v:5.3f}" for v in c))
    print("A        " + "  ".join(f"{v:5.3f}" for v in a))
    print("share>=k " + "  ".join(f"{r(ge[k]):5.3f}" for k in KS))
    print(f"F@8-F@3 {d_still:+.3f}  REACH-F@8 {gap8:+.3f}  F@6-MAJ {d_maj6:+.3f}")
    print(f"VOID {void}  CLOSE {close}  STILL {still}  TAIL {tail}  GATE {gate}")
    if void:
        print("VOID: thin. Hungry exam, not STOP of 609.")
    elif close or still:
        print("GO XFER: 609 SEARCH law holds on this tape.")
    else:
        print("STOP TAIL: extra READs after 3 do not pay here. Stories-local tickets.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), close=bool(close), still=bool(still),
        tail=bool(tail), gate=bool(gate), n=n, mean_n=mean_n,
        reach=fo, fill_bag_maj=fm, KS=list(KS),
        F=f, C=c, A=a, share_ge={str(k): r(ge[k]) for k in KS},
        d_still=d_still, gap8=gap8, d_maj6=d_maj6,
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
