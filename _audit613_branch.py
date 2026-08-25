"""613: hop1 is a door. Look for held in that word's bags. No Q.

612: pin only when extracts agree (~9%). Most trials TIE — nowhere to pin.
Idea: do not pin hop1. Each hop1 extract is a branch root.
Walk a few mid bags of that word (pin out of the frame). Hit if held is there.
If not, next hop1.

SEARCH  union of first K extracts of the pin (608)
BRANCH  sequential doors = those extracts except held itself;
        bag of a door-place contains held (mid extras, pin forbidden)
RAND    same budget, random mid words not pin/held

GATE  extra = P(BRANCH ∧ ¬SEARCH) > 0.05
      AND  extra_rand = P(RAND ∧ ¬SEARCH) ; extra - extra_rand > 0.05
VOID  n < 40 or n_miss < 40
DIAG  joint - SEARCH   (not gated)

Adequacy: mid band only, high-df doors skipped, cap K doors × L places.
606 bag0-filter dropped — otherwise SEARCH misses do not exist by construction.
608/612 not retrained. 557 closed.

    python _check613_branch.py
    python _audit613_branch.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit613_branch.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit613_branch.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import adjust_frame_stats, prefix_windows
from _audit606_bridge import bands, build_places, extract_at

OUT = Path("results/_stage613_branch.json")
K = 3
L = 3


def doors_of(rows):
    out = []
    seen = set()
    for row in rows[:K]:
        tok = row["extract"]
        if tok is None or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def branch_hit(pg, door, pin, held, qi, mid_set, rng):
    if door not in mid_set or door == pin or door == held:
        return False
    cids = [pi for pi in pg["by_place"].get(door, ()) if pi != qi]
    rng.shuffle(cids)
    used = 0
    for pi in cids:
        if used >= L:
            break
        place = pg["places"][pi]
        fr = set(place["keys"])
        if pin in fr or pin in place["vals"]:
            continue
        used += 1
        extra = [tok for tok in place["vals"] if tok in mid_set and tok != door]
        extra += [tok for tok in place["keys"] if tok in mid_set and tok != door]
        if held in extra:
            return True
    return False


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
    print(f"613 branch  {path}  {kind}  windows={len(windows)}  K={K} L={L}", flush=True)

    n = n_miss = 0
    s_hit = b_hit = r_hit = 0
    extra = extra_r = 0
    for lines in windows:
        pg = build_places(lines, args.frame_max, args.min_fillers)
        if pg is None:
            continue
        mid_set, high_set = bands(pg)
        if not mid_set:
            continue
        co, df, n_fr = pg["co"], pg["df"], pg["n_fr"]
        places = pg["places"]
        pins = list(mid_set)
        rng.shuffle(pins)
        for pin in pins:
            qids = list(pg["by_place"].get(pin, ()))
            if len(qids) < 3:
                continue
            rng.shuffle(qids)
            for qi in qids[: args.cap_probe]:
                query = places[qi]
                if pin not in set(query["keys"]):
                    continue
                env = set(query["keys"])
                env_m = (env & mid_set) - high_set or (env - high_set)
                if not env_m:
                    continue
                helds = [
                    tok for tok in dict.fromkeys(query["vals"])
                    if tok in mid_set and tok != pin and tok not in env
                ]
                if not helds:
                    continue
                held = rng.choice(helds)
                cids = [pi for pi in pg["by_place"][pin] if pi != qi]
                if len(cids) < 2:
                    continue
                qtoks = set(query["keys"])
                qtoks.add(held)
                adjust_frame_stats(co, df, qtoks, -1)
                n_use = max(n_fr - 1, 1)
                try:
                    rows = []
                    for pi in cids:
                        tok, _bag, uniq = extract_at(
                            places[pi], pin, env_m, mid_set, co, df, n_use,
                        )
                        rows.append(dict(extract=tok, has_u=bool(uniq)))
                finally:
                    adjust_frame_stats(co, df, qtoks, +1)
                n += 1
                search = any(row["extract"] == held for row in rows[:K])
                s_hit += int(search)
                if not search:
                    n_miss += 1
                doors = [d for d in doors_of(rows) if d != held and d not in high_set]
                br = any(
                    branch_hit(pg, d, pin, held, qi, mid_set, rng) for d in doors
                )
                b_hit += int(br)
                extra += int(br and not search)
                pool_r = [t for t in mid_set if t not in (pin, held) and t not in high_set]
                rnd_doors = rng.sample(pool_r, k=min(K, len(pool_r))) if pool_r else []
                rr = any(
                    branch_hit(pg, d, pin, held, qi, mid_set, rng) for d in rnd_doors
                )
                r_hit += int(rr)
                extra_r += int(rr and not search)

    def r(x, d):
        return x / d if d else 0.0

    void = n < 40 or n_miss < 40
    p_s, p_b, p_r = r(s_hit, n), r(b_hit, n), r(r_hit, n)
    p_e, p_er = r(extra, n), r(extra_r, n)
    d_rand = p_e - p_er
    gate = (not void) and p_e > 0.05 and d_rand > 0.05
    print(f"n {n}  miss {n_miss}  SEARCH {p_s:.3f}  BRANCH {p_b:.3f}  RAND {p_r:.3f}")
    print(f"extra {p_e:.3f}  extra_rand {p_er:.3f}  d {d_rand:+.3f}  joint {p_s + p_e:.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: thin SEARCH-miss. Hungry, not STOP of SEARCH.")
    elif gate:
        print("GO BRANCH: hop1 doors find held after SEARCH miss, above random mid.")
    else:
        print("STOP: depth bags do not add apples beyond SEARCH / random mid.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate), n=n, n_miss=n_miss,
        k=K, l=L, search=p_s, branch=p_b, rand=p_r,
        extra=p_e, extra_rand=p_er, d_rand=d_rand, joint=p_s + p_e,
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
