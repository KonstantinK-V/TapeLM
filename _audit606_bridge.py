"""606: BRIDGE ceiling. Choose PLACE; tape finishes with frozen unique+PMI.

P is an exact counted frame address (w,left,right), not a token and not an
extra. The chooser sees only address counts. After P is picked, tape reads its
rows and applies frozen unique+PMI; extras stay hidden until that read.

The query is one concrete tape row. Its row filler remains visible and one
different literal is hidden, so two fillers of one address do not create the
same visible query with different labels.

O       exists address P mentioning visible pin: extract(P)==held  REACH
F       first address then extract                                  fixed
C       max count-only address rule then extract                    count
A       random address then extract
MAJ-P   majority filler of the max-majority address                 majority
BAG-MAJ pooled value mode, report only (not a place action)

GATE  O - max(F,C,A,MAJ-P) > 0.05
VOID  n < 40 OR exact visible-query collision > 0.02
PMI is not a chooser feature.

    python _check606_bridge.py
    python _audit606_bridge.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit606_bridge.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit606_bridge.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit511_ring import pick_corpus
from _audit589_hop3 import add_pairs, prefix_windows
from _audit593_mix import pmi_rank

OUT = Path("results/_stage606_bridge.json")


def build_places(lines, frame_max, min_fillers):
    """Keep exact address ids; _audit511_ring.graph intentionally drops them."""
    keep, toks, _owner = tframes.frame_keep(lines, frame_max, min_fillers)
    if not keep:
        return None
    places = []
    by_place = defaultdict(set)
    by_row = defaultdict(list)
    df = Counter()
    co = Counter()
    n_fr = 0
    for pi, (addr, pos) in enumerate(keep):
        _w, left, right = addr
        keys = tuple(x for x in left + right if x)
        vals = [toks[i] for i in pos]
        cnt = Counter(vals)
        maj_tok, maj_n = cnt.most_common(1)[0]
        maj = maj_n / max(len(vals), 1)
        place = dict(
            addr=addr,
            keys=keys,
            vals=vals,
            count_key=(maj, -len(vals), -len(set(keys))),
            majority=maj_tok,
        )
        places.append(place)
        for val in vals:
            row = set(keys)
            row.add(val)
            add_pairs(co, row, 1)
            n_fr += 1
            for tok in row:
                df[tok] += 1
                by_place[tok].add(pi)
                by_row[tok].append(n_fr - 1)
    return dict(
        places=places,
        by_place={k: sorted(v) for k, v in by_place.items()},
        by_row=by_row,
        df=df,
        co=co,
        n_fr=n_fr,
    )


def bands(pg):
    vals = sorted(
        pg["df"][tok] for tok, rows in pg["by_row"].items()
        if len(rows) >= 8
    )
    if len(vals) < 8:
        return set(), set()
    p25 = vals[int(0.25 * (len(vals) - 1))]
    p75 = vals[int(0.75 * (len(vals) - 1))]
    mid = {tok for tok, rows in pg["by_row"].items()
           if len(rows) >= 8 and p25 <= pg["df"][tok] <= p75}
    high = {tok for tok, rows in pg["by_row"].items()
            if len(rows) >= 8 and pg["df"][tok] > p75}
    return mid, high


def place_offer(place, pin, env_m, mid_set):
    bag = []
    uniq = []
    seen = set()
    for val in place["vals"]:
        fr = set(place["keys"])
        fr.add(val)
        fr.discard(pin)
        extra = [tok for tok in fr if tok not in env_m and tok in mid_set]
        bag.extend(tok for tok in fr if tok in mid_set)
        if len(extra) == 1 and extra[0] not in seen:
            seen.add(extra[0])
            uniq.append(extra[0])
    return bag, uniq


def extract_at(place, pin, env_m, mid_set, co, df, n_use):
    bag, uniq = place_offer(place, pin, env_m, mid_set)
    if not uniq:
        return None, bag, uniq
    ranked = pmi_rank(uniq, env_m, co, df, n_use)
    tok = ranked[0] if ranked else uniq[0]
    return tok, bag, uniq


def adjust_frame_stats(co, df, toks, delta):
    """Remove/restore one hidden query row from joint and marginal counts."""
    row = set(toks)
    add_pairs(co, row, delta)
    for tok in row:
        df[tok] += delta


def chooser_features(place, pin, env_m, df, n_fr, frame_max):
    """Pre-READ tape counts and query equalities only."""
    vals = place["vals"]
    counts = sorted(Counter(vals).values(), reverse=True)
    n_rows = max(len(vals), 1)
    n_distinct = len(counts)
    m1 = counts[0] / n_rows if counts else 0.0
    m2 = counts[1] / n_rows if len(counts) > 1 else 0.0
    keys = set(place["keys"])
    ov = len(keys & env_m)
    union = len(keys | env_m)
    pin_rows = sum(1 for val in vals if val == pin)
    key_df = sum(df.get(tok, 0) for tok in keys) / max(len(keys), 1)
    width = place["addr"][0]
    return [
        math.log1p(n_rows) / 4.0,
        math.log1p(n_distinct) / 3.0,
        m1,
        m2,
        math.log1p(len(keys)) / 3.0,
        ov / max(len(env_m), 1),
        ov / max(union, 1),
        float(pin in keys),
        pin_rows / n_rows,
        math.log1p(df.get(pin, 0)) / max(math.log1p(n_fr), 1.0),
        math.log1p(key_df) / max(math.log1p(n_fr), 1.0),
        width / max(frame_max, 1),
    ]


def collect(lines, args, rng):
    pg = build_places(lines, args.frame_max, args.min_fillers)
    if pg is None:
        return []
    places = pg["places"]
    mid_set, high_set = bands(pg)
    if not mid_set:
        return []
    co, df, n_fr = pg["co"], pg["df"], pg["n_fr"]
    out = []
    pins = sorted(mid_set)
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
            visible_vals = [
                tok for tok in dict.fromkeys(query["vals"])
                if tok != pin
            ]
            helds = [
                tok for tok in dict.fromkeys(query["keys"])
                if tok in mid_set and tok != pin
            ]
            if not visible_vals or not helds:
                continue
            qval = rng.choice(visible_vals)
            held = rng.choice(helds)
            row_tokens = set(query["keys"])
            row_tokens.add(qval)
            env = row_tokens - {pin, held}
            env_m = (env & mid_set) - high_set or (env - high_set)
            if not env_m or qval not in env:
                continue
            cids = [pi for pi in pg["by_place"][pin] if pi != qi]
            if len(cids) < 2:
                continue
            adjust_frame_stats(co, df, row_tokens, -1)
            n_use = max(n_fr - 1, 1)
            try:
                rows = []
                bag0 = []
                for pi in cids:
                    place = places[pi]
                    tok, _bag, uniq = extract_at(
                        place, pin, env_m, mid_set, co, df, n_use,
                    )
                    bag0.extend(_bag)
                    rows.append(dict(
                        addr=place["addr"],
                        count_key=place["count_key"],
                        majority=place["majority"],
                        feat=chooser_features(
                            place, pin, env_m, df, n_use, args.frame_max,
                        ),
                        extract=tok,
                        has_u=bool(uniq),
                    ))
            finally:
                adjust_frame_stats(co, df, row_tokens, +1)
            if not bag0 or held not in set(bag0):
                continue
            out.append(dict(
                held=held,
                bag0=bag0,
                places=rows,
                query_sig=(pin, tuple(sorted(env))),
            ))
    return out


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
    rnd = random.Random(args.seed + 43)
    t0 = time.time()
    print(f"606 bridge  {path}  {kind}  windows={len(windows)}", flush=True)

    n = o = f = c = a = m = bm = 0
    sum_places = 0
    conflicts = 0
    for lines in windows:
        tape_rows = collect(lines, args, rng)
        tape_sig = defaultdict(Counter)
        for row in tape_rows:
            tape_sig[row["query_sig"]][row["held"]] += 1
        conflicts += sum(
            sum(cnt.values()) - max(cnt.values()) for cnt in tape_sig.values()
        )
        for row in tape_rows:
            n += 1
            held, places, bag0 = row["held"], row["places"], row["bag0"]
            sum_places += len(places)
            o += int(any(pl["extract"] == held for pl in places))
            f += int(places[0]["extract"] == held)
            pc = max(places, key=lambda pl: pl["count_key"])
            c += int(pc["extract"] == held)
            a += int(places[rnd.randrange(len(places))]["extract"] == held)
            m += int(pc["majority"] == held)
            bag_maj = Counter(bag0).most_common(1)[0][0]
            bm += int(bag_maj == held)

    def r(x):
        return x / n if n else 0.0

    fo, ff, fc, fa, fm, fbm = r(o), r(f), r(c), r(a), r(m), r(bm)
    strongest = max(ff, fc, fa, fm)
    collision = conflicts / max(n, 1)
    void = n < 40 or collision > 0.02
    gate = (not void) and (fo - strongest > 0.05)
    print(
        f"n {n}  places {sum_places / max(n, 1):.1f}  "
        f"collision {collision:.3f}  REACH {fo:.3f}  first {ff:.3f}  "
        f"count {fc:.3f}  rnd {fa:.3f}  MAJ-P {fm:.3f}  "
        f"BAG-MAJ {fbm:.3f}  strongest {strongest:.3f}"
    )
    print(f"REACH-strongest {fo - strongest:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: thin or the exact visible query has conflicting labels.")
    elif gate:
        print("BRIDGE OPEN: a place exists whose frozen extract hits held; hand routes lag.")
    else:
        print("STOP: REACH does not beat the strongest frozen route. No 607.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate), n=n,
        mean_places=sum_places / max(n, 1),
        collision=collision,
        reach=fo, fill_first=ff, fill_count=fc, fill_rnd=fa,
        fill_maj_place=fm, fill_bag_maj=fbm,
        strongest=strongest, d=fo - strongest, actual_place_ids=True,
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
