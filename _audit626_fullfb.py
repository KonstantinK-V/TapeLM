"""626: full-feedback rank on HOPONLY, diagnostic after 625.

Train sees every candidate's hop1 label (not chosen-only).
Test: pick place with best Q. Rival COUNT (count_key) and RAND.
Label = HOPONLY only. DIRECT is printed, not trained.

If pick − count > 0.05 → bandit was the hole (624 chosen-reward).
If STOP → features empty; new tape signal, not a bigger net.

Same hide as 624/625. Split windows 70/30.

    python _check626_fullfb.py
    python _audit626_fullfb.py --seed 1337 --corpus data/_tinystories_train.txt
    python _audit626_fullfb.py --seed 8642 --corpus data/_tinystories_train.txt
    python _audit626_fullfb.py --seed 2890 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from _audit511_ring import pick_corpus
from _audit589_hop3 import add_pairs, prefix_windows
from _audit606_bridge import bands, build_places, extract_at
from _audit615_depth import extracts, walk
from _audit622_trust import recip
from _audit624_pick import hide_two

OUT = Path("results/_stage626_fullfb.json")
K = 3
CAP = 8


def feat(place, tok, vot, rp):
    maj, nvals, nkeys = place["count_key"]
    v = vot[tok] if tok is not None else 0
    return (min(int(maj * 4), 4), min(v, 4), min(rp, 2))


def label_places(pg, pin, qi, use, held_ask, env_m, mid_set, high_set, co, df, n_use):
    vot = defaultdict(int)
    rows = []
    for pi in use:
        place = pg["places"][pi]
        tok, _bag, uniq = extract_at(place, pin, env_m, mid_set, co, df, n_use)
        if tok is not None and uniq:
            vot[tok] += 1
        rows.append((pi, place, tok, bool(uniq)))
    out = []
    for pi, place, tok, ok in rows:
        d = h = 0
        rp = 0
        if ok and tok == held_ask:
            d = 1
        elif ok and tok is not None and tok in mid_set and tok not in high_set:
            rp = recip(pg, tok, qi, env_m, mid_set, co, df, n_use)
            _f, cc = walk(
                pg, tok, held_ask, qi, env_m, mid_set, high_set,
                co, df, n_use, {pin, tok},
            )
            h = int(cc[1])
        out.append(dict(
            pi=pi, place=place, tok=tok, d=d, h=h,
            key=feat(place, tok, vot, rp),
            count_key=place["count_key"],
        ))
    return out


def trials(windows, rng, args):
    recs = []
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
                extras = [
                    tok for tok in dict.fromkeys(query["vals"])
                    if tok in mid_set and tok != pin and tok not in env
                    and tok not in high_set
                ]
                if len(extras) < 2:
                    continue
                rng.shuffle(extras)
                held_ctx, held_ask = extras[0], extras[1]
                rec = None
                hide_two(co, df, query["keys"], held_ctx, held_ask, -1)
                n_use = max(n_fr - 2, 1)
                try:
                    rows_p = extracts(
                        pg, pin, qi, env_m, mid_set, co, df, n_use,
                    )
                    if any(tok == held_ask for tok, _b, _u in rows_p[:K]):
                        continue
                    cids = [pi for pi in pg["by_place"][pin] if pi != qi]
                    if len(cids) < 2:
                        continue
                    rng.shuffle(cids)
                    use = cids[:CAP]
                    scored = label_places(
                        pg, pin, qi, use, held_ask, env_m, mid_set, high_set,
                        co, df, n_use,
                    )
                    if not scored:
                        continue
                    rec = scored
                finally:
                    hide_two(co, df, query["keys"], held_ctx, held_ask, +1)
                if rec:
                    recs.append(rec)
    return recs


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
    cut = max(1, int(0.7 * len(windows)))
    t0 = time.time()
    print(f"626 fullfb  {path}  {kind}  train {cut}  test {len(windows) - cut}", flush=True)

    train = trials(windows[:cut], random.Random(args.seed), args)
    test = trials(windows[cut:], random.Random(args.seed + 17), args)
    qsum = defaultdict(float)
    qn = defaultdict(int)
    for rec in train:
        for row in rec:
            qsum[row["key"]] += row["h"]
            qn[row["key"]] += 1

    def q_of(key):
        return qsum[key] / qn[key] if qn[key] else 0.0

    n_te = 0
    ph = ch = rh = 0
    pd = cd = rd = 0
    ora_h = ora_d = 0
    rng = random.Random(args.seed + 99)
    for rec in test:
        n_te += 1
        best = max(q_of(row["key"]) for row in rec)
        pick = next(row for row in rec if q_of(row["key"]) == best)
        cnt = max(rec, key=lambda row: row["count_key"])
        rnd = rng.choice(rec)
        ph += pick["h"]
        ch += cnt["h"]
        rh += rnd["h"]
        pd += pick["d"]
        cd += cnt["d"]
        rd += rnd["d"]
        ora_h += int(any(row["h"] for row in rec))
        ora_d += int(any(row["d"] for row in rec))

    def r(x, den):
        return x / den if den else 0.0

    void = n_te < 40
    Ph, Ch, Rh = r(ph, n_te), r(ch, n_te), r(rh, n_te)
    gate = (not void) and (Ph - max(Ch, Rh)) > 0.05
    print(f"train {len(train)}  test {n_te}  keys {len(qn)}")
    print(
        f"HOPONLY  pick {Ph:.3f}  count {Ch:.3f}  rand {Rh:.3f}  ora {r(ora_h, n_te):.3f}  "
        f"Δ {Ph - max(Ch, Rh):+.3f}"
    )
    print(
        f"DIRECT   pick {r(pd, n_te):.3f}  count {r(cd, n_te):.3f}  "
        f"rand {r(rd, n_te):.3f}  ora {r(ora_d, n_te):.3f}  (not trained)"
    )
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: test hungry.")
    elif gate:
        print("GO FULLFB: features rank hop1 when shown all labels. 624 hole was bandit.")
    else:
        print("STOP FULLFB: even with all labels, votes/recip/count miss hop1. New tape signal.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate), n_train=len(train), n_test=n_te,
        n_keys=len(qn),
        h_pick=Ph, h_count=Ch, h_rand=Rh, h_ora=r(ora_h, n_te),
        d_pick=r(pd, n_te), d_count=r(cd, n_te), d_rand=r(rd, n_te),
        d_ora=r(ora_d, n_te),
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
