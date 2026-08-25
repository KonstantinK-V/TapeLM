"""605: learn on unique extras; crowd refuses. No PMI in features.

Offer     unique extras of the pin (597), frozen
Crowd     no unique extra -> REFUSE (601-604)
Hop1      first unique, frozen
Q         among 2+ unique: which extra (counts, no PMI)
Teacher   extra == held
Rivals    first | random | PMI report-only

GATE  Q-rnd > 0.05 AND Q-first > 0.05 on n_uniq>=2
VOID  n_u2 < 40 OR n_crowd < 20
PMI is not a rival and not a feature.

    python _check605_ulearn.py
    python _audit605_ulearn.py --seed 1337 --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band
from _audit589_hop3 import adjust_frame_stats, co_table, env_mid, prefix_windows
from _audit593_mix import bag_of, pmi_rank

OUT = Path("results/_stage605_ulearn.json")


def feat(n_ord, i, df_tok, n_slots):
    ob = 1 if n_ord == 2 else (2 if n_ord == 3 else 3)
    ib = 0 if i == 0 else (1 if i == 1 else 2)
    if n_slots <= 0:
        bb = 1
    elif df_tok * 4 < n_slots:
        bb = 0
    elif df_tok * 2 > n_slots:
        bb = 2
    else:
        bb = 1
    return (ob, ib, bb)


class QTab:
    def __init__(self):
        self.s = defaultdict(float)
        self.n = defaultdict(int)

    def touch(self, key, r):
        self.n[key] += 1
        self.s[key] += (r - self.s[key]) / self.n[key]

    def get(self, key):
        return self.s[key] if self.n[key] else 0.0


def collect(lines, args, rng):
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        return []
    by = mentions(g)
    mid, high, _a, _b = pct_band(g, by)
    mid_set, high_set = set(mid), set(high)
    co, frames, n_fr = co_table(g, by)
    df = g.get("df") or {tok: len(slots) for tok, slots in by.items()}
    n_slots = g["n"]
    out = []
    keys = list(mid)
    rng.shuffle(keys)
    for v in keys:
        slots = list(by.get(v, ()))
        if len(slots) < 2:
            continue
        rng.shuffle(slots)
        for s_q in slots[1: args.cap_probe + 1]:
            frame = list(comps(g, s_q, v))
            if len(frame) < 2:
                continue
            rng.shuffle(frame)
            held, env = frame[0], set(frame[1:])
            if held not in mid_set or held == v:
                continue
            env_m = env_mid(env, mid_set, high_set)
            if not env_m:
                continue
            qtoks = frames.get(s_q)
            if qtoks:
                adjust_frame_stats(co, df, qtoks, -1)
                n_use = max(n_fr - 1, 1)
            else:
                n_use = n_fr
            try:
                bag, uniq = bag_of(g, by, v, s_q, env_m, mid_set)
            finally:
                if qtoks:
                    adjust_frame_stats(co, df, qtoks, +1)
            if not bag:
                continue
            ranked = pmi_rank(bag, env_m, co, df, n_use)
            urows = [
                (feat(len(uniq), i, df.get(tok, 1), n_slots), tok)
                for i, tok in enumerate(uniq)
            ]
            out.append(dict(
                held=held, bag=bag, uniq=uniq, ranked=ranked, urows=urows,
            ))
    return out


def pick_q(urows, tab):
    if not urows:
        return None
    best_key, best_tok = urows[0]
    best_s = tab.get(best_key)
    for key, tok in urows[1:]:
        s = tab.get(key)
        if s > best_s:
            best_key, best_tok, best_s = key, tok, s
    return best_tok


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
    all_lines = [
        ln.strip() for ln in text.split("\n") if len(ln.strip()) >= min_line
    ][: args.lines]
    line_cut = int(0.7 * len(all_lines))
    train_pool, eval_pool = all_lines[:line_cut], all_lines[line_cut:]
    n_train_w = max(1, int(0.7 * args.n_win))
    n_eval_w = max(1, args.n_win - n_train_w)
    train_w = prefix_windows(train_pool, args.window_lines, n_train_w)
    eval_w = prefix_windows(eval_pool, args.window_lines, n_eval_w)
    windows = train_w + eval_w
    rng = random.Random(args.seed)
    rnd = random.Random(args.seed + 41)
    t0 = time.time()
    print(
        f"605 ulearn  {path}  {kind}  train_w={len(train_w)} eval_w={len(eval_w)}",
        flush=True,
    )

    train_eps, eval_eps = [], []
    for lines in train_w:
        train_eps.extend(collect(lines, args, rng))
    for lines in eval_w:
        eval_eps.extend(collect(lines, args, rng))

    tab = QTab()
    for ep in train_eps:
        if len(ep["uniq"]) < 2:
            continue
        held = ep["held"]
        for key, tok in ep["urows"]:
            tab.touch(key, 1.0 if tok == held else 0.0)

    n_u2 = n_c = qh = rh = fh = ph = ref = 0
    for ep in eval_eps:
        held, uniq, ranked = ep["held"], ep["uniq"], ep["ranked"]
        if held in set(uniq):
            if len(uniq) < 2:
                continue
            n_u2 += 1
            qh += int(pick_q(ep["urows"], tab) == held)
            rh += int(uniq[rnd.randrange(len(uniq))] == held)
            fh += int(uniq[0] == held)
            u_set = set(uniq)
            pmi_u = next((tok for tok in ranked if tok in u_set), None)
            ph += int(pmi_u == held)
        else:
            # frozen: no unique -> refuse, not a learned action
            n_c += 1
            ref += 1

    def rate(x, d):
        return x / d if d else 0.0

    fq, fr, ff, fp = rate(qh, n_u2), rate(rh, n_u2), rate(fh, n_u2), rate(ph, n_u2)
    fref = rate(ref, n_c)
    void = n_u2 < 40 or n_c < 20
    gate = (not void) and (fq - fr > 0.05) and (fq - ff > 0.05)
    print(
        f"u2 {n_u2}  crowd {n_c}  Q {fq:.3f}  rnd {fr:.3f}  first {ff:.3f}  "
        f"PMI {fp:.3f}  refuse {fref:.3f}  keys {len(tab.s)}"
    )
    print(f"Q-rnd {fq - fr:+.3f}  Q-first {fq - ff:+.3f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("VOID: u2 or crowd thin.")
    elif gate:
        print("GO LEARN: Q picks among unique extras; crowd refuses. PMI not in features.")
    else:
        print("STOP: counts do not beat first/random among unique.")

    rec = dict(
        seed=args.seed, corpus=kind, path=str(path),
        n_win=len(windows), elapsed_s=round(time.time() - t0, 1),
        void=bool(void), gate=bool(gate),
        n_u2=n_u2, n_crowd=n_c, n_keys=len(tab.s),
        fill_q=fq, fill_rnd=fr, fill_first=ff, fill_pmi=fp, refuse=fref,
        d_rnd=fq - fr, d_first=fq - ff,
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
