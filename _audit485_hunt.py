"""485 NIGHT: uncontrolled hunt on big TinyStories.

Not a ladder gate. Sliding raw windows:

  HUNT   learn cheap pre(P); search for soft-majority value with unique or
         narrow fan-in (1..k other places). Finding a hop IS the result.
  WANDER mark LIVE/SOFT/DEAD; continue by any fan-in (no unique required).

On raw TinyStories true const places are rare — soft majority (>=0.6) is the
crazy concession. unique_next~0 still expected; narrow + wander are the play.

    python _audit485_hunt.py --seed 1337 --steps 4000
    python _audit485_hunt.py --corpus data/_tinystories_raw_scale.txt --steps 8000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes
from _audit440_compose import think_place

OUT = Path("results/_stage485_hunt.json")
CORPUS_CANDS = (
    Path("data/_tinystories_raw_scale.txt"),
    Path("data/external_tinystories_1m_85.txt"),
    Path("data/_tinystories_train.txt"),
    Path("data/external_tinystories_mini.txt"),
)


def pick_corpus(explicit: str) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"no corpus {p}")
        return p
    for p in CORPUS_CANDS:
        if p.exists():
            return p
    raise SystemExit("no tinystories corpus")


def load_lines(path: Path, n_bytes: int, min_line: int, rng: random.Random) -> list[str]:
    text = path.open("r", encoding="utf-8", errors="ignore").read(n_bytes)
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    if len(lines) > 80_000:
        rng.shuffle(lines)
        lines = lines[:80_000]
    return lines


def build_window(lines: list[str], rng: random.Random, win: int, frame_max: int):
    if len(lines) < max(win // 4, 50):
        return None
    w = min(win, len(lines))
    s0 = rng.randrange(max(len(lines) - w, 1))
    chunk = lines[s0:s0 + w]
    keep, toks, owner = tframes.frame_keep(chunk, frame_max, 2)
    if not keep or len(keep) < 8:
        return None
    place, value, line, keys = [], [], [], []
    for (_w0, left, right), ps in keep:
        pname = f"{' '.join(left)}|{' '.join(right)}"
        ks = {x for x in list(left) + list(right) if x}
        for i in ps:
            place.append(pname)
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
    return dict(
        place=place, value=value, line=line, keys=keys,
        slots_at=slots_at, by_key=by_key, n=len(place),
    )


def pre(P, g):
    sl = g["slots_at"][P]
    vs = [g["value"][i] for i in sl]
    n_s = len(sl)
    n_k = len({k for i in sl for k in g["keys"][i]})
    maj = Counter(vs).most_common(1)[0][1] / max(len(vs), 1)
    return (min(n_s, 12), min(n_k, 20), int(maj >= 0.999), int(maj >= 0.6))


def place_value(P, g, min_frac: float = 0.6):
    vs = [g["value"][i] for i in g["slots_at"][P]]
    if not vs:
        return None, 0.0
    v, c = Counter(vs).most_common(1)[0]
    frac = c / len(vs)
    if frac < min_frac:
        return None, frac
    return v, frac


def unique_next(P, g, min_frac: float = 0.6):
    if len(g["slots_at"][P]) < 2:
        return None
    v, frac = place_value(P, g, min_frac)
    if v is None:
        return None
    cands = g["by_key"].get(v, set()) - {P}
    if len(cands) != 1:
        return None
    return next(iter(cands)), v, frac


def narrow_next(P, g, kmax: int = 5, min_frac: float = 0.6):
    if len(g["slots_at"][P]) < 2:
        return None
    v, frac = place_value(P, g, min_frac)
    if v is None:
        return None
    cands = g["by_key"].get(v, set()) - {P}
    if not (1 <= len(cands) <= kmax):
        return None
    return cands, v, len(cands), frac


def touch(table, tot, win, key, r):
    tot[key] += 1
    win[key] += r
    table[key] = win[key] / tot[key]


def pick_by_q(opts, table, keyfn, rng, eps):
    if not opts:
        return None
    if rng.random() < eps:
        return rng.choice(opts)
    best, br = None, -1e9
    for x in opts:
        v = table.get(keyfn(x), 0.0)
        if v > br:
            br, best = v, x
    if best is None or br <= 0:
        return rng.choice(opts)
    return best


def hunt_episode(g, rng, q_pre, tot, win, eps, budget: int, kmax: int):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if len(places) < 4:
        return dict(tried=0, found=0, hopped=0, narrow=0, unique=0)
    tried = found = hopped = narrow = unique = 0
    seen = set()
    for _ in range(budget):
        pool = [P for P in places if P not in seen] or places
        P = pick_by_q(pool, q_pre, lambda x: pre(x, g), rng, eps)
        if P is None:
            break
        seen.add(P)
        tried += 1
        k = pre(P, g)
        nxt_u = unique_next(P, g)
        nxt_n = None if nxt_u else narrow_next(P, g, kmax)
        if nxt_u is None and nxt_n is None:
            touch(q_pre, tot, win, k, -0.05)
            continue
        found += 1
        if nxt_u is not None:
            unique += 1
            touch(q_pre, tot, win, k, 0.8)
            R = nxt_u[0]
        else:
            narrow += 1
            cands, _v, w, _f = nxt_n
            touch(q_pre, tot, win, k, 0.25 + 0.15 * (kmax - w + 1) / max(kmax, 1))
            R = rng.choice(list(cands))
        pin = think_place(list(g["slots_at"][R]), g["value"], rng)
        if pin is not None:
            hopped += 1
            touch(q_pre, tot, win, k, 0.4)
        else:
            touch(q_pre, tot, win, k, 0.1)
        break
    return dict(tried=tried, found=found, hopped=hopped, narrow=narrow, unique=unique)


def wander_episode(g, rng, q_mark, tot, win, eps):
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if len(places) < 4:
        return dict(marks=0, live=0, cont=0, chain=0, pin_ok=0)
    tape = {}
    P0 = rng.choice(places)
    pin0 = think_place(list(g["slots_at"][P0]), g["value"], rng)
    v0, frac0 = place_value(P0, g, 0.5)
    if pin0 is not None:
        v0 = g["value"][pin0]
        tape[P0] = "LIVE"
        touch(q_mark, tot, win, "LIVE", 0.15)
        pin_ok0 = 1
    elif v0 is not None:
        tape[P0] = "SOFT"
        touch(q_mark, tot, win, "SOFT", 0.05)
        pin_ok0 = 0
    else:
        tape[P0] = "DEAD"
        touch(q_mark, tot, win, "DEAD", -0.1)
        return dict(marks=1, live=0, cont=0, chain=0, pin_ok=0)
    nbrs = list(g["by_key"].get(v0, set()) - {P0})
    if not nbrs:
        return dict(marks=1, live=int(tape[P0] == "LIVE"), cont=0, chain=0, pin_ok=pin_ok0)
    sense = max(q_mark.get("LIVE", 0.0), q_mark.get("SOFT", 0.0))
    go = (rng.random() < max(eps, 0.25)) or (sense > 0)
    if not go:
        return dict(marks=1, live=int(tape[P0] == "LIVE"), cont=0, chain=0, pin_ok=pin_ok0)
    R = rng.choice(nbrs)
    pin1 = think_place(list(g["slots_at"][R]), g["value"], rng)
    cont = 1
    pin_ok1 = int(pin1 is not None)
    if pin_ok1:
        tape[R] = "LIVE"
        touch(q_mark, tot, win, "LIVE", 0.4)
        v1 = g["value"][pin1]
    else:
        v1, _f = place_value(R, g, 0.5)
        if v1 is not None:
            tape[R] = "SOFT"
            touch(q_mark, tot, win, "SOFT", 0.15)
        else:
            tape[R] = "DEAD"
            touch(q_mark, tot, win, "DEAD", 0.05)
            touch(q_mark, tot, win, "LIVE", -0.15)
            return dict(marks=len(tape), live=int(any(m == "LIVE" for m in tape.values())),
                        cont=1, chain=0, pin_ok=pin_ok0 + pin_ok1)
    chain = 0
    nbrs2 = list(g["by_key"].get(v1, set()) - {P0, R})
    if nbrs2 and max(q_mark.get("LIVE", 0.0), q_mark.get("SOFT", 0.0)) > -0.05:
        T = rng.choice(nbrs2)
        pin2 = think_place(list(g["slots_at"][T]), g["value"], rng)
        chain = 1
        if pin2 is not None:
            touch(q_mark, tot, win, "LIVE", 0.35)
        else:
            touch(q_mark, tot, win, "SOFT", 0.1)
    return dict(marks=len(tape), live=int(any(m == "LIVE" for m in tape.values())),
                cont=cont, chain=chain, pin_ok=pin_ok0 + pin_ok1)


def oracle_counts(g, kmax: int = 5) -> tuple[int, int, int]:
    n_c = n_u = n_n = 0
    for P, sl in g["slots_at"].items():
        if len(sl) < 2:
            continue
        v, _f = place_value(P, g, 0.6)
        if v is None:
            continue
        n_c += 1
        if unique_next(P, g) is not None:
            n_u += 1
        elif narrow_next(P, g, kmax) is not None:
            n_n += 1
    return n_c, n_u, n_n


def random_find_rate(g, rng, n_try: int, kmax: int) -> float:
    places = [P for P, sl in g["slots_at"].items() if len(sl) >= 2]
    if not places:
        return 0.0
    hit = 0
    for _ in range(n_try):
        P = rng.choice(places)
        if unique_next(P, g) is not None or narrow_next(P, g, kmax) is not None:
            hit += 1
    return hit / max(n_try, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--frame-max", type=int, default=4)
    ap.add_argument("--hunt-budget", type=int, default=12)
    ap.add_argument("--narrow-k", type=int, default=5)
    ap.add_argument("--min-line", type=int, default=20)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    path = pick_corpus(args.corpus)
    rng0 = random.Random(args.seed)
    print(f"485 NIGHT hunt  corpus={path}  steps={args.steps}  narrow_k={args.narrow_k}")
    lines = load_lines(path, args.bytes, args.min_line, rng0)
    print(f"line pool {len(lines)}")

    q_pre, tot_pre, win_pre = {}, defaultdict(int), defaultdict(float)
    q_mark, tot_m, win_m = {}, defaultdict(int), defaultdict(float)
    sum_h = defaultdict(float)
    n_win_ok = 0
    t0 = time.time()
    rng = random.Random(args.seed)

    for i in range(args.steps):
        if (i + 1) % 100 == 0:
            tframes._KEEP_MEMO.clear()
        g = build_window(lines, rng, args.window, args.frame_max)
        if g is None:
            continue
        n_win_ok += 1
        eps = max(0.05, 0.45 * (1.0 - i / max(args.steps, 1)))
        n_c, n_u, n_n = oracle_counts(g, args.narrow_k)
        sum_h["oracle_c"] += n_c
        sum_h["oracle_u"] += n_u
        sum_h["oracle_n"] += n_n
        base = random_find_rate(g, rng, 8, args.narrow_k)
        sum_h["base_u"] += base
        h = hunt_episode(
            g, rng, q_pre, tot_pre, win_pre, eps, args.hunt_budget, args.narrow_k
        )
        for k, v in h.items():
            sum_h[k] += v
        w = wander_episode(g, rng, q_mark, tot_m, win_m, eps)
        for k, v in w.items():
            sum_h["w_" + k] += v
        if (i + 1) % args.log_every == 0:
            el = time.time() - t0
            hu = sum_h["found"] / max(n_win_ok, 1)
            bu = sum_h["base_u"] / max(n_win_ok, 1)
            ou = sum_h["oracle_u"] / max(sum_h["oracle_c"], 1)
            on = sum_h["oracle_n"] / max(sum_h["oracle_c"], 1)
            print(
                f"  step {i+1}/{args.steps}  win {n_win_ok}  "
                f"hunt {hu:.3f} (u={sum_h['unique']:.0f} n={sum_h['narrow']:.0f})  "
                f"rand {bu:.3f}  oracle_u/n {ou:.4f}/{on:.4f}  "
                f"wander {sum_h['w_cont']/max(n_win_ok,1):.3f}  "
                f"chain {sum_h['w_chain']/max(n_win_ok,1):.3f}  "
                f"pin {sum_h['w_pin_ok']/max(n_win_ok,1):.2f}  "
                f"q_pre {len(q_pre)} mark {{{', '.join(f'{k}:{v:.2f}' for k,v in q_mark.items())}}}  "
                f"{el:.0f}s",
                flush=True,
            )
            _dump(args, path, n_win_ok, el, sum_h, q_pre, q_mark, partial=True)

    el = time.time() - t0
    rec = _dump(args, path, n_win_ok, el, sum_h, q_pre, q_mark, partial=False)
    print("---- done ----", flush=True)
    print("hunt found/win", round(rec["hunt"]["found_per_win"], 4),
          "unique", rec["hunt"]["unique"], "narrow", rec["hunt"]["narrow"],
          "vs rand", round(rec["hunt"]["random_find"], 4),
          "lift", round(rec["hunt"]["lift"], 4), flush=True)
    print("wander cont/chain/pin",
          round(rec["wander"]["cont_per_win"], 4),
          round(rec["wander"]["chain_per_win"], 4),
          round(rec["wander"]["pin_ok_per_win"], 4),
          "q_mark", q_mark, flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


def _dump(args, path, n_win_ok, el, sum_h, q_pre, q_mark, partial: bool):
    rec = dict(
        seed=args.seed,
        corpus=str(path),
        steps=args.steps,
        n_windows=n_win_ok,
        elapsed_s=round(el, 1),
        partial=partial,
        hunt=dict(
            tried=sum_h["tried"],
            found=sum_h["found"],
            hopped=sum_h["hopped"],
            unique=sum_h["unique"],
            narrow=sum_h["narrow"],
            found_per_win=sum_h["found"] / max(n_win_ok, 1),
            hopped_per_win=sum_h["hopped"] / max(n_win_ok, 1),
            random_find=sum_h["base_u"] / max(n_win_ok, 1),
            oracle_unique=sum_h["oracle_u"] / max(sum_h["oracle_c"], 1),
            oracle_narrow=sum_h["oracle_n"] / max(sum_h["oracle_c"], 1),
            oracle_soft=sum_h["oracle_c"],
            lift=(sum_h["found"] / max(n_win_ok, 1))
                 - (sum_h["base_u"] / max(n_win_ok, 1)),
            n_pre_keys=len(q_pre),
            top_pre=[[list(k), v] for k, v in
                     sorted(q_pre.items(), key=lambda x: -x[1])[:12]],
        ),
        wander=dict(
            cont_per_win=sum_h["w_cont"] / max(n_win_ok, 1),
            chain_per_win=sum_h["w_chain"] / max(n_win_ok, 1),
            pin_ok_per_win=sum_h["w_pin_ok"] / max(n_win_ok, 1),
            q_mark={k: round(v, 4) for k, v in q_mark.items()},
        ),
        note="uncontrolled night; soft majority; not a ladder gate",
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    key = str(args.seed) + ("_partial" if partial else "")
    if not partial:
        prev.pop(f"{args.seed}_partial", None)
    prev[key if partial else str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    return rec


if __name__ == "__main__":
    raise SystemExit(main())
