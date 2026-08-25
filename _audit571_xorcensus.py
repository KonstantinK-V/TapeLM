"""571: census of genuine two-place gaps. No learner, no new law.

For exactly two neighbors after a READ miss, execute both branches off-policy:

    hop2: 00 / XOR / 11
    hop3: retired for this held. If hop2 hits, the record is answered and the
          route must stop; continuing toward the same held was an invalid arena.

VOID only describes mass. This audit has no mind GATE.

    python _check571_xorcensus.py
    python _audit571_xorcensus.py --seed 1337 \
        --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band

OUT = Path("results/_stage571_xorcensus.json")


def stand_frame(g, by, addr, env_m, exclude=None, cap=8):
    slots = [t for t in by.get(addr, []) if t != exclude]
    if len(slots) < 2:
        return None, set()
    scored = []
    for t in slots[: max(cap, 1)]:
        fr = set(comps(g, t, addr))
        ov = len(fr & env_m)
        scored.append((ov / max(len(fr), 1), ov, t, fr))
    if not scored:
        return None, set()
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return scored[0][2], scored[0][3]


def branch(g, by, addr, v, held, env_m, mid_set):
    stood2, fr2 = stand_frame(g, by, addr, env_m)
    if stood2 is None:
        return dict(h2=0, reach2=0, ov2=0, reach3=0, h3=-1, n3=-1)
    h2 = int(held in fr2)
    if h2:
        return dict(h2=1, reach2=1, ov2=len(fr2 & env_m),
                    reach3=0, h3=-1, n3=0)
    cand3 = [
        c for c in fr2
        if c in mid_set and c not in {v, addr}
    ]
    out = dict(
        h2=h2,
        reach2=1,
        ov2=len(fr2 & env_m),
        reach3=0,
        h3=-1,
        n3=len(cand3),
    )
    if len(cand3) != 1:
        return out
    addr3 = cand3[0]
    stood3, fr3 = stand_frame(g, by, addr3, env_m)
    if stood3 is None:
        return out
    out["reach3"] = 1
    out["h3"] = int(held in fr3)
    return out


def one_probe(g, by, v, s, mid_set, high_set, rng):
    frame = list(comps(g, s, v))
    if len(frame) < 3:
        return None, "frame"
    rng.shuffle(frame)
    held, env = frame[0], set(frame[1:])
    env_m = (env & mid_set) - high_set or (env - high_set)
    if not env_m:
        return None, "env"
    stood, place = stand_frame(g, by, v, env_m, exclude=s)
    if stood is None:
        return None, "stand"
    if held in place:
        return None, "read_hit"
    cand = [
        c for c in place
        if c in mid_set and c != v
    ]
    if len(cand) != 2:
        return None, f"cand{min(len(cand), 3)}"
    rng.shuffle(cand)
    a, b = cand
    return dict(
        a=branch(g, by, a, v, held, env_m, mid_set),
        b=branch(g, by, b, v, held, env_m, mid_set),
    ), None


def windows(pool, n_win, length, rng):
    out = []
    for _ in range(n_win):
        if len(pool) <= length:
            out.append(pool)
        else:
            start = rng.randrange(len(pool) - length + 1)
            out.append(pool[start:start + length])
    return out


def collect(lines, args, rng):
    g = graph(lines, args.frame_max, args.min_fillers)
    if g is None:
        return [], Counter(nograph=1)
    by = mentions(g)
    mid, high, _a, _b = pct_band(g, by)
    mid_set, high_set = set(mid), set(high)
    rows, skip = [], Counter()
    keys = list(mid)
    rng.shuffle(keys)
    for v in keys:
        slots = list(by[v])
        if len(slots) < 8:
            skip["slots"] += 1
            continue
        rng.shuffle(slots)
        for s in slots[: args.cap_probe]:
            row, why = one_probe(g, by, v, s, mid_set, high_set, rng)
            if row is None:
                skip[why] += 1
                continue
            rows.append(row)
    return rows, skip


def category(a, b):
    return f"{int(bool(a))}{int(bool(b))}"


def rate(num, den):
    return num / den if den else 0.0


def summarize(rows):
    st = Counter()
    for row in rows:
        a, b = row["a"], row["b"]
        st["two"] += 1
        c2 = category(a["h2"], b["h2"])
        st[f"h2_{c2}"] += 1
        if c2 in {"10", "01"}:
            st["h2_xor"] += 1
            winner = "a" if c2 == "10" else "b"
            if a["ov2"] == b["ov2"]:
                rank = "a" if st["two"] % 2 else "b"
            else:
                rank = "a" if a["ov2"] > b["ov2"] else "b"
            st["rank_h2"] += int(rank == winner)
        if c2 != "11":
            continue
        st["both_h2"] += 1
        if a["reach3"] != b["reach3"]:
            st["h3_one_reach"] += 1
        if not (a["reach3"] and b["reach3"]):
            continue
        st["h3_pair"] += 1
        c3 = category(a["h3"], b["h3"])
        st[f"h3_{c3}"] += 1
        if c3 in {"10", "01"}:
            st["h3_xor"] += 1
            winner = "a" if c3 == "10" else "b"
            if a["ov2"] == b["ov2"]:
                rank = "a" if st["h3_pair"] % 2 else "b"
            else:
                rank = "a" if a["ov2"] > b["ov2"] else "b"
            st["rank_h3"] += int(rank == winner)
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--n-win", type=int, default=12)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--cap-probe", type=int, default=6)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [
        line.strip() for line in text.split("\n")
        if len(line.strip()) >= min_line
    ]
    pool = all_lines[: int(0.7 * len(all_lines))][: args.lines]
    rng = random.Random(args.seed)
    t0 = time.time()
    print(f"571 XOR census  corpus={path}  {kind}", flush=True)

    rows, skip = [], Counter()
    for lines in windows(pool, args.n_win, args.window_lines, rng):
        part, skipped = collect(lines, args, rng)
        rows.extend(part)
        skip.update(skipped)
    st = summarize(rows)

    n_two = st["two"]
    n_h3 = st["h3_pair"]
    void_h2 = n_two < 20
    void_h3 = n_h3 < 20
    print(
        f"two {n_two}  hop2 00={st['h2_00']} XOR={st['h2_xor']} "
        f"11={st['h2_11']}  xor_rate={rate(st['h2_xor'], n_two):.3f} "
        f"rank_xor={rate(st['rank_h2'], st['h2_xor']):.3f}"
    )
    print(
        f"both_h2 {st['both_h2']}  h3_pair {n_h3}  "
        f"hop3 00={st['h3_00']} XOR={st['h3_xor']} 11={st['h3_11']}  "
        f"xor_rate={rate(st['h3_xor'], n_h3):.3f} "
        f"rank_xor={rate(st['rank_h3'], st['h3_xor']):.3f}  "
        f"one_reach={st['h3_one_reach']}"
    )
    print(f"skip {dict(skip)}")
    print(
        f"VOID_H2 {void_h2}  VOID_H3 {void_h3}  "
        "(census only; no learner GATE)"
    )

    rec = dict(
        seed=args.seed,
        corpus=kind,
        path=str(path),
        n_two=n_two,
        h2_00=st["h2_00"],
        h2_xor=st["h2_xor"],
        h2_11=st["h2_11"],
        h2_xor_rate=rate(st["h2_xor"], n_two),
        rank_h2_xor=rate(st["rank_h2"], st["h2_xor"]),
        both_h2=st["both_h2"],
        n_h3_pair=n_h3,
        h3_00=st["h3_00"],
        h3_xor=st["h3_xor"],
        h3_11=st["h3_11"],
        h3_xor_rate=rate(st["h3_xor"], n_h3),
        rank_h3_xor=rate(st["rank_h3"], st["h3_xor"]),
        h3_one_reach=st["h3_one_reach"],
        skip=dict(skip),
        elapsed_s=round(time.time() - t0, 1),
        void_h2=bool(void_h2),
        void_h3=bool(void_h3),
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
