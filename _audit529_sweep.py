"""529: experiment grid — novelty/cover teachers on frozen v1. Contract walk fixed."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _audit511_ring import graph, mentions, pick_corpus
from _audit517_window import comps
from _audit518_reldf import pct_band
from _audit527_learn import majority, v1_nodes

OUT = Path("results/_stage529_sweep.json")
STORIES = "data/_tinystories_train.txt"
MINI = "data/external_tinystories_mini.txt"


def cover(taken, held):
    if not held:
        return 0.0
    return len(set(taken) & held) / len(held)


def novel_hit(c, held, seen, maj):
    return c in held and c not in seen and c != maj


def run_ep(nodes, held, choose, band):
    if not nodes:
        return dict(hops=0, cover=0.0, take=[])
    take = [nodes[0]]
    for c in nodes[1:]:
        if not choose(band):
            break
        take.append(c)
    return dict(hops=len(take), cover=cover(take, held), take=take)


def mean_ep(rows):
    n = max(len(rows), 1)
    return dict(
        n=len(rows),
        hops=sum(r["hops"] for r in rows) / n,
        cover=sum(r["cover"] for r in rows) / n,
    )


def trials(g, by, vs, rng, cap=12):
    out = []
    for v in vs:
        sl = list(by[v])
        if len(sl) < 8:
            continue
        rng.shuffle(sl)
        sl = sl[:cap]
        held = set(comps(g, sl[0], v))
        rest = sl[1:]
        if not held or not rest:
            continue
        maj = majority(g, rest, v)
        out.append((v, rest, held, maj))
    return out


def train_step_nov(Q, band, nodes, held, maj, rng, eps, lr, c_step, hop1_free):
    """Per-step: +1 new held≠maj, else 0; −c on extra hops (optional hop1 free)."""
    seen = set()
    for i, c in enumerate(nodes):
        new = novel_hit(c, held, seen, maj)
        if c in held:
            seen.add(c)
        cost = 0.0 if (hop1_free and i == 0) else c_step
        r = (1.0 if new else 0.0) - cost
        if i == 0:
            Q[(band, "go")] += lr * (r - Q[(band, "go")])
            continue
        qg, qs = Q[(band, "go")], Q[(band, "stop")]
        go = rng.choice([True, False]) if rng.random() < eps else qg >= qs
        if not go:
            # correct-stop bonus if no remaining novelty
            rest_nov = any(novel_hit(x, held, seen, maj) for x in nodes[i:])
            rs = 0.1 if not rest_nov else 0.0
            Q[(band, "stop")] += lr * (rs - Q[(band, "stop")])
            break
        Q[(band, "go")] += lr * (r - Q[(band, "go")])


def train_episodic(Q, band, nodes, held, maj, rng, eps, lr, c_step, mode):
    """Roll out with eps-greedy, then credit go/stop with episode return."""
    if not nodes:
        return
    take = [nodes[0]]
    decisions = []  # (band_action, chose_go)
    seen = {nodes[0]} if nodes[0] in held else set()
    for i, c in enumerate(nodes[1:], start=1):
        qg, qs = Q[(band, "go")], Q[(band, "stop")]
        go = rng.choice([True, False]) if rng.random() < eps else qg >= qs
        decisions.append(go)
        if not go:
            break
        take.append(c)
        if c in held:
            seen.add(c)
    cov = cover(take, held)
    cov1 = cover(nodes[:1], held)
    hops = len(take)
    if mode == "adv":
        R = cov - cov1 - c_step * max(hops - 1, 0)
    else:  # cover_end
        R = cov - c_step * hops
    # hop1 always: touch go lightly with novelty of hop1
    n0 = novel_hit(nodes[0], held, set(), maj)
    Q[(band, "go")] += lr * ((1.0 if n0 else 0.0) - Q[(band, "go")])
    for go in decisions:
        if go:
            Q[(band, "go")] += lr * (R - Q[(band, "go")])
        else:
            Q[(band, "stop")] += lr * (R - Q[(band, "stop")])


def build_graph(path, kind, min_line, window, seed, nbytes, frame_max, min_fillers, lines_cap):
    text = path.open("r", encoding="utf-8", errors="ignore").read(nbytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = all_lines[: int(0.7 * len(all_lines))][:lines_cap]
    rng = random.Random(seed)
    if window < len(pool):
        s0 = rng.randrange(len(pool) - window + 1)
        lines = pool[s0:s0 + window]
    else:
        lines = pool
    g = graph(lines, frame_max, min_fillers)
    return g, lines, rng


def one_run(cfg):
    path, kind, min_line = pick_corpus(cfg["corpus"] or None)
    g, lines, rng = build_graph(
        path, kind, min_line, cfg["window"], cfg["seed"], cfg["bytes"],
        cfg["frame_max"], cfg["min_fillers"], cfg["lines"])
    if g is None:
        return dict(void=True, reason="no_tape")
    by = mentions(g)
    mid, high, p25, p75 = pct_band(g, by)
    k = 200.0 / max(g["n"], 1)
    high_set = set(high)
    rng.shuffle(mid)
    rng.shuffle(high)
    cut_m, cut_h = int(0.6 * len(mid)), int(0.6 * len(high))
    train_v = mid[:cut_m] + high[:cut_h]
    test_m, test_h = mid[cut_m:], high[cut_h:]
    Q = defaultdict(float)
    lr, cache = 0.2, {}
    teacher = cfg["teacher"]
    for _ in range(cfg["epochs"]):
        rng.shuffle(train_v)
        for v, rest, held, maj in trials(g, by, train_v, rng):
            saved = by[v]
            by[v] = rest
            cache.pop(v, None)
            nodes = v1_nodes(g, by, v, cache, k, high_set)
            by[v] = saved
            if not nodes:
                continue
            band = "high" if v in high_set else "mid"
            if cfg.get("mid_only") and band == "high":
                continue
            if teacher in ("nov", "nov_stop"):
                train_step_nov(
                    Q, band, nodes, held, maj, rng, cfg["eps"], lr,
                    cfg["c_step"], cfg.get("hop1_free", True))
            elif teacher in ("cover_end", "adv"):
                train_episodic(
                    Q, band, nodes, held, maj, rng, cfg["eps"], lr,
                    cfg["c_step"], "adv" if teacher == "adv" else "cover_end")

    def learned(band):
        return Q[(band, "go")] >= Q[(band, "stop")]

    def once(_b):
        return False

    def always(_b):
        return True

    def collect(vs, choose):
        rows = []
        rr = random.Random(cfg["seed"] + 9)
        for v, rest, held, maj in trials(g, by, vs, rr):
            saved = by[v]
            by[v] = rest
            cache.pop(v, None)
            nodes = v1_nodes(g, by, v, cache, k, high_set)
            by[v] = saved
            band = "high" if v in high_set else "mid"
            rows.append(run_ep(nodes, held, choose, band))
        return mean_ep(rows)

    lm = collect(test_m, learned)
    lh = collect(test_h, learned)
    a1 = collect(test_m, once)
    ag = collect(test_m, always)
    ceiling = ag["cover"] - a1["cover"]
    thin_mid = lm["n"] < 15
    thin_ceil = ceiling <= 0.05
    void = thin_mid or thin_ceil
    gate = (not void) and (lm["cover"] > a1["cover"] + 0.05) and (
        lm["hops"] < ag["hops"] - 0.5) and (lh["hops"] < 1.5)
    tag = "wiki" if kind == "wiki" else Path(path).stem
    return dict(
        name=cfg["name"], seed=cfg["seed"], tag=tag, n_lines=len(lines),
        teacher=teacher, c_step=cfg["c_step"], epochs=cfg["epochs"],
        window=cfg["window"], mid_only=bool(cfg.get("mid_only")),
        p25=p25, p75=p75, ceiling=ceiling,
        learn_mid=lm, learn_high=lh, hop1=a1, allgo=ag,
        Q={f"{a}_{b}": round(v, 4) for (a, b), v in Q.items()},
        void=bool(void), thin_ceil=bool(thin_ceil), gate=bool(gate),
    )


def configs():
    """Grid: stories + mini, windows, costs, teachers."""
    out = []
    seeds = (1337, 8642, 2890)
    # A: ceiling hunt on stories — windows × soft nov
    for W in (200, 300, 400, 500, 700, 1000):
        for s in seeds:
            out.append(dict(
                name=f"nov_w{W}", teacher="nov", c_step=0.02, hop1_free=True,
                window=W, seed=s, epochs=8, eps=0.25, corpus=STORIES,
                bytes=30_000_000, frame_max=3, min_fillers=2, lines=25000,
                mid_only=False,
            ))
    # B: cost sweep at W=400 stories
    for c in (0.0, 0.01, 0.02, 0.05):
        for s in seeds:
            out.append(dict(
                name=f"nov_c{c}", teacher="nov", c_step=c, hop1_free=True,
                window=400, seed=s, epochs=8, eps=0.25, corpus=STORIES,
                bytes=30_000_000, frame_max=3, min_fillers=2, lines=25000,
            ))
    # C: episodic teachers at best-ish W=400
    for t in ("cover_end", "adv", "nov_stop"):
        for s in seeds:
            out.append(dict(
                name=t, teacher=t, c_step=0.01, hop1_free=True,
                window=400, seed=s, epochs=10, eps=0.3, corpus=STORIES,
                bytes=30_000_000, frame_max=3, min_fillers=2, lines=25000,
            ))
    # D: mid_only + adv
    for s in seeds:
        out.append(dict(
            name="adv_mid", teacher="adv", c_step=0.01, hop1_free=True,
            window=400, seed=s, epochs=10, eps=0.3, corpus=STORIES,
            bytes=30_000_000, frame_max=3, min_fillers=2, lines=25000,
            mid_only=True,
        ))
    # E: mini corpus if present (denser?), W=400
    if Path(MINI).exists():
        for s in seeds:
            out.append(dict(
                name="mini_adv", teacher="adv", c_step=0.01, hop1_free=True,
                window=400, seed=s, epochs=10, eps=0.3, corpus=MINI,
                bytes=30_000_000, frame_max=3, min_fillers=2, lines=25000,
            ))
            out.append(dict(
                name="mini_nov", teacher="nov", c_step=0.01, hop1_free=True,
                window=400, seed=s, epochs=8, eps=0.25, corpus=MINI,
                bytes=30_000_000, frame_max=3, min_fillers=2, lines=25000,
            ))
    # F: frame_max=4 denser frames on stories 400
    for s in seeds:
        out.append(dict(
            name="nov_f4", teacher="nov", c_step=0.01, hop1_free=True,
            window=400, seed=s, epochs=8, eps=0.25, corpus=STORIES,
            bytes=30_000_000, frame_max=4, min_fillers=2, lines=25000,
        ))
        out.append(dict(
            name="adv_f4", teacher="adv", c_step=0.01, hop1_free=True,
            window=400, seed=s, epochs=10, eps=0.3, corpus=STORIES,
            bytes=30_000_000, frame_max=4, min_fillers=2, lines=25000,
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one seed only")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    cfgs = configs()
    if args.quick:
        cfgs = [c for c in cfgs if c["seed"] == 1337]
    rows = []
    print(f"529 sweep  n={len(cfgs)}", flush=True)
    for i, cfg in enumerate(cfgs):
        rec = one_run(cfg)
        rows.append(rec)
        st = "GO" if rec.get("gate") else ("VOID" if rec.get("void") else "STOP")
        print(
            f"[{i+1}/{len(cfgs)}] {rec.get('name')} s{rec.get('seed')} "
            f"W{rec.get('n_lines')} ceil {rec.get('ceiling', 0):+.3f} "
            f"Lcov {rec.get('learn_mid', {}).get('cover', 0):.3f} "
            f"H1 {rec.get('hop1', {}).get('cover', 0):.3f} "
            f"AG {rec.get('allgo', {}).get('cover', 0):.3f}  {st}",
            flush=True,
        )
    # summary: best by ceiling, best gate, best learn-hop1 gap among non-void
    alive = [r for r in rows if not r.get("void") and r.get("learn_mid")]
    goes = [r for r in rows if r.get("gate")]
    alive.sort(key=lambda r: r["ceiling"], reverse=True)
    print("\n=== TOP ceiling (non-VOID) ===", flush=True)
    for r in alive[:12]:
        gap = r["learn_mid"]["cover"] - r["hop1"]["cover"]
        print(
            f"  {r['name']:12} s{r['seed']} W{r['n_lines']} "
            f"ceil {r['ceiling']:+.3f}  L-H1 {gap:+.3f}  "
            f"hops L {r['learn_mid']['hops']:.1f} AG {r['allgo']['hops']:.1f}",
            flush=True,
        )
    print(f"\nGO count: {len(goes)} / {len(rows)}", flush=True)
    for r in goes:
        print(f"  GO {r['name']} s{r['seed']} W{r['n_lines']} ceil {r['ceiling']:+.3f}",
              flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(n=len(rows), go=len(goes), rows=rows)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
