"""PLACE-WALK: 553–557 + W wire + W[env] LIVE/DEAD (565/566 OPEN).

    STAND JACC -> READ
      unique mid -> PIN -> hop2
        hit  -> W[env]=(addr, LIVE)  -> same env reuses mark
        miss -> W[env]=(addr, DEAD)  -> same env does not walk that addr
      many -> REFUSE, no write

566 OPEN 1/3: LIVE holds; DEAD-follow rare on stories (VOID!=STOP). 481 not closed.
Do not chase DEAD with more windows. Learn not baked.

    python _place_walk.py --corpus data/_tinystories_train.txt
"""
from __future__ import annotations

import argparse
import random
import re

from _audit511_ring import comps, graph, mentions, pick_corpus
from _audit518_reldf import pct_band

LAW = """PLACE-WALK
  STAND  JACC mention of v vs env
  READ   frame
  PIN    unique mid -> hop2
           hit  -> W[env]=(addr, LIVE)  reuse walks mark
           miss -> W[env]=(addr, DEAD)  reuse skips that addr
  REFUSE many mid -> stop, no write
  566 OPEN: LIVE ok; DEAD thin on stories. Not chasing DEAD windows.
"""


def toks(s: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z0-9']+|[.!?]", s) if t]


def load_tape(args, need_token: str | None = None):
    path, kind, min_line = pick_corpus(args.corpus or None)
    text = path.open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= min_line]
    pool = lines[: int(0.7 * len(lines))][: args.lines]
    rng = random.Random(args.seed)
    L = args.window_lines

    def line_has(ln: str, tok: str) -> bool:
        low = tok.lower()
        return any(t.lower() == low for t in toks(ln))

    def build(win):
        g = graph(win, args.frame_max, args.min_fillers)
        if g is None:
            return None
        by = mentions(g)
        mid, high, p25, p75 = pct_band(g, by)
        return dict(path=path, kind=kind, g=g, by=by, mid=set(mid), high=set(high),
                    p25=p25, p75=p75, rng=rng, n_lines=len(win))

    if need_token:
        hits = [i for i, ln in enumerate(pool) if line_has(ln, need_token)]
        if not hits:
            raise SystemExit(f"token {need_token!r} not in corpus pool "
                             f"({len(pool)} lines)")
        rng.shuffle(hits)
        for i in hits[:24]:
            if len(pool) > L:
                s0 = max(0, min(i - L // 2, len(pool) - L))
                win = pool[s0:s0 + L]
            else:
                win = pool
            tape = build(win)
            if tape is None:
                continue
            by = tape["by"]
            if need_token in by:
                return tape
            low = need_token.lower()
            for k in by:
                if k.lower().rstrip(".,!?;:'\"") == low:
                    tape["alias"] = k
                    return tape
        raise SystemExit(f"token {need_token!r} in text but not on frame tape "
                         f"(tried {min(24, len(hits))} windows)")
    if len(pool) > L:
        s0 = rng.randrange(len(pool) - L + 1)
        win = pool[s0:s0 + L]
    else:
        win = pool
    tape = build(win)
    if tape is None:
        raise SystemExit("no tape")
    return tape


def env_of(tokens: list[str], hide: str, mid: set, high: set) -> set:
    env = {t for t in tokens if t != hide}
    env_m = (env & mid) - high
    if not env_m:
        env_m = env - high
    return env_m


def stand_jacc(g, by, v, env_m, rng, exclude=None, cap=32):
    sl = [t for t in by.get(v, []) if t != exclude]
    if not sl:
        return None, None, "no_mentions"
    rng.shuffle(sl)
    scored = []
    for t in sl[: max(cap, 1)]:
        fr = set(comps(g, t, v))
        if not fr:
            continue
        ov = len(fr & env_m)
        scored.append((ov / max(len(fr), 1), ov, t, fr))
    if not scored:
        return None, None, "empty_frames"
    scored.sort(key=lambda x: (-x[0], -x[1]))
    _j, _ov, slot, fr = scored[0]
    return slot, fr, None


def hop2_stand(g, by, addr, env_m, rng, cap=8):
    sl = list(by.get(addr, []))
    if len(sl) < 2:
        return None, "thin_addr"
    rng.shuffle(sl)
    scored = []
    for t in sl[: max(cap, 1)]:
        fr = set(comps(g, t, addr))
        ov = len(fr & env_m)
        scored.append((ov / max(len(fr), 1), fr, t))
    if not scored:
        return None, "no_frame"
    scored.sort(key=lambda x: -x[0])
    return scored[0][1], None


def hop_hit(g, by, addr, env_m, held, rng):
    fr2, why = hop2_stand(g, by, addr, env_m, rng)
    if why:
        return False, None
    return held in fr2, fr2


def rank_cand(g, by, cand, env_m, rng):
    scored = []
    for a in cand:
        sl = list(by.get(a, []))
        fr = set(comps(g, sl[0], a)) if sl else set()
        ov = len(fr & env_m)
        scored.append((-ov, a))
    scored.sort()
    return [a for _ov, a in scored]


def slot_lines(lines, frame_max, min_fillers):
    """Audit hook: line pool for one_query (unchanged text lines)."""
    return lines


def one_query(g, by, v, s, mid_set, high_set, lines, slines, W_e=None, rng=None):
    """Single probe: STAND -> PIN|REFUSE -> hop2. Same law as place_walk."""
    if W_e is None:
        W_e = {}
    if rng is None:
        rng = random.Random((hash(v) ^ hash(s)) & 0xFFFF)
    frame = list(comps(g, s, v))
    if len(frame) < 3:
        return None
    held, env = frame[0], set(frame[1:])
    env_m = (env & mid_set) - high_set or (env - high_set)
    if not env_m:
        return None
    ek = frozenset(env_m)
    reuse = 0

    if ek in W_e:
        addr, mark = W_e[ek]
        reuse = 1
        if mark == "DEAD":
            return dict(hop="REFUSE", hit1=0, hit2=0, reuse=reuse, n_cand=0,
                        why="dead_mark")
        hit, fr2 = hop_hit(g, by, addr, env_m, held, rng)
        return dict(hop="PIN", hit1=int(hit), hit2=int(hit), reuse=reuse,
                    n_cand=1, pin=addr, mark=mark)

    rest = [x for x in by.get(v, []) if x != s]
    if len(rest) < 2:
        return None
    slot, fr, why = stand_jacc(g, by, v, env_m, rng, exclude=s)
    if why:
        return None
    if held in fr:
        return dict(hop="READ", hit1=1, hit2=0, reuse=0, n_cand=0,
                    why="read_hit")
    cand = [c for c in fr if c in mid_set and c != v]
    n = len(cand)
    n_key = min(n, 5)
    if n == 0:
        return dict(hop="REFUSE", hit1=0, hit2=0, reuse=0, n_cand=0, why="no_mid")
    if n >= 3:
        return dict(hop="REFUSE", hit1=0, hit2=0, reuse=0, n_cand=3, why="many")

    addrs = cand if n == 1 else rank_cand(g, by, cand, env_m, rng)
    hit = False
    addr = addrs[0]
    fr2 = None
    for addr in addrs:
        hit, fr2 = hop_hit(g, by, addr, env_m, held, rng)
        if hit:
            break
    if not hit:
        W_e[ek] = (addrs[0], "DEAD")
        return dict(hop="REFUSE", hit1=0, hit2=0, reuse=0, n_cand=n_key,
                    why="dead1")
    W_e[ek] = (addr, "LIVE")
    return dict(hop="PIN", hit1=1, hit2=1, reuse=0, n_cand=n_key, pin=addr,
                mark="LIVE")


def place_walk(g, by, v, env_m, mid, high, rng, exclude=None,
               W_stood=None, W_env=None, held=None):
    """W[env]=(addr, LIVE|DEAD). LIVE reuse walks; DEAD reuse skips."""
    if W_stood is None:
        W_stood = {}
    if W_env is None:
        W_env = {}
    ek = frozenset(env_m)
    out = dict(v=v, action=None, stand=None, frame=[], pin=None, hop2=None,
               refuse=False, why=None, wrote=False, reuse_env=False,
               mark=None, env_key=sorted(ek))
    if not env_m:
        out["why"] = "no_env"
        return out
    if v not in by or len(by[v]) < 2:
        out["why"] = "thin_v"
        return out

    if ek in W_env:
        addr, mark = W_env[ek]
        out["mark"] = mark
        out["reuse_env"] = True
        out["pin"] = addr
        if mark == "DEAD":
            out["action"] = "SKIP_DEAD"
            out["why"] = "dead_mark"
            return out
        # LIVE: walk the mark, no neighbor re-search
        out["action"] = "PIN"
        fr2, why2 = hop2_stand(g, by, addr, env_m, rng)
        if why2:
            out["why"] = why2
            return out
        out["hop2"] = sorted(fr2)
        if held is not None and held not in fr2:
            out["action"] = "REFUSE"
            out["refuse"] = True
            out["why"] = "live_reuse_miss"
        return out

    slot, fr, why = stand_jacc(g, by, v, env_m, rng, exclude=exclude)
    if why:
        out["why"] = why
        return out
    out["stand"] = int(slot)
    out["frame"] = sorted(fr)
    if held is not None and held in fr:
        out["action"] = "READ"
        out["why"] = "read_hit"
        return out
    cand = [c for c in fr if c in mid and c != v and c not in high]
    if not cand:
        out["action"] = "REFUSE"
        out["refuse"] = True
        out["why"] = "no_mid_neighbor"
        return out
    if len(cand) >= 3:
        out["action"] = "REFUSE"
        out["refuse"] = True
        out["why"] = f"many={len(cand)}"
        out["cand"] = cand
        return out
    addrs = cand if len(cand) == 1 else rank_cand(g, by, cand, env_m, rng)
    if held is None:
        for addr in addrs:
            fr2, why2 = hop2_stand(g, by, addr, env_m, rng)
            if not why2:
                out["action"] = "STEP"
                out["pin"] = addr
                out["hop2"] = sorted(fr2)
                out["why"] = "unscored_read"
                return out
        out["action"] = "REFUSE"
        out["refuse"] = True
        out["why"] = "no_readable_addr"
        return out
    hit = False
    addr = addrs[0]
    fr2 = None
    for addr in addrs:
        fr2, why2 = hop2_stand(g, by, addr, env_m, rng)
        if why2:
            continue
        if held in fr2:
            hit = True
            break
    if not hit:
        out["action"] = "REFUSE"
        out["refuse"] = True
        out["why"] = "dead1"
        W_env[ek] = (addrs[0], "DEAD")
        out["mark"] = "DEAD"
        return out
    W_stood[slot] = addr
    out["action"] = "PIN"
    out["pin"] = addr
    out["wrote"] = True
    out["hop2"] = sorted(fr2) if fr2 else []
    out["mark"] = "LIVE"
    W_env[ek] = (addr, "LIVE")
    return out


def demo_smoke(tape, args):
    g, by, mid, high, rng = tape["g"], tape["by"], tape["mid"], tape["high"], tape["rng"]
    W_stood, W_env = {}, {}
    vs = [v for v in mid if len(by[v]) >= 8]
    rng.shuffle(vs)
    print(LAW)
    print(f"tape {tape['path']}  {tape['kind']}  n={g['n']}  lines={tape['n_lines']}  "
          f"mid={len(mid)}", flush=True)
    n_pin = n_ref = n_skip = n_live = n_dead = 0
    for v in vs[: args.n_demo]:
        sl = list(by[v])
        rng.shuffle(sl)
        probe = sl[0]
        fr0 = list(comps(g, probe, v))
        if len(fr0) < 2:
            n_skip += 1
            continue
        rng.shuffle(fr0)
        held, env = fr0[0], set(fr0[1:])
        env_m = (env & mid) - high or env - high
        r = place_walk(g, by, v, env_m, mid, high, rng, exclude=probe,
                       W_stood=W_stood, W_env=W_env, held=held)
        if r["action"] == "PIN":
            n_pin += 1
            hit = held in set(r.get("hop2") or ())
            print(f"  PIN  v={v!r} pin={r['pin']!r} mark={r['mark']} "
                  f"hop2_hit={hit}")
            if r["wrote"]:
                r2 = place_walk(g, by, v, env_m, mid, high, rng,
                                W_stood=W_stood, W_env=W_env, held=held)
                if r2.get("mark") == "LIVE" and r2.get("reuse_env"):
                    n_live += 1
                    print(f"  LIVE reuse pin={r2['pin']!r}")
                elif r2.get("action") == "SKIP_DEAD":
                    n_dead += 1
                    print(f"  DEAD skip  pin={r2['pin']!r} (not walked)")
        elif r["action"] == "SKIP_DEAD":
            n_dead += 1
            print(f"  DEAD skip v={v!r} pin={r['pin']!r}")
        elif r["action"] == "REFUSE":
            n_ref += 1
            print(f"  REFUSE v={v!r} why={r['why']}")
        else:
            n_skip += 1
            print(f"  SKIP  v={v!r} why={r['why']}")
    print(f"demo pin={n_pin} live_reuse={n_live} dead_skip={n_dead} "
          f"refuse={n_ref} skip={n_skip}  |W_env|={len(W_env)}")


def demo_ask(tape, ask: str, hide: str):
    g, by, mid, high, rng = tape["g"], tape["by"], tape["mid"], tape["high"], tape["rng"]
    words = toks(ask)
    hide_v = tape.get("alias", hide)
    hide_ask = hide
    low = {w.lower(): w for w in words}
    if hide.lower() in low:
        hide_ask = low[hide.lower()]
    env_m = env_of(words, hide_ask, mid, high)
    print(LAW)
    print(f"ask  {ask!r}")
    print(f"hide {hide!r}  on_tape={hide_v!r}  env_m={sorted(env_m)}")
    if hide_v not in by:
        print(f"REFUSE: {hide_v!r} never on this tape window")
        return
    W_stood, W_env = {}, {}
    r1 = place_walk(g, by, hide_v, env_m, mid, high, rng,
                    W_stood=W_stood, W_env=W_env)
    print(f"1 action={r1['action']} mark={r1['mark']} pin={r1['pin']!r} "
          f"hop2={r1['hop2']}")
    r2 = place_walk(g, by, hide_v, env_m, mid, high, rng,
                    W_stood=W_stood, W_env=W_env)
    print(f"2 action={r2['action']} mark={r2['mark']} reuse={r2['reuse_env']} "
          f"pin={r2['pin']!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="PLACE-WALK + W[env] LIVE/DEAD")
    ap.add_argument("--corpus", default="")
    ap.add_argument("--ask", default="", help="query sentence")
    ap.add_argument("--hide", default="", help="token to recover (v)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bytes", type=int, default=40_000_000)
    ap.add_argument("--window-lines", type=int, default=800)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--n-demo", type=int, default=12)
    args = ap.parse_args()
    tape = load_tape(args, need_token=args.hide or None)
    if args.ask:
        if not args.hide:
            raise SystemExit("--ask needs --hide <token>")
        demo_ask(tape, args.ask, args.hide)
    else:
        demo_smoke(tape, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
