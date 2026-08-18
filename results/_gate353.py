"""Prereg gate for 353: CONFIRM pooled z >= 0 AND walk-only z >= +10."""
from __future__ import annotations

import glob
import json
import math


def z_of(b, c):
    return (b - c) / math.sqrt(b + c) if b + c else float("nan")


def pool(patterns):
    b = c = n = 0
    cb = cc = cn = 0
    rout = []
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        return None
    for p in files:
        d = json.load(open(p, encoding="utf-8"))
        r = d["reach"]["held_out"]
        wp = r.get("walk_only_paired") or {}
        if wp:
            b += wp["mind_only"]
            c += wp["rival_only"]
            n += wp["n"]
        else:
            nn = int(round(r["walk_only_rate"] * r["n"]))
            b += int(round(r["hit_of_walk_only"] * nn))
            c += int(round(r["rival_of_walk_only"] * nn))
            n += nn
        op = r.get("own_paired") or {}
        if op:
            cb += op["mind_only"]
            cc += op["rival_only"]
            cn += op["n"]
        rt = r.get("router") or {}
        if rt.get("mind_enrichment") == rt.get("mind_enrichment"):
            rout.append(rt["mind_enrichment"])
    return {
        "wo": (b, c, n, z_of(b, c)),
        "conf": (cb, cc, cn, z_of(cb, cc)),
        "router": (sum(rout) / len(rout)) if rout else float("nan"),
        "nfiles": len(files),
    }


def main() -> None:
    arms = [
        ("352deep", ["results/stage289_decision_352deep_s*.json"]),
        ("353twoway", ["out/_stage289_decision_353twoway_s*.json",
                       "results/stage289_decision_353twoway_s*.json"]),
        ("353margin", ["out/_stage289_decision_353margin_s*.json",
                       "results/stage289_decision_353margin_s*.json"]),
    ]
    for name, pats in arms:
        p = pool(pats)
        if not p:
            print(f"{name}: missing")
            continue
        wo, cf = p["wo"], p["conf"]
        ok = cf[3] >= 0 and wo[3] >= 10
        print(
            f"{name} ({p['nfiles']} files): "
            f"WALK-ONLY {wo[0]}/{wo[1]} of {wo[2]} z{wo[3]:+.2f}  "
            f"CONFIRM {cf[0]}/{cf[1]} of {cf[2]} z{cf[3]:+.2f}  "
            f"ROUTER {p['router']:.2f}x  -> {'PASS' if ok else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
