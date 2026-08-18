import glob
import json
import math


def z(b, c):
    return (b - c) / math.sqrt(b + c) if b + c else float("nan")


def one(p):
    d = json.load(open(p, encoding="utf-8"))
    r = d["reach"]["held_out"]
    wp = r.get("walk_only_paired") or {}
    op = r.get("own_paired") or {}
    rt = r.get("router") or {}
    return dict(
        seed=d["seed"],
        reachable=r.get("reachable_rate"),
        hit=r.get("hit_rate"),
        step=r.get("step_rate"),
        wo_hit=r.get("hit_of_walk_only"),
        own=r.get("own_hit_rate"),
        wo=(wp.get("mind_only"), wp.get("rival_only"), wp.get("n"), wp.get("mcnemar_z")),
        conf=(op.get("mind_only"), op.get("rival_only"), op.get("n"), op.get("mcnemar_z")),
        router=rt.get("mind_enrichment"),
        n_step=rt.get("n_stepped"),
    )


def dump(title, paths):
    print(title)
    rows = [one(p) for p in paths]
    b = c = n = cb = cc = cn = 0
    rs = []
    for x in rows:
        print(
            f"  s{x['seed']}  reach {x['reachable']:.3f}  hit {x['hit']:.3f}  "
            f"step {x['step']:.3f}  own {x['own']:.3f}  wo_hit {x['wo_hit']:.3f}"
        )
        print(
            f"         WALK-ONLY {x['wo'][0]}/{x['wo'][1]} of {x['wo'][2]} "
            f"z{x['wo'][3]:+.2f}  CONFIRM {x['conf'][0]}/{x['conf'][1]} of "
            f"{x['conf'][2]} z{x['conf'][3]:+.2f}  ROUTER {x['router']:.2f}x  "
            f"n_step {x['n_step']}"
        )
        b += x["wo"][0]
        c += x["wo"][1]
        n += x["wo"][2]
        cb += x["conf"][0]
        cc += x["conf"][1]
        cn += x["conf"][2]
        rs.append(x["router"])
    print(
        f"POOLED {len(rows)}: WALK-ONLY {b}/{c} of {n} z{z(b, c):+.2f}  "
        f"CONFIRM {cb}/{cc} of {cn} z{z(cb, cc):+.2f}  "
        f"ROUTER {sum(rs)/len(rs):.2f}x"
    )
    return [x["seed"] for x in rows]


seeds = dump(
    "=== 353twoway ===",
    sorted(glob.glob("results/stage289_decision_353twoway_s*.json")),
)
dump(
    "=== 352deep same seeds ===",
    [f"results/stage289_decision_352deep_s{s}.json" for s in seeds],
)
m = sorted(glob.glob("results/stage289_decision_353margin_s*.json"))
if m:
    dump("=== 353margin ===", m)
print("gate: CONFIRM z>=0 AND walk-only z>=+10; ROUTER>1 diagnostic")
