import glob
import json
import statistics as st


def mean_keys(pat, keys):
    rows = []
    for p in sorted(glob.glob(pat)):
        r = json.load(open(p, encoding="utf-8"))["reach"]["held_out"]
        rows.append({k: r.get(k) for k in keys})
    out = {}
    for k in keys:
        xs = [row[k] for row in rows if row[k] == row[k]]
        out[k] = sum(xs) / len(xs) if xs else float("nan")
    return out, len(rows)


keys = ("reachable_rate", "hit_rate", "hit_of_walk_only", "step_rate")
for name, pat in (
    ("352deep", "results/stage289_decision_352deep_s*.json"),
    ("353twoway", "results/stage289_decision_353twoway_s*.json"),
    ("353margin", "results/stage289_decision_353margin_s*.json"),
):
    m, n = mean_keys(pat, keys)
    print(
        f"{name} ({n}): reachable {m['reachable_rate']:.3f}  "
        f"hit {m['hit_rate']:.3f}  wo_hit {m['hit_of_walk_only']:.3f}  "
        f"step {m['step_rate']:.3f}"
    )
