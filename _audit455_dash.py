"""455: mixed dashboard. One pick_cost, no per-family branch."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from _audit453_depth import train_mixed, world_d2
from _audit454_cost import eval_fam, world_both, world_d1, world_d4, world_stop

OUT = Path("results/_stage455_dash.json")
FAM = {
    "D1": world_d1,
    "D2": world_d2,
    "BOTH": world_both,
    "D4": world_d4,
    "STOP": world_stop,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--test", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rate = train_mixed(rng, args.train)
    reps = {k: eval_fam(rng, fn, args.test, rate) for k, fn in FAM.items()}
    void = any(reps[k]["ep"] < 5 for k in FAM)
    extra = {k: reps[k]["greedy_hops"] - reps[k]["mean_hops"] for k in FAM}
    gate = ((not void)
            and (reps["D1"]["pin"] == 1.0) and (reps["D1"]["mean_hops"] == 1.0)
            and (reps["D2"]["pin"] == 1.0) and (reps["D2"]["mean_hops"] == 2.0)
            and (reps["BOTH"]["pin"] == 1.0) and (reps["BOTH"]["mean_hops"] == 2.0)
            and (reps["D4"]["pin"] == 1.0) and (reps["D4"]["mean_hops"] == 4.0)
            and (reps["STOP"]["refuse"] == 1.0) and (reps["STOP"]["pin"] == 0.0)
            and (extra["BOTH"] > 0) and (extra["D4"] > 0))
    rec = dict(seed=args.seed, void=bool(void), gate=bool(gate),
               extra={k: round(extra[k], 2) for k in FAM}, **reps)
    print(f"{'fam':4} {'ep':>3} {'pin':>5} {'ref':>5} {'hops':>5} {'gdy':>5} {'extra':>6}")
    for k in FAM:
        r = reps[k]
        print(f" {k:4} {r['ep']:3} {r['pin']:5.2f} {r['refuse']:5.2f} "
              f"{r['mean_hops']:5.1f} {r['greedy_hops']:5.1f} {extra[k]:6.1f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: a family had <5 test eps.")
    elif gate:
        print("\nGO DASH: one pick_cost covers D1/D2/BOTH/D4/STOP.")
    else:
        print("\nSTOP: a family failed under the shared chooser.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
