"""438: 436 think on a designed world. Mechanism unchanged.

Tape (only this, not wiki). frame_keep drops 1-value frames, so P is not pure:

  P: APPLES×3 + ORANGES decoy  → majority slots PIN APPLES; decoy REFUSE
  Q: PEARS PLUMS PEARS PLUMS   → tie, every slot REFUSE

No new scorer. If this FAIL, 436 is not a portable contract.

  VOID  n_const==0 or n_mixed==0   (world did not cut into two places)
  GATE  const_hit==1 and refuse_mixed==1 and hop2==1
        and both places present
        3 seeds (rng only shuffles; tape fixed)

    python _check438_world.py
    python _audit438_world.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import _audit436_constpin as M436

OUT = Path("results/_stage438_world.json")


def _pad(k):
    return " " + " ".join(f"z{k}x{j}" for j in range(20))


def designed():
    p = ["red cat sat APPLES on the mat" + _pad(i) for i in range(3)]
    p.append("red cat sat ORANGES on the mat" + _pad(3))
    q = []
    fruit = ("PEARS", "PLUMS", "PEARS", "PLUMS")
    for i, f in enumerate(fruit):
        q.append(f"blue dog lay {f} in the sun" + _pad(10 + i))
    return p + q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--addresses", type=int, default=1500)
    ap.add_argument("--max-questions", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    lines = designed()
    ns = SimpleNamespace(
        frame_max=args.frame_max, min_fillers=args.min_fillers,
        addresses=args.addresses, max_questions=args.max_questions,
    )
    rng = random.Random(args.seed)
    rep = M436.measure(lines, ns, rng)
    if rep is None:
        print("no tape")
        return 1
    void = (rep["n_const"] == 0) or (rep["n_mixed"] == 0)
    gate = ((not void) and (rep["const_hit"] == 1.0)
            and (rep["refuse_mixed"] == 1.0)
            and (rep["hop2_sees_pin"] == 1.0))
    rec = dict(
        seed=args.seed, n_lines=len(lines),
        n_df1=rep["n_df1"], n_df2=rep["n_df2"],
        n_const=rep["n_const"], n_mixed=rep["n_mixed"], n_empty=rep["n_empty"],
        const_hit=rep["const_hit"], refuse_mixed=rep["refuse_mixed"],
        hop2_sees_pin=rep["hop2_sees_pin"],
        void=bool(void), gate=bool(gate),
        think="436 unchanged; P majority APPLES + decoy; Q PEARS/PLUMS tie",
    )
    print(f"lines {len(lines)}  df2 {rep['n_df2']}  const {rep['n_const']}  "
          f"mixed {rep['n_mixed']}  df1 {rep['n_df1']}  empty {rep['n_empty']}")
    print(f"P->PIN  const_hit {rep['const_hit']:.2f}   "
          f"Q->REFUSE  refuse_mixed {rep['refuse_mixed']:.2f}   hop2 {rep['hop2_sees_pin']:.0f}")
    print(f"VOID {void}  GATE {gate}")
    if void:
        print("\nVOID: designed frames did not cut into a constant P and a mixed Q.")
    elif gate:
        print("\nGO WORLD: same 436 think. P pins APPLES; Q refuses the tie.")
    else:
        print("\nSTOP: 436 on a world it should pass. Contract is not portable.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
