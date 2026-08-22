"""439: both axes, same 436 think. No new mind."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import _audit436_constpin as M436
import _audit438_world as M438

OUT = Path("results/_stage439_axes.json")
WIKI = Path("data/_wikitext103_train.txt")
LENGTHS = (100, 400, 1600, 4000)


def _ns(args):
    return SimpleNamespace(
        frame_max=args.frame_max, min_fillers=args.min_fillers,
        addresses=args.addresses, max_questions=args.max_questions,
    )


def axis_a(args, rng):
    rep = M436.measure(M438.designed(), _ns(args), rng)
    if rep is None:
        return dict(void=True, gate=False)
    void = (rep["n_const"] == 0) or (rep["n_mixed"] == 0)
    gate = ((not void) and (rep["const_hit"] == 1.0)
            and (rep["refuse_mixed"] == 1.0)
            and (rep["hop2_sees_pin"] == 1.0))
    return dict(
        n_const=rep["n_const"], n_mixed=rep["n_mixed"], n_df1=rep["n_df1"],
        const_hit=rep["const_hit"], refuse_mixed=rep["refuse_mixed"],
        hop2_sees_pin=rep["hop2_sees_pin"], void=bool(void), gate=bool(gate),
    )


def axis_b(args, pool, rng0):
    by_len = {}
    for L in args.lengths:
        if L > len(pool):
            continue
        s0 = 0 if L == len(pool) else rng0.randrange(len(pool) - L + 1)
        rep = M436.measure(pool[s0:s0 + L], _ns(args), random.Random(args.seed + L))
        if rep is None:
            continue
        void = rep["const_live"] <= 0.05
        gate = ((not void) and (rep["const_hit"] > 0.90)
                and (rep["n_mixed"] == 0 or rep["refuse_mixed"] > 0.90)
                and (rep["hop2_sees_pin"] == 1.0))
        by_len[str(L)] = dict(
            const_live=rep["const_live"], mixed_of_df2=rep["mixed_of_df2"],
            n_const=rep["n_const"], n_mixed=rep["n_mixed"],
            const_hit=rep["const_hit"], refuse_mixed=rep["refuse_mixed"],
            hop2_sees_pin=rep["hop2_sees_pin"], void=bool(void), gate=bool(gate),
        )
    lives = [v["const_live"] for v in by_len.values()]
    mixeds = [v["mixed_of_df2"] for v in by_len.values()]
    return dict(
        by_len=by_len,
        contract_holds=all(v["gate"] for v in by_len.values()) if by_len else False,
        const_live_span=(min(lives), max(lives)) if lives else None,
        mixed_span=(min(mixeds), max(mixeds)) if mixeds else None,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=3)
    ap.add_argument("--min-fillers", type=int, default=2)
    ap.add_argument("--addresses", type=int, default=1500)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--max-questions", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default=str(WIKI))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--lengths", default=",".join(str(x) for x in LENGTHS))
    args = ap.parse_args()
    args.lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

    a = axis_a(args, random.Random(args.seed))
    print(f"A designed  const {a.get('n_const')} mixed {a.get('n_mixed')}  "
          f"hit {a.get('const_hit')} refuse {a.get('refuse_mixed')}  "
          f"hop2 {a.get('hop2_sees_pin')}  VOID {a['void']} GATE {a['gate']}")

    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    pool = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    b = axis_b(args, pool, random.Random(args.seed))
    print(f"{'L':>6}  const_live  mixed_of_df2  GATE")
    for L, row in b["by_len"].items():
        print(f"{L:>6}  {row['const_live']:.3f}       {row['mixed_of_df2']:.3f}        {int(row['gate'])}")
    print(f"B wiki  contract_holds {b['contract_holds']}  "
          f"const_live {b['const_live_span']}  mixed {b['mixed_span']}")

    both = bool(a["gate"] and b["contract_holds"])
    rec = dict(seed=args.seed, A=a, B=b, both_hold=both)
    print(f"BOTH {both}")
    if both:
        print("\nGO AXES: designed pins/refuses; wiki contract holds across L. "
              "Still 436 think. Composition is not this file.")
    else:
        print("\nSTOP: one axis failed. Do not compose.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rec
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
