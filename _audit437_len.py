"""437: 436 think on several tape lengths. No new intelligence."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _audit436_constpin as M436

OUT = Path("results/_stage437_len.json")
WIKI = Path("data/_wikitext103_train.txt")
LENGTHS = (100, 400, 1600, 4000)


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

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    pool = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    by_len = {}
    print(f"seed {args.seed}  pool {len(pool)}")
    print(f"{'L':>6}  const_live  mixed_of_df2  const_hit  refuse_m  hop2  VOID  GATE")
    for L in lengths:
        if L > len(pool):
            print(f"{L:>6}  skip (pool {len(pool)})")
            continue
        s0 = 0 if L == len(pool) else rng.randrange(len(pool) - L + 1)
        lines = pool[s0:s0 + L]
        args.window_lines = L
        rep = M436.measure(lines, args, random.Random(args.seed + L))
        if rep is None:
            print(f"{L:>6}  no tape")
            continue
        void = rep["const_live"] <= 0.05
        gate = ((not void) and (rep["const_hit"] > 0.90)
                and (rep["n_mixed"] == 0 or rep["refuse_mixed"] > 0.90)
                and (rep["hop2_sees_pin"] == 1.0))
        rep["void"], rep["gate"], rep["window_lines"] = bool(void), bool(gate), L
        by_len[str(L)] = {k: rep[k] for k in (
            "n_df2", "n_const", "n_mixed", "const_live", "mixed_of_df2",
            "const_hit", "refuse_mixed", "hop2_sees_pin", "void", "gate")}
        print(f"{L:>6}  {rep['const_live']:.3f}       {rep['mixed_of_df2']:.3f}        "
              f"{rep['const_hit']:.2f}      {rep['refuse_mixed']:.2f}      "
              f"{rep['hop2_sees_pin']:.0f}     {int(void)}     {int(gate)}")
    lives = [by_len[str(L)]["const_live"] for L in lengths if str(L) in by_len]
    mixeds = [by_len[str(L)]["mixed_of_df2"] for L in lengths if str(L) in by_len]
    gates = [by_len[str(L)]["gate"] for L in lengths if str(L) in by_len]
    out_rep = dict(
        seed=args.seed, lengths=lengths, by_len=by_len,
        contract_holds=all(gates) if gates else False,
        const_live_span=(min(lives), max(lives)) if lives else None,
        mixed_span=(min(mixeds), max(mixeds)) if mixeds else None,
    )
    print(f"contract_holds {out_rep['contract_holds']}  "
          f"const_live {out_rep['const_live_span']}  mixed {out_rep['mixed_span']}")
    print("436 unchanged. mixed is still a counter.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = out_rep
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
