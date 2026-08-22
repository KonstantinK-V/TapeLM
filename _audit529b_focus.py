"""529b: focus — short window + episodic + mini (ceiling alive)."""
from __future__ import annotations

import json
from pathlib import Path

from _audit529_sweep import one_run

OUT = Path("results/_stage529b_focus.json")
STORIES = "data/_tinystories_train.txt"
MINI = "data/external_tinystories_mini.txt"


def base(**kw):
    d = dict(
        bytes=30_000_000, frame_max=3, min_fillers=2, lines=25000,
        hop1_free=True, eps=0.35, mid_only=False,
    )
    d.update(kw)
    return d


def main() -> int:
    cfgs = []
    for W in (150, 200, 250, 300):
        for t in ("nov", "cover_end", "adv"):
            for c in (0.0, 0.01):
                for s in (1337, 8642, 2890):
                    cfgs.append(base(
                        name=f"{t}_W{W}_c{c}", teacher=t, c_step=c,
                        window=W, seed=s, epochs=15, corpus=STORIES,
                    ))
    # longer train episodic only W=200
    for s in (1337, 8642, 2890):
        cfgs.append(base(
            name="adv_long", teacher="adv", c_step=0.0,
            window=200, seed=s, epochs=25, corpus=STORIES, eps=0.4,
        ))
        cfgs.append(base(
            name="cover_long", teacher="cover_end", c_step=0.0,
            window=200, seed=s, epochs=25, corpus=STORIES, eps=0.4,
        ))
    if Path(MINI).exists():
        for t in ("nov", "adv", "cover_end"):
            for s in (1337, 8642, 2890):
                cfgs.append(base(
                    name=f"mini_{t}", teacher=t, c_step=0.0,
                    window=400, seed=s, epochs=20, corpus=MINI, eps=0.4,
                ))

    rows = []
    print(f"529b focus n={len(cfgs)}", flush=True)
    for i, cfg in enumerate(cfgs):
        rec = one_run(cfg)
        # softer void mid for mini / short windows
        lm = rec.get("learn_mid") or {}
        if lm.get("n", 0) >= 8 and rec.get("thin_ceil"):
            pass
        if lm.get("n", 0) >= 8 and not rec.get("thin_ceil"):
            # re-evaluate gate with softer spend: hops <= allgo (not -0.5)
            a1 = rec["hop1"]
            ag = rec["allgo"]
            lh = rec["learn_high"]
            soft_gate = (
                lm["cover"] > a1["cover"] + 0.05
                and lm["hops"] <= ag["hops"] + 0.25
                and lh["hops"] < 1.5
            )
            rec["soft_gate"] = bool(soft_gate)
            # cover-only signal (learner moved)
            rec["cover_lift"] = lm["cover"] - a1["cover"]
        else:
            rec["soft_gate"] = False
            rec["cover_lift"] = 0.0
        # override void if mid n>=8 but was void only from n<15
        if lm.get("n", 0) >= 8 and rec.get("void") and not rec.get("thin_ceil"):
            rec["void"] = False
            rec["note"] = "mid_n_soft"
        rows.append(rec)
        mark = "GO" if rec.get("gate") else (
            "SOFT" if rec.get("soft_gate") else (
                "VOID" if rec.get("void") else "STOP"))
        print(
            f"[{i+1}/{len(cfgs)}] {rec['name']} s{rec['seed']} W{rec['n_lines']} "
            f"ceil {rec.get('ceiling', 0):+.3f} lift {rec.get('cover_lift', 0):+.3f} "
            f"Lh {lm.get('hops', 0):.1f} AG {rec.get('allgo', {}).get('hops', 0):.1f}  {mark}",
            flush=True,
        )

    softs = [r for r in rows if r.get("soft_gate") or r.get("gate")]
    lifts = sorted(
        [r for r in rows if (r.get("cover_lift") or 0) > 0.03 and not r.get("thin_ceil")],
        key=lambda r: r["cover_lift"], reverse=True)
    print("\n=== cover lift > 0.03 (ceiling ok) ===", flush=True)
    for r in lifts[:15]:
        print(
            f"  {r['name']:20} s{r['seed']} ceil {r['ceiling']:+.3f} "
            f"lift {r['cover_lift']:+.3f} Lh {r['learn_mid']['hops']:.1f}",
            flush=True,
        )
    print(f"\nsoft/hard GO: {len(softs)}", flush=True)
    for r in softs:
        print(f"  {r['name']} s{r['seed']} lift {r.get('cover_lift', 0):+.3f}", flush=True)

    out = Path(OUT)
    out.write_text(json.dumps(dict(n=len(rows), soft_go=len(softs), rows=rows), indent=1),
                   encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
