"""Twenty lines out of a two-megabyte report - the numbers 299 is actually read by.

WHY THIS EXISTS. The stage writes everything it measured, which is right: a report that keeps
only what looked interesting on the day cannot be re-read when the question changes. But the
file is megabytes, and pasting it into a conversation burns the budget on braces. This prints
the void conditions first and the claim second, in the order HANDOFF 0 says to read them, and
nothing else.

    python _read299.py out/_stage289_decision_299e_both.json [more.json ...]
    python _read299.py out/_stage289_decision_299e_s*.json --held      seeds of one arm

POOLING SEEDS. Given several files it adds the walk-only counts and prints one McNemar over the
total - the honest way to buy power on a subset of ~104 questions per run: more seeds, same
mechanism, counts added, not a rerun kept because it read better. It pools only files that agree
on fingerprint, lookahead, refusal, import budget, places and candidate cap; mixed arms are
printed side by side and refused, because summing two different configurations produces a
real-looking z for a comparison nobody made. `--held` drops the train control.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


VOID = ("reachable_rate", "reachable_random", "reachable_wide", "own_hit_rate", "ceiling",
        "walk_only_rate", "step_rate")
CLAIM = ("hit_of_walk_only", "rival_of_walk_only", "hit_rate", "own_rival_hit_rate",
         "rival_hit_rate", "hit_of_own", "own_rival_of_own", "steps_on_walk_only")
SIZE = ("cand_places", "world_rows_own", "world_rows_candidate", "world_rows_expand", "world_rows_expand_when_stepped",
        "world_rows_expand_when_not", "step_vs_size_r")


def counts(r):
    """The walk-only contest as integers. Rates rounded to four places hide the denominator,
    and on 104 questions the denominator is the whole story."""
    wp = r.get("walk_only_paired")
    if wp:
        return wp["mind_only"], wp["rival_only"], wp["n"]
    n = int(round(r["walk_only_rate"] * r["n"]))
    return (int(round(r["hit_of_walk_only"] * n)), int(round(r["rival_of_walk_only"] * n)), n)


def z_of(b, c):
    return (b - c) / math.sqrt(b + c) if b + c else float("nan")


def one(path, arm):
    d = json.load(open(path, encoding="utf-8"))
    rc = d.get("reach")
    if not rc or not rc.get(arm):
        return None
    r = rc[arm]
    cfg = (rc.get("frame_fp"), rc.get("lookahead"), rc.get("no_refuse"),
           d.get("import_k"), rc.get("places"), rc.get("cands_cap"),
           rc.get("speak_batch"), rc.get("speak_weight"),
           rc.get("calib_batch"), rc.get("calib_weight"))
    print(f"\n{path}  [{arm}]  {d['wall_s']:.0f}s  fp={rc.get('frame_fp')} "
          f"lookahead={rc.get('lookahead')} import={rc.get('import')} "
          f"no_refuse={rc.get('no_refuse')} seed={d['seed']}"
          # 338: WHAT THE TAPE KEPT. Printed on the header line because every number below is
          # about a different tape when this is set, and a retention run read as an ordinary
          # one would look like a collapse rather than a smaller tape.
          + (f"  retain={d['retain']}/{d.get('retain_by')}" if d.get("retain") else "")
          # 341: a comparative speaking price is a different arm, not a different run, and
          # _read299 refuses to pool across arms - so it has to be visible on the header.
          + (f"  SPEAK-BATCH={rc['speak_batch']}x{rc.get('speak_weight')}"
             if rc.get("speak_batch") else "")
          # 389: the same rule for the gauge. A calibrated mind and an uncalibrated one measure
          # their raw scores on different rulers, so pooling them would average two gauges.
          + (f"  CALIB={rc['calib_batch']}x{rc.get('calib_weight')}"
             if rc.get("calib_batch") else "")
          # 377: THE ARM'S CHANNELS, on the header line. The 377copy dump could not be read
          # against its baseline because nothing printed here said whether --copy was even on:
          # `line=False` was visible and `copy=True` was not, so a run with the lever and a run
          # without it looked IDENTICAL at the top. A flag that changes the tape or the offer and
          # is invisible in the reader is a collided tag waiting to happen.
          + f"  min_fillers={d.get('min_fillers')}"
          + ("  CONNECT" if d.get("connect") else "")
          + (f"  COPY d={d.get('copy_d')}"
             + ("-BACKFILL" if d.get("copy_backfill") else "") if d.get("copy") else "")
          + ("  CHANNEL" if d.get("reach_channel") else "")
          + (f"  MOVES[{','.join(d.get('move_set') or [])}]" if d.get("moves") else "")
          + ("  OWN-IN-OFFER" if d.get("own_in_offer") else "")
          + ("  OWN-IMPORT" if d.get("own_import") else ""))
    print(f"  tape   resample_overlap {d['resample']['mean_overlap']:.3f}   "
          f"rare_nnz {d['rare_nonzero_rate']:.5f}   params {d['params']}")
    # cos_mean is NOT a channel health check: most row pairs in a world share an address and so
    # share a fingerprint exactly, which pins the average near 1 whatever the ink is doing. It
    # is printed because it was once read as evidence, and a number that misled once should stay
    # visible with its warning attached rather than quietly disappear.
    print(f"  cos    mean {d['cos_mean']:.3f} std {d['cos_std']:.3f}  "
          f"(dominated by same-address pairs - not a channel test)")
    print("  void   " + "  ".join(f"{k.replace('_rate', '')} {r[k]:.4f}"
                                  for k in VOID if k in r))
    print("  size   " + "  ".join(f"{k.replace('world_rows_', '')} {r[k]:.2f}"
                                  for k in SIZE if k in r and r[k] == r[k]))
    print("  claim  " + "  ".join(f"{k} {r[k]:.4f}" for k in CLAIM if k in r and r[k] == r[k]))
    # 385: HOW THE MIND SPENT ITS MOVE. One move on every question is a constant with extra
    # steps; the split is the only thing that says a searcher exists rather than a relabelling.
    if r.get("move_share"):
        mh = r.get("move_hit") or {}
        print("  MOVES  " + "  ".join(
            f"{m} {v:.4f}" + (f"/{mh[m]:.4f}" if m in mh else "")
            for m, v in r["move_share"].items())
            + "   (share/hit)")
    lp = r.get("line_only_paired")
    if lp and lp["n"]:
        print(f"  LINE-ONLY  mind {lp['mind_only']} / rival {lp['rival_only']} of {lp['n']}"
              f"   z {lp['mcnemar_z']:+.2f}   "
              f"line_reach {r.get('line_reach_rate', float('nan')):.4f}  "
              f"step_line {r.get('step_line_rate', float('nan')):.4f}")
    op = r.get("own_paired")
    if op and op["n"]:
        print(f"  CONFIRM    mind {op['mind_only']} / rival {op['rival_only']} of {op['n']}"
              f"   z {op['mcnemar_z']:+.2f}"
              f"{'   UNDERPOWERED' if math.sqrt(op['mind_only'] + op['rival_only']) <= 1.645 else ''}")
    rt = r.get("router")
    if rt and rt["mind_enrichment"] == rt["mind_enrichment"]:
        print(f"  ROUTER     stepped {rt['n_stepped']}  mind {rt['mind_enrichment']:.2f}x  "
              f"counting {rt['count_enrichment']:.2f}x  "
              f"|own| stepped {rt['n_own_when_stepped']:.2f} vs {rt['n_own_when_not']:.2f}")
        # THE SHARE ROUTER, same budget, and it is the last counting rule that could take the
        # route away: unlike |own| it reads the WALK. If it approaches the mind's enrichment,
        # the route is a count in a costume.
        if "share_enrichment" in rt:
            print(f"  ROUTER share  {rt['share_enrichment']:.2f}x   "
                  f"top_share stepped {rt['top_share_when_stepped']:.3f} vs "
                  f"{rt['top_share_when_not']:.3f}"
                  # 383: top_share reads ~1.000 everywhere, so the rule saturates on values that
                  # OWN their place. `ties` says whether that left the rival ARBITRARY (many
                  # tied, walk order decided) or merely BLUNT (one candidate, determinate).
                  + (f"   ties {r['count_rival_ties']:.2f}"
                     if "count_rival_ties" in r else ""))
    # ARRIVING AND CHOOSING, PRINTED APART. A higher hit_of_walk_only can be pure coverage:
    # an arm stepping on 30% of questions reaches far more walk_only truths than one stepping
    # on 1%, with an identical pick. `arrive` is the coverage, `PICK` is the contest among the
    # questions the mind actually walked on - the only place a claim about CHOOSING can live.
    wp = r.get("walk_only_pick") or {}
    if wp.get("n"):
        print(f"  PICK       stepped {wp['n']} of walk-only   "
              f"mind {wp['mind']} ({wp['hit_rate']:.4f}) / rival {wp['rival']} "
              f"({wp['rival_rate']:.4f})   paired {wp['mind_only']}/{wp['rival_only']}   "
              f"z {wp['mcnemar_z']:+.2f}   arrive {r.get('walk_only_arrive', float('nan')):.4f}")
        # THE LINE THAT DECIDES WHETHER THE PICK IS A FACULTY. Same candidates, same walk,
        # exact counts - the strongest counting rival the walk allows. If its rate approaches
        # the mind's, there was nothing to choose between and the claim is routing alone.
        if "count_rival" in wp:
            print(f"  PICK vs COUNT   count-rival {wp['count_rival']} "
                  f"({wp['count_rival_rate']:.4f})   paired "
                  f"{wp['vs_count_mind_only']}/{wp['vs_count_rival_only']}   "
                  f"z {wp['vs_count_z']:+.2f}")
    if "deep_rate" in r:
        # VOID FIRST: a depth that is never taken measures nothing, exactly as step_rate 0 did
        # in 299b. Read deep_rate before hit_of_deep, always.
        print(f"  DEPTH      deep {r['deep_rate']:.4f}   hit_of_deep "
              f"{r.get('hit_of_deep', float('nan')):.4f}   "
              f"hit_of_depth1 {r.get('hit_of_depth1', float('nan')):.4f}")
    bi = r.get("bisect") or {}
    if bi.get("n"):
        # PAIRED against the mind's own flat argmax, on the questions it stepped on. The point
        # is not that bisection wins - it is whether log2(c) comparisons of UNFILLED halves
        # carry what c comparisons of completed worlds carry. A tie is the interesting result.
        print(f"  BISECT     {bi['bisect_right']}/{bi['n']} vs flat {bi['flat_right']}/{bi['n']}"
              f"   paired {bi['bisect_only']}/{bi['flat_only']}   z {bi['mcnemar_z']:+.2f}"
              f"   splits {bi['splits_mean']:.1f}")
    om = r.get("other_mind")
    if om:
        # 336: THIS RUN'S MIND AGAINST A SECOND SAVED ONE, paired inside the run. A SMALL z IS
        # THE CLAIM HERE, so `underpowered` is printed next to it - the two look identical and
        # mean opposite things.
        for sub in ("all", "walk_only", "confirm"):
            m = om[sub]
            if not m["n"]:
                continue
            print(f"  OTHER {sub:<10} this {m['this']} / other {m['other']} of {m['n']}   "
                  f"paired {m['this_only']}/{m['other_only']}   z {m['mcnemar_z']:+.2f}"
                  + ("   IDENTICAL - right on exactly the same questions"
                     if m.get("identical") else
                     "   UNDERPOWERED" if m["underpowered"] else ""))
        print(f"  OTHER step   this {om['step_rate']:.4f} vs other {om['other_step_rate']:.4f}")
    qr = r.get("question_rank") or {}
    for tgt in ("answerable", "ceiling", "right"):
        t = qr.get(tgt)
        if not t:
            continue
        # 337: CAN THE MIND TELL WHICH QUESTIONS THE TAPE CAN ANSWER. AUC first because it is
        # the pre-registered statistic and needs no cut; precision@k is the readable form, and
        # base is what a coin scores. The mind's number means nothing on its own - the two
        # counting rivals on the same line are the whole reading.
        def a(nm):
            return t[nm]["auc"]

        def pk(nm):
            return t[nm]["prec"][-1]
        print(f"  RANK {tgt:<10} base {t['base_rate']:.4f}   AUC mind {a('mind_margin'):.4f} "
              f"(score {a('mind_score'):.4f})  |own| {a('count_n_own'):.4f}  "
              f"share {a('count_top_share'):.4f}"
              f"   p@{t['k'][-1]} mind {pk('mind_margin'):.4f} vs "
              f"{max(pk('count_n_own'), pk('count_top_share')):.4f}")
        # 389: THE GAUGE, pulled out of the parenthesis it hid in for eighty steps. `margin` is
        # gauge invariant and has always been the headline; `score` is the RAW value, and its
        # AUC across questions is the only column that says whether Phi's number means anything
        # outside the one question it was computed on. A free per-question offset predicts 0.50
        # here no matter how good the mind is - so this is the number --calib-batch must move,
        # and the two counts beside it are the same rivals the margin has to beat.
        gap = a('mind_score') - 0.5
        print(f"       GAUGE    raw-score AUC {a('mind_score'):.4f}  ({gap:+.4f} vs a coin)"
              f"   margin {a('mind_margin'):.4f}   rivals {a('count_n_own'):.4f}/"
              f"{a('count_top_share'):.4f}"
              f"   {'FREE' if abs(gap) < 0.03 else 'TIED'}")
    mb = r.get("margin_by_stage")
    if mb:
        # 341's post-mortem. `right` vs `wrong` is what the margin is FOR (calibration, AUC
        # 0.866). `stayed` vs `stepped` is the confound: if stepping carries a bigger margin,
        # then training the margin trains the route, which is what 341 appears to have done.
        print(f"  MARGIN     right {mb['by_right']['right']:.3f} vs wrong "
              f"{mb['by_right']['wrong']:.3f}   |   stayed {mb['stayed']:.3f} (n "
              f"{mb['n_stayed']}) vs stepped {mb['stepped']:.3f} (n {mb['n_stepped']})")
        print(f"             within-stage: stayed {mb['stayed_right']:.3f}/"
              f"{mb['stayed_wrong']:.3f}   stepped {mb['stepped_right']:.3f}/"
              f"{mb['stepped_wrong']:.3f}   (right/wrong)")
    for gname, g in (("GATE", r.get("gate")), ("GATE-WO", r.get("gate_walk_only"))):
        if not g:
            continue
        # 337 USED: the mind answers only its top fraction and refuses the rest, against the
        # same two counts at EXACTLY the same coverage. `yield` is printed next to precision
        # because a gate can always buy precision by answering less, and `payoff` is the only
        # column that says whether refusing was worth it at all.
        # VOID CONDITION FIRST, as everywhere else in this reader. `always_silent` is what
        # refusing every question pays, and on a tape where most holes are unanswerable it is
        # high - so a gate's payoff is only ever readable as `gain` against it. The ungated
        # payoff is NOT the reference and reading it as one made a losing column look winning.
        print(f"  {gname:<9}  n {g['n']}  ungated hit {g['ungated_hit_rate']:.4f}  "
              f"payoff {g['ungated_payoff']:+.4f}   "
              f"ALWAYS-SILENT PAYS {g.get('always_silent', float('nan')):+.4f}")
        for fr in g["fractions"]:
            d = g[f"{fr:.2f}"]
            m, a, s = d["mind"], d["count_n_own"], d["count_top_share"]
            r0 = d.get("random") or {}
            co = d.get("composition") or {}
            print(f"    top {fr:>4.0%} k={d['k']:<5} mind {m['precision']:.4f} "
                  f"({m['yield']} right, gain {m.get('gain', float('nan')):+.4f})   "
                  f"|own| {a['precision']:.4f} ({a['yield']})   "
                  f"share {s['precision']:.4f} ({s['yield']})   "
                  f"rand {r0.get('precision', float('nan')):.4f} ({r0.get('yield', 0)})   "
                  f"z vs |own| {d['vs_count_n_own']['mcnemar_z']:+.2f}  "
                  f"vs share {d['vs_count_top_share']['mcnemar_z']:+.2f}"
                  # WHERE THE KEPT ANSWERS CAME FROM. 83-90% of everything the mind gets right
                  # on this tape is a confirm, which a lookup answers - so a gate read without
                  # this split is mostly a report about the index.
                  + (f"   [kept {co.get('confirm', 0)}c/{co.get('walk_only', 0)}w, right "
                     f"{co.get('right_confirm', 0)}c/{co.get('right_walk_only', 0)}w]"
                     if co and gname == "GATE" else ""))
    b, c, n = counts(r)
    print(f"  WALK-ONLY  mind {b} / rival {c} of {n}   z {z_of(b, c):+.2f}"
          f"{'   UNDERPOWERED' if math.sqrt(b + c) <= 1.645 else ''}")
    return b, c, n, cfg


def main(argv) -> int:
    files = [a for a in argv if not a.startswith("--")]
    if not files:
        print(__doc__)
        return 1
    # out/ and results/ hold the SAME report under the same name, so a glob over both doubles
    # every pooled count and prints "8 runs" for 4 seeds. Pooling is a sum of counts, so a
    # duplicate is not a harmless repeat: it halves the honest z. Dedupe by file NAME.
    seen, uniq = set(), []
    for f in files:
        nm = Path(f).name.lstrip("_")
        if nm in seen:
            continue
        seen.add(nm)
        uniq.append(f)
    if len(uniq) != len(files):
        print(f"note: dropped {len(files) - len(uniq)} duplicate report name(s) - out/ and "
              f"results/ hold the same file")
        files = uniq
    arms = ["held_out"] if "--held" in argv else ["held_out", "train_control"]
    tot = {a: [0, 0, 0] for a in arms}
    con = {a: [0, 0, 0] for a in arms}
    # 336: the native-against-transplanted contest, pooled. Its power is in the pooling - a
    # single seed's walk-only subset carries almost no discordant pairs.
    oth = {a: {"all": [0, 0, 0], "walk_only": [0, 0, 0]} for a in arms}
    gate = {a: {} for a in arms}      # fraction -> [k, mind hits, share hits, random hits]
    cfgs = {a: set() for a in arms}
    for f in files:
        for a in arms:
            got = one(f, a)
            if got:
                for i in range(3):
                    tot[a][i] += got[i]
                cfgs[a].add(got[3])
                r_a = json.load(open(f, encoding="utf-8"))["reach"][a]
                op = r_a.get("own_paired")
                if op:
                    con[a][0] += op["mind_only"]
                    con[a][1] += op["rival_only"]
                    con[a][2] += op["n"]
                # THE WALK-ONLY GATE IS ~100 ROWS PER SEED, so at 5% coverage k is five and one
                # question moves the precision by 0.2. It is pooled or it is not read.
                gw = r_a.get("gate_walk_only")
                if gw:
                    for fr in gw["fractions"]:
                        e = gate[a].setdefault(fr, [0, 0, 0, 0])
                        e[0] += gw[f"{fr:.2f}"]["k"]
                        e[1] += gw[f"{fr:.2f}"]["mind"]["yield"]
                        e[2] += gw[f"{fr:.2f}"]["count_top_share"]["yield"]
                        e[3] += (gw[f"{fr:.2f}"].get("random") or {}).get("yield", 0)
                om = r_a.get("other_mind")
                if om:
                    for sub in ("all", "walk_only"):
                        oth[a][sub][0] += om[sub]["this_only"]
                        oth[a][sub][1] += om[sub]["other_only"]
                        oth[a][sub][2] += om[sub]["n"]
    if len(files) > 1:
        for a in arms:
            b, c, n = tot[a]
            # POOLING IS FOR SEEDS, NOT FOR ARMS. Adding the walk-only counts of two different
            # configurations answers no question anyone asked: the first version of this script
            # summed 299d and 299e and printed z 0.00, which is a real-looking number for a
            # comparison that does not exist. Different arms are reported side by side and never
            # added.
            if len(cfgs[a]) > 1:
                print(f"\nNOT POOLED ({a}): {len(cfgs[a])} different arms among these files. "
                      f"Pool seeds of ONE configuration; arms are compared, not summed.")
                continue
            print(f"\nPOOLED {a} over {len(files)} runs: mind {b} / rival {c} of {n}   "
                  f"z {z_of(b, c):+.2f}"
                  f"{'   UNDERPOWERED' if math.sqrt(b + c) <= 1.645 else ''}")
            cb, cc, cn = con[a]
            if cn:
                print(f"POOLED CONFIRM {a} over {len(files)} runs: mind {cb} / rival {cc} "
                      f"of {cn}   z {z_of(cb, cc):+.2f}")
            for fr, (k, mh, sh, rh) in sorted(gate[a].items()):
                if not k:
                    continue
                # THE HALF THAT MATTERS, pooled: speaking only where an index cannot answer.
                print(f"POOLED GATE-WO {a} top {fr:>4.0%}: k {k}   mind {mh} ({mh / k:.4f})   "
                      f"share {sh} ({sh / k:.4f})   random {rh} ({rh / k:.4f})")
            for sub in ("all", "walk_only"):
                ob, oc, on = oth[a][sub]
                if not on:
                    continue
                # A SMALL z IS THE CLAIM in 336, so the discordant TOTAL is what makes it
                # readable: 73/71 of 8000 is a powered null, 1/0 of 402 is no measurement.
                print(f"POOLED OTHER {sub} {a} over {len(files)} runs: this {ob} / other {oc}"
                      f"   z {z_of(ob, oc):+.2f}   ({ob + oc} discordant of {on})"
                      f"{'   TOO FEW DISCORDANT TO READ' if ob + oc < 10 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
