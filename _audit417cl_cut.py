"""CUT THE DECOY, NOT FILL THE HOLE. The ceiling of a different ACTION, torch-free.

415 says filling on wiki is the frame's majority again - the count wins. So the action changes:
the window holds ONE WORD TOO MANY and the job is to say WHICH. The alphabet is unchanged (a slot,
not a letter), the tape still holds the words, and the mind only points at what to cut. The label
is ours and exact, so every slot of the window is a candidate and the teacher is dense by
construction - no hidden token, no offer of eight, no GATE-WO.

THE DECOY MUST BE HARD OR THE MEASUREMENT MEANS NOTHING. A word drawn at random from the corpus is
found by FREQUENCY alone, and this project has been eaten by that prior three times (399, 400,
411). Here the decoy is a filler taken from a place that SHARES ONE HALF of the position's frame
and not the other: it fits half the window, is locally plausible and is frequency-ordinary. The
EASY decoy - a plain random token - is run beside it as the check on the construction itself: if
the two are equally findable, the injection is not hard and nothing below is about coherence.

THE RIVALS, DECLARED BEFORE THE RUN - all counts, no Phi anywhere in this file:

    rare       cut the least frequent word in the window
    miss       cut the word that is NOT among the fillers its own frame holds on the tape
    local      cut the word standing at fewest places whose frame shares a half with this one
    random     the floor, 1 / positions

  VOID / READING, both declared:
    SIGNAL   best count - floor > 0.05, or the arena is empty and cutting says nothing.
    ROOM     best count < 0.90. Above that the tape ANSWERS IT BY LOOKUP and Phi has nothing to
             learn here - which is 38.3's law, and the reason a "yes" that is too big is a no.
    HARD     hard-decoy accuracy must be materially below easy-decoy accuracy, or the decoy is
             not hard.

Both must hold for the direction to be worth a mind. This file cannot say whether Phi beats the
counts - only whether there is a gap for it to beat them IN.

    python _audit417cl_cut.py --seed 1337
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import _tape_frames as tframes


def build(lines, frame_max, min_fillers):
    keep, toks, owner = tframes.frame_keep(lines, frame_max, min_fillers)
    fills = defaultdict(set)          # (left,right) -> the words that stand there
    by_half = defaultdict(set)        # a half -> the frames carrying it
    at = Counter()                    # word -> how many places hold it
    freq = Counter(toks)
    for (w, left, right), pos in keep:
        key = (left, right)
        for i in pos:
            fills[key].add(toks[i])
        by_half[("L", left)].add(key)
        by_half[("R", right)].add(key)
    for key, ws in fills.items():
        for x in ws:
            at[x] += 1
    by_band = defaultdict(list)          # 373's frequency bands, for the matched decoy
    for x, n in freq.items():
        by_band[n.bit_length()].append(x)
    return {"fills": dict(fills), "by_half": dict(by_half), "at": at, "freq": freq,
            "by_band": dict(by_band)}


def frame_at(words, i):
    return ((words[i - 1],), (words[i + 1],))


def hard_decoy(T, words, i, rng):
    """A filler from a place that shares ONE half of this frame and not the other - fits half the
    window, ordinary in frequency, and therefore not findable by rarity."""
    left, right = frame_at(words, i)
    # AND IT MUST NOT BE A LEGAL WORD FOR THIS FRAME. A candidate that also stands at
    # (left, right) is not a decoy at all - injecting it would make the "extra" word one the tape
    # says belongs there, and every rival would be scored against a wrong label.
    legal = T["fills"].get((left, right), set()) | {words[i]}
    cands = []
    for tag, half in (("L", left), ("R", right)):
        for key in T["by_half"].get((tag, half), ()):
            if key != (left, right):
                cands.extend(x for x in T["fills"][key] if x not in legal)
    if not cands:
        return None
    return cands[rng.randrange(len(cands))]


def matched_decoy(T, words, i, rng):
    """THE MIDDLE SETTING, and the one that should have been primary.

    Frequency-matched to the word it replaces - so the `rare` rival has no grip, which was the
    whole reason a random decoy is useless - and fitting NEITHER half of the frame, so the
    evidence on both sides of the window is left intact. The `hard` decoy fits one half, which at
    w=1 leaves exactly one neighbour to decide by: not hard, unanswerable by construction. That
    was a calibration error, and this is the setting between the two.
    """
    left, right = frame_at(words, i)
    true = words[i]
    # ONLY TWO CONDITIONS, AND NEITHER IS A RIVAL'S CRITERION. The first version also excluded
    # every word standing at a HALF-SHARING place - which is exactly what `local` counts, so the
    # decoy was defined as "the word `local` will flag" and `local` read 0.958 on plumbing. The
    # construction must not be the signal; that is the fault of section 27 and of 400's Return
    # bit, and it is the third time this shape has appeared.
    bad = set(T["fills"].get((left, right), ())) | {true}
    b = T["freq"].get(true, 1).bit_length()
    src, w = list(T["by_band"].get(b, ())), 0
    while len(src) < 16 and w < 20:
        w += 1
        src += T["by_band"].get(b - w, [])
        src += T["by_band"].get(b + w, [])
    src = [x for x in src if x not in bad]
    return src[rng.randrange(len(src))] if src else None


def scores(T, words, positions):
    """Every rival, per position. Higher = more suspicious."""
    rare, miss, local = [], [], []
    for i in positions:
        w = words[i]
        key = frame_at(words, i)
        rare.append(-float(T["freq"].get(w, 0)))
        miss.append(0.0 if w in T["fills"].get(key, ()) else 1.0)
        n = 0
        for tag, half in (("L", key[0]), ("R", key[1])):
            for k2 in T["by_half"].get((tag, half), ()):
                if w in T["fills"].get(k2, ()):
                    n += 1
        local.append(-float(n))
    return {"rare": rare, "miss": miss, "local": local}


def acc(s, target_ix):
    top = max(s)
    best = [j for j in range(len(s)) if s[j] == top]
    return (1.0 / len(best)) if target_ix in best else 0.0


def run(T, lines, args, rng):
    c = Counter()
    order = list(range(len(lines)))
    rng.shuffle(order)
    for li in order:
        if c["n"] >= args.max_lines:
            break
        words = lines[li].split()
        if len(words) < args.min_words:
            continue
        i = rng.randrange(1, len(words) - 1)
        # THE SAME LINE AND THE SAME POSITION FOR ALL THREE, so the three numbers are three
        # decoys and not three populations.
        pick = list(T["freq"])
        for tag, dec in (("matched", matched_decoy(T, words, i, rng)),
                         ("hard", hard_decoy(T, words, i, rng)),
                         ("easy", pick[rng.randrange(len(pick))])):
            if dec is None or dec == words[i]:
                continue
            w2 = list(words)
            w2.insert(i, dec)                     # ONE WORD TOO MANY, at a known slot
            pos = list(range(1, len(w2) - 1))
            if i not in pos:
                continue
            s = scores(T, w2, pos)
            t = pos.index(i)
            c[tag + "_n"] += 1
            c[tag + "_floor"] += 1.0 / len(pos)
            for k, v in s.items():
                c[f"{tag}_{k}"] += acc(v, t)
        c["n"] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=30_000_000)
    ap.add_argument("--frame-max", type=int, default=1)
    ap.add_argument("--min-fillers", type=int, default=1)
    ap.add_argument("--lines", type=int, default=25000)
    ap.add_argument("--window-lines", type=int, default=400)
    ap.add_argument("--min-words", type=int, default=8)
    ap.add_argument("--max-lines", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus", default="data/_wikitext103_train.txt")
    ap.add_argument("--out", default="results/_stage417cl_cut.json")
    args = ap.parse_args()
    text = Path(args.corpus).open("r", encoding="utf-8", errors="ignore").read(args.bytes)
    all_lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    if args.window_lines and args.window_lines < len(lines):
        s0 = rng.randrange(len(lines) - args.window_lines)
        lines = lines[s0:s0 + args.window_lines]
    T = build(lines, args.frame_max, args.min_fillers)
    if not T["fills"]:
        print("no tape")
        return 1
    c = run(T, lines, args, rng)
    rep = {"seed": args.seed, "lines": c["n"], "frames": len(T["fills"])}
    for tag in ("matched", "hard", "easy"):
        n = max(1, c[tag + "_n"])
        rep[tag] = {k: c[f"{tag}_{k}"] / n for k in ("rare", "miss", "local")}
        rep[tag]["floor"] = c[tag + "_floor"] / n
        rep[tag]["n"] = c[tag + "_n"]
        rep[tag]["best"] = max(rep[tag][k] for k in ("rare", "miss", "local"))
        rep[tag]["best_over_floor"] = rep[tag]["best"] - rep[tag]["floor"]
    print(f"{rep['frames']} frames, {rep['lines']} lines")
    for tag in ("matched", "hard", "easy"):
        r = rep[tag]
        print(f"{tag.upper():5s}  floor {r['floor']:.4f}   rare {r['rare']:.4f}   "
              f"miss {r['miss']:.4f}   local {r['local']:.4f}   best-floor "
              f"{r['best_over_floor']:+.4f}   n {r['n']}")
    # THE PRIMARY IS `matched`. `easy` and `hard` are the two ends: easy says how much of any
    # separation is FREQUENCY, hard says what is left when the decoy also fits half the window -
    # which at w=1 is one neighbour, and was the calibration error of the first version.
    m = rep["matched"]
    rep["freq_share"] = rep["easy"]["best"] - m["best"]
    sig = m["best_over_floor"] > 0.05
    room = m["best"] < 0.90
    # THE PRIOR CHECK, and it is a GATE: if `rare` is the best rival even after the frequency
    # band, then frequency still decides and everything above it is 399/400/411 once more.
    prior = m["rare"] < max(m["miss"], m["local"])
    rep["signal"], rep["room"], rep["prior_removed"] = bool(sig), bool(room), bool(prior)
    print(f"DIAGNOSTIC  easy best - matched best {rep['freq_share']:+.4f} (how much of the easy "
          f"arm was rarity)   on matched: rare {m['rare']:.4f} vs best other "
          f"{max(m['miss'], m['local']):.4f}")
    if not prior:
        print("\nTHE FREQUENCY PRIOR STILL DECIDES: `rare` is the best rival even against a "
              "band-matched decoy, so this is 399/400/411 again and nothing else here counts.")
        sig = room = False
    print("\n" + ("SIGNAL AND ROOM on the MATCHED decoy: cutting is decidable and the counts do NOT "
                 "already solve it - there is a gap for a mind to be measured in."
                 if sig and room else
                 ("NO SIGNAL: no count finds the extra word above the floor - the arena is empty."
                  if not sig else
                  "NO ROOM: a count finds it almost always, so the tape answers this by lookup "
                  "and Phi has nothing to learn here (38.3).")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[str(args.seed)] = rep
    out.write_text(json.dumps(prev, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
