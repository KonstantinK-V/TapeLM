"""Is there composition on this tape at all - measured before anything is built for it.

WHAT WE HAVE ESTABLISHED, and why this is the question left. The mind beats exact counting when
deciding WHERE to look and when answering where no lookup on the place can answer; that survives
two corpus transplants and does not survive removing the message passing. What it has never done
is assemble an answer from more than one record, and three candidate ways to chain records were
measured and refuted: two hops (2.0% at 138 fillers a step), sentence adjacency (which turned out
to be a word repeating inside one line) and the pointer relation (0.16%). All three looked for
composition IN THE DATA - a chain from record to record.

THIS ASKS THE OTHER FORM: composition IN THE ANSWER. Hide TWO fillers on one line, at two
different places. The answer is a PAIR and it has to be consistent.

    the capital of ___ is ___          hidden: france, paris

Why that is composition rather than a lookup in disguise:

  THE RIVAL IS STRUCTURALLY BLIND on a nameable subset. Counting takes each hole on its own - the
  product of two marginals - and cannot use the joint constraint. If the true pair ALSO never
  stood together anywhere else on the tape, counting has no joint statistic either. Those
  questions are `comp_only`, and they are built the same way walk_only was: not "beat the lookup
  where it is right" but "answer where it has nothing".

  THE MIND NEEDS NOTHING NEW. Phi already scores a COMPLETED world. Filling two holes instead of
  one is the same operation, and how well the completed world hangs together IS the consistency
  of the pair - which is the definition of Phi rather than an addition to it.

  THE TEACHER IS EXACT. The record knows both values.

THE LEAK THAT HAD TO BE CLOSED FIRST. In `the capital of france is paris` the address of the
second hole is `france is|...` - it CONTAINS the first hidden token, so one hole's surroundings
hand over the other's answer. Two holes are only taken when their positions are further apart
than frame_max, so no window can cover the other's token. Exact, checkable, no judgement.

    python _audit308_pairs.py --bytes 30000000 --frame-max 3 --sample region --window-lines 400
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import _tape_frames as tframes
WIKI = Path('data/_wikitext103_train.txt')
OUT = Path('results/_stage308_pairs.json')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bytes', type=int, default=30000000)
    ap.add_argument('--frame-max', type=int, default=3)
    ap.add_argument('--min-fillers', type=int, default=2)
    ap.add_argument('--addresses', type=int, default=1500)
    ap.add_argument('--lines', type=int, default=25000)
    ap.add_argument('--window-lines', type=int, default=0)
    ap.add_argument('--sample', choices=('uniform', 'region'), default='region')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--pairs-per-line', type=int, default=2)
    ap.add_argument('--max-questions', type=int, default=20000)
    ap.add_argument('--joint-lines', type=int, default=400, help='how many co-occurring lines are read when asking whether the true pair ever stood together. A cost bound; raising it can only turn comp_only DOWN, so the number this prints is an upper bound on the subset')
    args = ap.parse_args()
    text = WIKI.open('r', encoding='utf-8', errors='ignore').read(args.bytes)
    all_lines = [l.strip() for l in text.split('\n') if len(l.strip()) >= 80]
    lines = all_lines[:int(0.7 * len(all_lines))][:args.lines]
    rng = random.Random(args.seed)
    keep, toks, owner = tframes.frame_keep(lines, args.frame_max, args.min_fillers)
    if args.sample == 'region':
        if args.window_lines:
            by_line = tframes._by_line(keep, owner)
            start = rng.randrange(max(1, len(lines)))
            acc = defaultdict(list)
            for d in range(args.window_lines):
                for k, i in by_line.get((start + d) % len(lines), ()):
                    acc[k].append(i)
            keep = [(k, sorted(v)) for k, v in acc.items() if len({toks[i] for i in v}) >= args.min_fillers]
            if args.addresses and len(keep) > args.addresses:
                keep = rng.sample(keep, args.addresses)
        else:
            keep = tframes.frame_region(keep, toks, owner, len(lines), args.addresses, rng, args.min_fillers)
    elif args.addresses and len(keep) > args.addresses:
        keep = rng.sample(keep, args.addresses)
    if not keep:
        print('no tape')
        return 1
    place, value, line, pos = ([], [], [], [])
    for (w, left, right), ps in keep:
        name = f"{' '.join(left)}|{' '.join(right)}"
        for i in ps:
            place.append(name)
            value.append(toks[i])
            line.append(owner[i])
            pos.append(i)
    n = len(place)
    slots_at = defaultdict(list)
    on_line = defaultdict(list)
    lines_of_place = defaultdict(set)
    for s in range(n):
        slots_at[place[s]].append(s)
        on_line[line[s]].append(s)
        lines_of_place[place[s]].add(line[s])
    line_map = defaultdict(dict)
    for s in range(n):
        line_map[line[s]][place[s]] = value[s]
    qs = []
    for li, ss in on_line.items():
        if len(ss) < 2:
            continue
        cand = [(a, b) for ai, a in enumerate(ss) for b in ss[ai + 1:] if place[a] != place[b] and abs(pos[a] - pos[b]) > args.frame_max]
        if not cand:
            continue
        rng.shuffle(cand)
        qs.extend(cand[:args.pairs_per_line])
    rng.shuffle(qs)
    if args.max_questions:
        qs = qs[:args.max_questions]
    c = Counter()
    price = []
    for a, b in qs:
        pa, pb, va, vb = (place[a], place[b], value[a], value[b])
        ca = Counter((value[t] for t in slots_at[pa] if t != a))
        cb = Counter((value[t] for t in slots_at[pb] if t != b))
        c['n'] += 1
        price.append(len(ca) * len(cb))
        both = va in ca and vb in cb
        c['both_offered'] += both
        if not both:
            continue
        marg = (ca.most_common(1)[0][0], cb.most_common(1)[0][0])
        right = marg == (va, vb)
        c['marginal_right'] += right
        seen = False
        common = lines_of_place[pa] & lines_of_place[pb]
        for k, lj in enumerate(common):
            if lj == line[a] or k >= args.joint_lines:
                continue
            m = line_map[lj]
            if m.get(pa) == va and m.get(pb) == vb:
                seen = True
                break
        c['joint_seen'] += seen
        c['comp_only'] += not right and (not seen)

    def pct(k, d='n'):
        return c[k] / max(1, c[d])
    rep = {'bytes': args.bytes, 'frame_max': args.frame_max, 'sample': args.sample, 'min_fillers': args.min_fillers, 'window_lines': args.window_lines, 'slots': n, 'questions': c['n'], 'pairs_per_line': args.pairs_per_line, 'both_offered': pct('both_offered'), 'marginal_right': pct('marginal_right', 'both_offered'), 'joint_seen': pct('joint_seen', 'both_offered'), 'comp_only': pct('comp_only'), 'comp_only_of_offered': pct('comp_only', 'both_offered'), 'pair_price_mean': sum(price) / max(1, len(price))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f"tape   {n} slots, {c['n']} two-hole questions (holes further apart than frame_max, so neither window covers the other)")
    print(f"reach  both truths offered {rep['both_offered']:.4f}   pairs to consider {rep['pair_price_mean']:.0f}")
    print(f"rival  marginals right {rep['marginal_right']:.4f}   pair seen together elsewhere {rep['joint_seen']:.4f}   (of those offered)")
    print(f"COMP   counting has nothing either way: {rep['comp_only']:.4f} of all, {rep['comp_only_of_offered']:.4f} of the answerable")
    print(f'\nwritten to {OUT}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())