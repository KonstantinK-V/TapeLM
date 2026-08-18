"""297: can the tape be cut into slots WITHOUT a parser - and what does it cost?

Three tapes so far were cut by the same rule: extract (anchor | relation | value) from a
sentence, then group by an absolute cosine tau conjoined with a word overlap. The 293 audit
showed what that rule actually produces on wikitext - `canada|and`, `december|the`,
`british|unk` - and `rare_nonzero_rate = 0.0` said the same from the other side: everything two
rows share is ambient English. Every open failure since runs through that cut.

So: cut by the corpus instead of by a grammar. Three rules, all exact, all with NO constant to
choose - which is the whole point, because a threshold at the cutting step is a heuristic at the
place the invariant forbids one:

  words    a slot is a run of non-space characters. Nothing linguistic, nothing tunable.
  repeats  a slot is the LONGEST substring starting here that the corpus writes at least twice.
           Not a chosen length and not a chosen vocabulary size - BPE without the vocabulary
           constant. A piece that occurs once is not knowledge, it is a one-off, and the corpus
           itself says which is which.
  frames   a PLACE is a hole whose left+right frame the corpus writes at least twice, with a
           different filler. Width is the MAXIMUM that still repeats - same idea as repeats, but
           applied to the surroundings of a hole, not to a solid span. The text is punctured,
           not sliced. A frame with several fillers IS an address with several mentions, with no
           tau and no stop-words.

FOUR NUMBERS DECIDE for words/repeats, and none of them needs a model:

  SUPPLY       slots and distinct slots. A tape with no repetition is not a tape, it is text.
  COST         slots per KB, and how many occurrences a type has - that is how many rows an
               import brings, and the graph is quadratic in rows.
  COMPOSITION  the number this audit exists for: what fraction of held-out slots have a type
               the training corpus never wrote. Those cannot be answered by selecting an
               existing row - they must be ASSEMBLED, and 1-NN cannot assemble. If this is
               near zero the new cut changes nothing and we are back to selection; if it is
               large, composition is real and its supply is unlimited.
  CEILING      of the answerable ones, how often the answer is the single most frequent
               continuation - the floor any mind has to clear before it is interesting.

FRAMES add the place-shaped numbers, compared in the face to the parser tape (~800 addresses /
~2500 slots, ~3.1 mentions per address):

  addresses            frames with >= 2 distinct fillers
  mentions_per_address occurrences (rows) per such frame
  corpus_coverage      fraction of token positions that sit in such a hole
  median_frame_width   tokens of context on EACH side at the max width taken

    python _audit297_spans.py [--bytes 200000] [--cap 32] [--frame-max 12]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

WIKI = Path("data/_wikitext103_train.txt")
RES = Path("results")

# the parser tape this audit has to beat or match on supply, measured on the hash/wiki runs
PARSER_REF = {"addresses": 800, "slots": 2500, "mentions_per_address": 3.1}


class SAM:
    """Suffix automaton: every substring of the text, with its occurrence count, in O(n).

    Used for one question only - "is the piece starting here written more than once?" - which is
    exactly what makes `repeats` a rule of the corpus rather than a rule of ours.
    """

    def __init__(self):
        self.nxt = [{}]
        self.link = [-1]
        self.length = [0]
        self.cnt = [0]
        self.last = 0

    def extend(self, ch):
        cur = len(self.nxt)
        self.nxt.append({})
        self.link.append(-1)
        self.length.append(self.length[self.last] + 1)
        self.cnt.append(1)
        pv = self.last
        while pv != -1 and ch not in self.nxt[pv]:
            self.nxt[pv][ch] = cur
            pv = self.link[pv]
        if pv == -1:
            self.link[cur] = 0
        else:
            q = self.nxt[pv][ch]
            if self.length[pv] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = len(self.nxt)
                self.nxt.append(dict(self.nxt[q]))
                self.link.append(self.link[q])
                self.length.append(self.length[pv] + 1)
                self.cnt.append(0)            # a clone owns no endpos of its own
                while pv != -1 and self.nxt[pv].get(ch) == q:
                    self.nxt[pv][ch] = clone
                    pv = self.link[pv]
                self.link[q] = self.link[cur] = clone
        self.last = cur

    def finish(self):
        order = sorted(range(len(self.nxt)), key=lambda i: -self.length[i])
        for v in order:
            p = self.link[v]
            if p > 0 or p == 0:
                if p >= 0:
                    self.cnt[p] += self.cnt[v]


def cut_words(text, _sam=None, _cap=0):
    """A slot is a run of non-space characters. Exact, and there is nothing to tune."""
    out, i, n = [], 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        j = i
        while j < n and not text[j].isspace():
            j += 1
        if j > i:
            out.append(text[i:j])
        i = j
    return out


def cut_repeats(text, sam, cap):
    """A slot is the longest piece starting here that the corpus writes at least twice.

    `cap` bounds the forward walk. It is a COST bound, not a decision: it can only shorten a
    slot the corpus would have allowed, never invent one, and it is reported so its effect is
    visible rather than assumed.
    """
    out, i, n = [], 0, len(text)
    while i < n:
        v, best = 0, 0
        for k in range(i, min(n, i + cap)):
            v = sam.nxt[v].get(text[k], -1)
            if v < 0 or sam.cnt[v] < 2:
                break
            best = k - i + 1
        if best == 0:
            best = 1                          # a character the corpus wrote once: keep it whole
        out.append(text[i:i + best])
        i += best
    return out


def stats(name, train_slots, held_slots, n_bytes, cap):
    tc, hc = Counter(train_slots), Counter(held_slots)
    n_tr = max(1, len(train_slots))
    recur = sum(c for s, c in tc.items() if c >= 2) / n_tr
    occ = sorted(tc.values())
    unseen_types = sum(1 for s in hc if s not in tc)
    unseen_occ = sum(c for s, c in hc.items() if s not in tc)
    n_h = max(1, len(held_slots))
    # of the held slots the train corpus DID write, how often is the answer simply its most
    # frequent piece - the floor a mind must clear before any of this is interesting
    top = tc.most_common(1)[0][0] if tc else None
    ceiling = sum(c for s, c in hc.items() if s == top) / n_h
    return {
        "cutter": name, "cap": cap if name == "repeats" else None,
        "slots_train": len(train_slots), "distinct_train": len(tc),
        "slots_per_kb": len(train_slots) / max(1.0, n_bytes / 1024),
        "mean_len": sum(len(s) for s in train_slots) / n_tr,
        "recurrence": recur,
        "occurrences_median": occ[len(occ) // 2] if occ else 0,
        "occurrences_p90": occ[int(0.9 * (len(occ) - 1))] if occ else 0,
        # THE NUMBER THIS AUDIT EXISTS FOR
        "must_compose_types": unseen_types / max(1, len(hc)),
        "must_compose_occurrences": unseen_occ / n_h,
        "trivial_ceiling": ceiling,
    }


def tokenize(text):
    """Whitespace tokens only - the frame cutter may not smuggle a grammar in through the split."""
    return cut_words(text)


def audit_frames(tokens, w_max):
    """Places = holes whose maximal repeating frame has >= 2 distinct fillers.

    Hole = one token. Frame width w = w tokens on EACH side. For every position take the largest
    w at which that exact (left, right) still occurs somewhere else in the corpus (>= 2 hits);
    keep the frame as an address only when those hits carry at least two different fillers.
    No tau, no stop-list, no chosen width - only counting, with w_max a cost bound like repeats'
    cap (can only shrink, never invent).
    """
    n = len(tokens)
    by_w = []
    for w in range(1, w_max + 1):
        m = defaultdict(list)
        for i in range(w, n - w):
            left = tuple(tokens[i - w: i])
            right = tuple(tokens[i + 1: i + 1 + w])
            m[(left, right)].append(tokens[i])
        by_w.append(m)

    # per address: full multiset of fillers at the chosen max width (from the global map)
    addr_lists = {}
    widths_by_addr = {}
    covered = 0
    for i in range(n):
        chosen = None
        for w in range(min(w_max, i, n - i - 1), 0, -1):
            left = tuple(tokens[i - w: i])
            right = tuple(tokens[i + 1: i + 1 + w])
            key = (left, right)
            if len(by_w[w - 1][key]) >= 2:
                chosen = (w, key)
                break
        if chosen is None:
            continue
        w, key = chosen
        fills = by_w[w - 1][key]
        if len(set(fills)) < 2:
            continue
        ak = (w, key)
        if ak not in addr_lists:
            addr_lists[ak] = list(fills)       # full multiset once per address
            widths_by_addr[ak] = w
        covered += 1

    n_addr = len(addr_lists)
    n_slots = sum(len(v) for v in addr_lists.values())
    # NOTE: covered counts hole positions that landed on a multi-filler frame; n_slots sums the
    # global occurrence lists (each address once). Coverage uses positions; supply uses lists.
    mpa = sorted(len(v) for v in addr_lists.values())
    n_fill = sorted(len(set(v)) for v in addr_lists.values())
    widths = list(widths_by_addr.values())

    samples = []
    for (w, (left, right)), fills in list(addr_lists.items())[:5]:
        samples.append({"w": w, "left": list(left), "right": list(right),
                        "fillers": sorted(set(fills))[:8], "n_fillers": len(set(fills)),
                        "rows": len(fills)})

    # PURE CONFIRM: frame repeats with only one filler - excluded from the multi-filler tape
    # above, but countable here (the §293 supply the parser made unmeasurable)
    confirm_only_addr = confirm_only_rows = 0
    seen_co = set()
    for w, m in enumerate(by_w, start=1):
        for key, fills in m.items():
            if len(fills) < 2 or len(set(fills)) != 1:
                continue
            ak = (w, key)
            if ak in seen_co:
                continue
            seen_co.add(ak)
            confirm_only_addr += 1
            confirm_only_rows += len(fills)

    return {
        "cutter": "frames",
        "frame_max": w_max,
        "hole": "one_token",
        "tokens": n,
        "addresses": n_addr,
        "slots": n_slots,
        "mentions_per_address_mean": (n_slots / n_addr) if n_addr else 0.0,
        "mentions_per_address_median": mpa[len(mpa) // 2] if mpa else 0,
        "distinct_fillers_per_address_median": n_fill[len(n_fill) // 2] if n_fill else 0,
        "corpus_coverage": covered / max(1, n),
        "median_frame_width": sorted(widths)[len(widths) // 2] if widths else 0,
        "mean_frame_width": (sum(widths) / len(widths)) if widths else 0.0,
        "confirm_only_addresses": confirm_only_addr,
        "confirm_only_rows": confirm_only_rows,
        "parser_ref": PARSER_REF,
        "vs_parser_addresses": (n_addr / PARSER_REF["addresses"]) if n_addr else 0.0,
        "vs_parser_slots": (n_slots / PARSER_REF["slots"]) if n_slots else 0.0,
        "sample": samples,
        "_addr_lists": addr_lists,
        "_widths": widths_by_addr,
    }


def frame_quality(addr_lists, widths_by_addr):
    """Slice the frame tape: filler histogram, CONFIRM/DISPUTE, width vs informativeness.

    Quality of a frame = how many DISTINCT fillers it takes - a count, not a threshold. Narrow
    ambient frames (`the ___ of`) take hundreds; informative ones take few. CONFIRM = extra
    rows of a value already on the address; DISPUTE = a different value at the same frame -
    measurable here because the address is the frame, not a tau-group.
    """
    # 1 histogram of distinct fillers, and slot mass in "few-filler" frames
    hist = Counter(len(set(fills)) for fills in addr_lists.values())
    slot_by_nf = Counter()
    for fills in addr_lists.values():
        slot_by_nf[len(set(fills))] += len(fills)
    n_slots = sum(slot_by_nf.values()) or 1
    # cumulative: slots living in frames with <= k distinct fillers
    cum = {}
    running = 0
    for k in sorted(slot_by_nf):
        running += slot_by_nf[k]
        cum[str(k)] = running / n_slots

    # 2 CONFIRM vs DISPUTE per address, then totals
    # confirm_rows = sum_v (count_v - 1) = rows - distinct  (extra supports of a known value)
    # dispute_values = distinct - 1 if we treat first value as seed; better report both masses
    confirm_rows = dispute_addrs = pure_multi = 0
    for fills in addr_lists.values():
        c = Counter(fills)
        confirm_rows += sum(n - 1 for n in c.values() if n > 1)
        if len(c) >= 2:
            dispute_addrs += 1
            pure_multi += sum(c.values())
    n_addr = max(1, len(addr_lists))
    total_rows = sum(len(f) for f in addr_lists.values()) or 1

    # 3 width vs n_fillers
    by_w = defaultdict(list)
    for ak, fills in addr_lists.items():
        by_w[widths_by_addr[ak]].append(len(set(fills)))
    width_vs = {
        str(w): {
            "addresses": len(xs),
            "mean_distinct_fillers": sum(xs) / len(xs),
            "median_distinct_fillers": sorted(xs)[len(xs) // 2],
            "pct_addresses": len(xs) / n_addr,
        }
        for w, xs in sorted(by_w.items())
    }

    return {
        "cutter": "frames_quality",
        "n_addresses": len(addr_lists),
        "n_slots": total_rows,
        "distinct_fillers_hist": {str(k): hist[k] for k in sorted(hist)},
        "slots_by_distinct_fillers": {str(k): slot_by_nf[k] for k in sorted(slot_by_nf)},
        "slot_mass_cumulative_le_k_fillers": cum,
        # usable band without a threshold: report the mass at small k; mind will weigh k itself
        "slot_mass_at_2_fillers": slot_by_nf.get(2, 0) / n_slots,
        "slot_mass_at_3_to_5": sum(slot_by_nf.get(k, 0) for k in range(3, 6)) / n_slots,
        "slot_mass_at_6_to_20": sum(slot_by_nf.get(k, 0) for k in range(6, 21)) / n_slots,
        "slot_mass_above_20": sum(v for k, v in slot_by_nf.items() if k > 20) / n_slots,
        "confirm_extra_rows": confirm_rows,
        "confirm_rate_of_rows": confirm_rows / total_rows,
        # dispute_address_rate is 1.0 BY CONSTRUCTION on the multi-filler tape; the real CONFIRM
        # supply sits in confirm_only_* on the parent frames block
        "addresses_with_dispute": dispute_addrs,
        "dispute_address_rate": dispute_addrs / n_addr,
        "rows_on_dispute_addresses": pure_multi / total_rows,
        "width_vs_fillers": width_vs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=200_000)
    ap.add_argument("--cap", type=int, default=32)
    ap.add_argument("--frame-max", type=int, default=12,
                    help="cost bound on frame half-width in tokens; shrinks only, never invents")
    ap.add_argument("--skip-spans", action="store_true",
                    help="only run the frames audit (spans already measured)")
    a = ap.parse_args()
    if not WIKI.exists():
        print(f"missing {WIKI}")
        return 1
    raw = WIKI.read_text(encoding="utf-8", errors="ignore")[: a.bytes * 2]
    cut = len(raw) // 2
    train, held = raw[:cut], raw[cut: cut + a.bytes // 2]
    print(f"train {len(train)} bytes, held {len(held)} bytes")

    out = []
    if not a.skip_spans:
        sam = SAM()
        for ch in train:
            sam.extend(ch)
        sam.finish()
        print(f"suffix automaton: {len(sam.nxt)} states")

        for name, fn in (("words", cut_words), ("repeats", cut_repeats)):
            tr = fn(train, sam, a.cap)
            hd = fn(held, sam, a.cap)
            # THE CUT IS A PARTITION, and it is worth one line to know it: a cutter that drops or
            # duplicates characters would quietly change the corpus under every number below.
            # `words` legitimately drops whitespace; `repeats` may not drop anything at all.
            if name == "repeats":
                assert "".join(tr) == train and "".join(hd) == held, "repeats lost characters"
            s = stats(name, tr, hd, len(train), a.cap)
            out.append(s)
            print(json.dumps(s))
            print("  sample:", tr[200:215])

    toks = tokenize(train)
    print(f"frames: {len(toks)} tokens, frame-max {a.frame_max}")
    fr = audit_frames(toks, a.frame_max)
    lists = fr.pop("_addr_lists")
    widths = fr.pop("_widths")
    out.append(fr)
    print(json.dumps({k: v for k, v in fr.items() if k != "sample"}))
    print("  sample frames:", json.dumps(fr["sample"][:3], ensure_ascii=False))

    q = frame_quality(lists, widths)
    out.append(q)
    print(json.dumps(q, indent=2))

    RES.mkdir(exist_ok=True)
    tag = f"_{a.bytes}" if a.bytes != 200_000 else ""
    path = RES / f"stage297_span_audit{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
