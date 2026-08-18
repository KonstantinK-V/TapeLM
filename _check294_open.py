"""Prove 294 before spending an hour on it. Seconds, no corpus and no model.

294 is 292's question with 293's two disciplines applied and the addressing heuristic removed:
an address is one exact anchor string, and the three wrong answers are drawn rather than built.
Five things have to hold, and each is a way a plausible number could mean nothing:

  1 AN ADDRESS IS AN EXACT STRING. Every row of an anchor address was written under that anchor
    and no other, so nothing about which rows count as evidence went through a cosine, a tau or
    a word overlap. That is the whole point: the invariant forbids an approximation at the place
    where something is decided, and this is that place.
  2 THE TARGET IS FOREIGN. The hidden value is on none of the evidence rows, so retrieval inside
    the address cannot reach it - 292's property, kept, because it is what makes the comparison
    against the mind worth running.
  3 EVERY WORLD CARRIES THE SAME ROWS. Equal imports through the shared budget and an identical
    evidence set, or Phi can read a row count as the answer, which is the bookkeeping tell that
    made 289's ladder unreadable.
  4 NO CANDIDATE IS BUILT FROM ITS DISTANCE. Distractors are any value the address does not
    carry; `bucket_of` records where each one happens to live and is used only in the report.
    292's rungs were built BY relatedness and `mean_phi` read that construction back as an
    inverted landscape on six seeds out of six - this is the arm that tells the two apart.
  5 THE ROW CAP IS A BUDGET, NOT A DECISION. An anchor carries dozens of mentions and the graph
    is quadratic; the cap keeps the nearest in tape order, in tape order, identically in all
    four worlds.

    python _check294_open.py
"""
from __future__ import annotations

import random
from collections import Counter

import torch

import _stage289_derivation as s289
from _check293_identity import FakeBank, fake_pack


def pack294():
    """293's toy tape plus a second anchor, so the values hidden at `kostya` have a mention
    somewhere else and the shared import budget is not zero."""
    p = fake_pack()
    extra = ["dynamo signed spartak players before the tournament",
             "dynamo travelled to moscow for the away leg",
             "dynamo recruited gorky graduates that summer",
             "dynamo opened a berlin office after the merger",
             "dynamo sent reserves to paris for the friendly",
             "dynamo loaned players to lyon that autumn"]
    vals = ["Spartak", "Moscow", "Gorky", "Berlin", "Paris", "Lyon"]
    straddr = ["dynamo|signed", "dynamo|travelled to", "dynamo|recruited",
               "dynamo|opened", "dynamo|sent", "dynamo|loaned"]
    base = len(p["texts"])
    p["texts"] += extra
    p["texts_lc"] = [t.lower() for t in p["texts"]]
    p["tape"].values += vals
    p["straddr"] += straddr
    p["items"].append({"S": "dynamo", "address": "fp9:dynamo|signed",
                       "slots": list(range(base, base + len(extra))), "kind": "clean"})
    for i, t in enumerate(extra):
        for w in t.split():
            p["postings"].setdefault(w, []).append(base + i)
    n = len(p["texts"])
    p["n_slots"] = n
    g = torch.Generator().manual_seed(294)
    p["ctx_keys"] = torch.nn.functional.normalize(torch.randn(n, 16, generator=g), dim=-1)
    p["anc_keys"] = torch.nn.functional.normalize(torch.randn(n, 16, generator=g), dim=-1)
    p["slot_keys_slot"] = list(range(n))
    p.pop("_ident", None)
    return p


def main() -> int:
    ok = True
    dev = torch.device("cpu")
    s289.IDENTITY, s289.NEIGHBOURS = False, 0
    s289.OPEN, s289.IMPORT_K = True, 2
    s289.ADDRESS_FROM, s289.OPEN_CANDS, s289.OPEN_N_CANDS = "anchor", "uniform", 4
    s289.ANCHOR_MAX_ROWS = 3
    s289.EDGES_ON = set(s289.EDGES)

    p, bank = pack294(), FakeBank()

    items = s289.anchor_items(p)
    by_anc = {it["S"]: it for it in items}
    v = (sorted(by_anc) == ["dynamo", "kostya"]
         and all(s289.str_parts(p["straddr"][s])[0] == it["S"]
                 for it in items for s in it["slots"]))
    ok &= bool(v)
    print(f"1 anchor addresses {[(it['S'], it['slots']) for it in items]}")
    print(f"  every row written under that exact anchor, no grouping rule involved: {bool(v)}")

    qs = [q for q in s289.open_questions_for(p, random.Random(0)) if q.get("uniform")]
    print(f"  open questions built: {len(qs)}")
    ok &= len(qs) > 0
    if not qs:
        print("\nOPEN294 FAILED (no questions on the toy tape)")
        return 1
    q = qs[0]
    truth = q["cands"][q["label"]]
    ev = [p["tape"].values[s] for s in q["slots"][:q["query_row"]]]
    v = truth not in ev and all(c not in ev for c in q["cands"] if c == truth)
    ok &= bool(v)
    print(f"2 truth {truth!r} vs evidence {ev} - foreign to every row: {bool(v)}")

    k = s289.shared_import_budget(p, q, list(q["cands"]))
    ns = {len(s289.build_graph(p, q, bank, dev, query_value=c, import_k=k)[2])
          for c in q["cands"]}
    v = k >= 1 and len(ns) == 1
    ok &= bool(v)
    print(f"3 shared budget {k}, world sizes {ns} - one size for four worlds: {bool(v)}")

    v = (set(q["bucket_of"]) == {c for c in q["cands"] if c != truth}
         and all(b in ("same_anchor", "elsewhere") for b in q["bucket_of"].values())
         and truth not in q["bucket_of"])
    ok &= bool(v)
    print(f"4 buckets {q['bucket_of']} recorded for the distractors only, truth excluded: "
          f"{bool(v)}")
    pos = Counter()
    for seed in range(300):
        qq = s289.lookup_open_uniform(p, by_anc["kostya"], random.Random(seed), 0,
                                      list(p["tape"].values))
        if qq is not None:
            pos[qq["label"]] += 1
    print(f"  label position over {sum(pos.values())} draws: {dict(sorted(pos.items()))}")

    v = (q["query_row"] == len(q["slots"]) - 1
         and q["query_row"] <= s289.ANCHOR_MAX_ROWS
         and q["slots"][:q["query_row"]] == sorted(q["slots"][:q["query_row"]]))
    ok &= bool(v)
    print(f"5 rows {q['slots']} capped at {s289.ANCHOR_MAX_ROWS}, in tape order, query row "
          f"last: {bool(v)}")

    print("\nOPEN294 OK" if ok else "\nOPEN294 FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
