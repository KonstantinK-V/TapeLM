"""Prove 290 and 291 before spending hours on them. Seconds, no corpus and no model.

Everything checked here is a property of the construction rather than of the tape, so a fake
pack tests it honestly and instantly. Seven things have to hold, and each one is a way the run
could produce a plausible number that means nothing:

  1 the neighbourhood is DETERMINISTIC and does not include the query address. A neighbourhood
    that resamples makes every question a different question at each evaluation - the defect
    the fixed probe tape exists to prevent - and one that includes its own address hands the
    hidden row's siblings back as evidence.
  2 the graph base is invalidated by every helper that derives a question from another. A view
    that reuses the full question's base scores the wrong rows while every printed number still
    looks reasonable, which is the exact shape of failure this project keeps catching late.
  3 the base cache changes nothing: base-built graphs equal freshly-built ones bit for bit
    (torch.equal), so the speedup is a speedup and not a second implementation.
  4 the sparse verb keeps the dense conventions - query row last, sentinel value, rows in tape
    order - because drop_rows, view_of, region_views_of and both rivals all assume them.
  5 the refusal world is the question with the query row left unknown, and it is scored by the
    same Phi as every candidate. If it were anything else, refusal would be a second head.
  6 §19.7's growth: adding the two edge channels with zero-initialised input columns leaves the
    function bit-identical, so a narrower mind can be widened without being retrained.
  7 292's target is FOREIGN to the evidence and all four completed worlds carry the same number
    of rows. Unequal counts are the bookkeeping tell that made the ladder unreadable in 289, and
    a zero shared budget would make the four worlds one graph scored four times.

    python _check290_sparse.py
"""
from __future__ import annotations

import random

import torch

import _stage289_derivation as s289


class FakeTape:
    def __init__(self, values):
        self.values = values


def fake_pack():
    """Four addresses over eight slots. Two share an anchor, two share a relation, and one pair
    shares a rare word - one witness for each of N(a)'s three routes."""
    texts = [
        "kostya was born in the year nineteen eighty five in kaluga",
        "kostya was born in kaluga according to the parish register",
        "kostya played for spartak during the winter season",
        "the parish register of kaluga records several births",
        "sweden defeated canada in the final match of the tournament",
        "canada defeated sweden in the final match of the tournament",
        "leipzig and weimar were connected by the same railway line",
        "weimar and leipzig were connected by the same railway line",
    ]
    vals = ["1985", "Kaluga", "Spartak", "Kaluga", "Canada", "Sweden", "Weimar", "Leipzig"]
    items = [
        {"S": "kostya", "address": "fp0:kostya|born in", "slots": [0, 1], "kind": "clean"},
        {"S": "kostya", "address": "fp1:kostya|played for", "slots": [2], "kind": "clean"},
        {"S": "register", "address": "fp2:register|born in", "slots": [3], "kind": "clean"},
        {"S": "match", "address": "fp3:match|defeated", "slots": [4, 5], "kind": "clean"},
        {"S": "line", "address": "fp4:line|connected by", "slots": [6, 7], "kind": "clean"},
    ]
    postings = {}
    for i, t in enumerate(texts):
        for w in t.split():
            postings.setdefault(w, []).append(i)
    return {"tape": FakeTape(vals), "texts": texts, "texts_lc": [t.lower() for t in texts],
            "items": items, "postings": postings, "n_slots": len(vals)}


class FakeBank:
    def ctx_fp(self, text, exclude=None):
        g = torch.Generator().manual_seed(abs(hash(text)) % (2 ** 31))
        return torch.nn.functional.normalize(torch.randn(16, generator=g), dim=-1)


def main() -> int:
    ok = True
    dev = torch.device("cpu")
    s289.NEIGHBOURS, s289.REFUSE = 2, True
    s289.EDGES_ON = set(s289.EDGES) | set(s289.EDGES_NB)
    s289.IMPORT_K = 0

    p, bank = fake_pack(), FakeBank()
    a = "fp0:kostya|born in"
    nb1 = s289.neighbourhood(p, a, 2)
    p.pop("_nb")
    nb2 = s289.neighbourhood(p, a, 2)
    v = nb1 == nb2 and a not in nb1 and nb1
    ok &= bool(v)
    print(f"1 N(a) = {nb1}")
    print(f"  deterministic, excludes itself, non-empty: {bool(v)}")
    routes = ("fp1:kostya|played for" in nb1, "fp2:register|born in" in nb1)
    print(f"  anchor route hit: {routes[0]}   relation route hit: {routes[1]}")
    ok &= all(routes)

    # 4 the sparse verb's conventions
    q = s289.lookup_sparse_question(p, p["items"][0], random.Random(0), 0, 2)
    qr = q["query_row"]
    v = (qr == len(q["slots"]) - 1
         and not isinstance(q["vals"][qr], str)              # sentinel, matches nothing
         and q["slots"][:qr] == sorted(q["slots"][:qr])      # tape order, so regions cut stretches
         and q["slots"][qr] == p["items"][0]["slots"][0]
         and q["own_rows"] == {1})
    ok &= bool(v)
    print(f"4 sparse question: rows {q['slots']} query_row {qr} own {sorted(q['own_rows'])}  "
          f"conventions kept: {bool(v)}")
    print(f"  candidates {q['cands']}  answerable {q['answerable']}  label {q['label']}")

    # 2 base invalidation
    s289.graph_base(p, q, bank, dev)
    derived = [("view_of", s289.view_of(q, random.Random(1), 0.6)),
               ("drop_rows", s289.drop_rows(q, random.Random(1), 0.6)),
               ("region_views_of", s289.region_views_of(q, 3)[0])]
    for name, d in derived:
        if d is None:
            continue
        v = "_base" not in d
        ok &= v
        print(f"2 {name:16s} drops the cached base: {v}")

    # 3 the cache changes nothing
    same_all = True
    for c in q["cands"]:
        q.pop("_base", None)
        fresh = s289.graph_from_base(p, q, bank, dev, None if c == s289.REFUSE_LABEL else c)
        cached = s289.graph_from_base(p, q, bank, dev, None if c == s289.REFUSE_LABEL else c)
        same_all &= all(torch.equal(x, y) for x, y in zip(fresh, cached))
    ok &= same_all
    print(f"3 cached graph == freshly built graph, every candidate (torch.equal): {same_all}")

    # 5 the refusal world IS the unknown-query-row world, scored by the same Phi
    q.pop("_base", None)
    Er, sr, nr = s289.build_graph(p, q, bank, dev, query_value=s289.REFUSE_LABEL, import_k=0)
    Eu, su, nu = s289.graph_from_base(p, q, bank, dev, None)
    v = torch.equal(Er, Eu) and torch.equal(sr, su) and torch.equal(nr, nu)
    ok &= v
    qrow = q["query_row"]
    v2 = float(nr[qrow][0]) == 0.0 and float(sr[qrow].sum()) == 0.0
    ok &= v2
    print(f"5 refusal world == unknown-query-row world: {v}   "
          f"query row has no value share and no same-value edge: {v2}")
    net = s289.Deriver(dev, d=8, n_edge=5, n_node=9)
    print(f"  Phi scores it like any candidate: {float(net.phi(Er, sr, nr)):+.4f}")

    # 6 §19.7 growth: zero-initialised new columns leave the function untouched
    torch.manual_seed(0)
    narrow = s289.Deriver(dev, d=8, n_edge=3, n_node=9)
    torch.manual_seed(0)
    wide = s289.Deriver(dev, d=8, n_edge=5, n_node=9, grown=2)
    with torch.no_grad():
        wide.edge[0].weight[:, :3] = narrow.edge[0].weight
        wide.edge[0].bias.copy_(narrow.edge[0].bias)
        wide.node[0].weight.copy_(narrow.node[0].weight)
        wide.node[0].bias.copy_(narrow.node[0].bias)
        for i in (0, 2):
            wide.lookup[i].weight.copy_(narrow.lookup[i].weight)
            wide.lookup[i].bias.copy_(narrow.lookup[i].bias)
    with torch.no_grad():
        a_ = narrow.phi(Er[..., :3], sr, nr)
        b_ = wide.phi(Er, sr, nr)
    v = torch.equal(a_, b_)
    ok &= v
    print(f"6 widening 3 -> 5 edge channels is function-identical: {v}  "
          f"({float(a_):+.6f} vs {float(b_):+.6f})")

    # ---------------------------------------------------------------------------- 292
    s289.NEIGHBOURS, s289.OPEN, s289.IMPORT_K = 0, True, 2
    s289.EDGES_ON = set(s289.EDGES)
    s289.LADDER_ON = True
    p2 = fake_pack()
    # give the values mentions elsewhere, or the shared budget is zero and every world is equal
    p2["items"].append({"S": "elsewhere", "address": "fp9:elsewhere|noted",
                        "slots": [8, 9, 10], "kind": "clean"})
    p2["texts"] += ["kaluga was noted in the register of that province",
                    "spartak was noted in the register of that province",
                    "weimar was noted in the register of that province"]
    p2["texts_lc"] = [t.lower() for t in p2["texts"]]
    p2["tape"].values += ["Kaluga", "Spartak", "Weimar"]
    p2["n_slots"] = len(p2["tape"].values)
    for i in (8, 9, 10):
        for w in p2["texts"][i].split():
            p2["postings"].setdefault(w, []).append(i)
    from collections import defaultdict as _dd
    by_anchor = _dd(list)
    for it in p2["items"]:
        by_anchor[it["address"].split(":", 1)[-1].split("|")[0]].append(it)
    allv = list(p2["tape"].values)
    made = []
    for it in p2["items"]:
        for hid in range(len(it["slots"])):
            qq = s289.lookup_open_question(p2, it, random.Random(3), hid, by_anchor, allv)
            if qq is not None:
                made.append(qq)
    print(f"\n-- 292 --\n7 open questions built on the toy tape: {len(made)}")
    if made:
        qo = made[0]
        truth = qo["cands"][qo["label"]]
        ev = [p2["tape"].values[s_] for s_ in qo["slots"][:qo["query_row"]]]
        v = truth not in ev
        ok &= v
        print(f"  truth {truth!r} vs evidence {ev} - foreign to every row: {v}")
        v = len(qo["cands"]) == 4 and qo["cands"] == sorted(qo["cands"])
        ok &= v
        print(f"  four candidates, sorted so position says nothing: {v}  {qo['cands']}")
        k = s289.shared_import_budget(p2, qo, list(qo["cands"]))
        ok &= k >= 1
        print(f"  shared import budget {k} - equal for the truth and all three rungs: {k >= 1}")
        ns = {c: len(s289.build_graph(p2, qo, bank, dev, query_value=c, import_k=k)[2])
              for c in qo["cands"]}
        v = len(set(ns.values())) == 1
        ok &= v
        print(f"  every completed world has the same row count {sorted(set(ns.values()))}: {v}  "
              f"(unequal counts are the bookkeeping tell that killed the ladder in 289)")
    else:
        print("  (toy tape supplied none - the real check is `open.n` in the run report)")

    print("\nSPARSE OK" if ok else "\nSPARSE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
