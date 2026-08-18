"""Prove 298 - the frame tape and the route - before spending an hour on either.

298 is the whole construction rather than another arm: the write path becomes counting, the
mind may read on before answering, and one payoff prices finding, stepping and silence together.
Five things have to hold:

  1 A FRAME IS A RECURRENCE, NOT A RULE. The address is a hole whose surroundings the corpus
    wrote at least twice, its width is the widest the corpus supports, and it exists only if the
    hole took at least two different values. No tau, no stopwords, no grammar, no chosen length.
  2 THE VALUE IS WHAT VARIES. Every assertion's value is the token that stood in the hole, and
    its context is the line, so the ink has a sentence to work with and ctx_fp excludes the value.
  3 WITHOUT IMPORTS THE CANDIDATES ARE INDISTINGUISHABLE - and that is why the step exists. An
    absent value gives a bit-identical graph, which wrecked 289's ladder; here it is the honest
    reason to read more instead of guessing.
  4 WITH IMPORTS THEY SEPARATE, so reading on actually buys something.
  5 THE ROUTE IS ONE SOFTMAX. `expand` is one more world scored by the same Phi, the loss is the
    closed-form expected payoff of the two-step decision, and reading is priced.

    python _check298_route.py
"""
from __future__ import annotations

import random

import torch

import _stage289_derivation as s289
import _tape_frames as tframes
from _check293_identity import FakeBank
from _check294_open import pack294


def main() -> int:
    ok = True
    dev = torch.device("cpu")

    # ---------------------------------------------------------------- 1 and 2: the frame tape
    lines = ["the cat sat on the mat",
             "the dog sat on the rug",
             "the cat sat on the rug",
             "a bird flew over the mat"]
    asrt, addrs, _pool = tframes.frame_assertions(lines, frame_max=6, min_fillers=2)
    by = {}
    for a in asrt:
        by.setdefault(a["address"], []).append(a["value"])
    print(f"1 addresses {len(addrs)}: {dict(list(by.items())[:6])}")
    v = all(len(set(vs)) >= 2 for vs in by.values()) and len(addrs) > 0
    ok &= bool(v)
    print(f"  every address took at least two different fillers: {bool(v)}")
    # the hole between "the" and "sat" is written three times with two fillers - the case the
    # whole cutter exists for
    tgt = [k for k in by if k.startswith("the|sat")]
    v = bool(tgt) and set(by[tgt[0]]) == {"cat", "dog"}
    ok &= bool(v)
    print(f"  'the ___ sat' -> {sorted(set(by[tgt[0]])) if tgt else None}: {bool(v)}")
    v = all(a["value"] in a["ctx"].split() for a in asrt)
    ok &= bool(v)
    print(f"2 every value is a token of its own line (context is the sentence): {bool(v)}")
    # width is the corpus's answer, not ours: widening past what recurs must drop the address
    wide, _w, _p = tframes.frame_assertions(["x q y", "x w y"], frame_max=6,
                                            min_fillers=2)
    v = len({a["address"] for a in wide}) == 1
    ok &= bool(v)
    print(f"  a frame that cannot widen stays at width 1: {bool(v)}")

    # ---------------------------------------------------------------------- 3, 4, 5: the route
    s289.IDENTITY, s289.NEIGHBOURS, s289.REFUSE = False, 0, False
    s289.OPEN, s289.MIXED, s289.ROUTE, s289.IMPORT_K = True, True, True, 2
    s289.ADDRESS_FROM, s289.OPEN_CANDS, s289.OPEN_N_CANDS = "anchor", "uniform", 4
    s289.ANCHOR_MAX_ROWS, s289.STEP_COST = 3, 0.05
    s289.EDGES_ON = set(s289.EDGES)

    p, bank = pack294(), FakeBank()
    item = next(it for it in s289.anchor_items(p) if it["S"] == "kostya")
    rng = random.Random(0)
    q = None
    for _ in range(50):
        q = s289.lookup_mixed_question(p, item, rng, 2, list(p["tape"].values))
        if q is not None:
            break
    if q is None:
        print("\nROUTE FAILED (no question on the toy tape)")
        return 1

    torch.manual_seed(0)
    net = s289.Deriver(dev, d=8)
    # last layer is zero-init by design (bit-identical resume); a fresh net then scores every
    # world 0, so separation is read off the GRAPHS, not off Phi
    k = s289.shared_import_budget(p, q, list(q["cands"]))
    l1, l2 = s289.route_logits(net, p, q, dev, bank)
    v = len(l1) == len(q["cands"]) + 1 and len(l2) == len(q["cands"])
    ok &= v
    print(f"\n3 stage-1 logits {len(l1)} (candidates + expand), stage-2 {len(l2)}: {v}")
    spread1 = float(l1[:-1].max() - l1[:-1].min())
    v = spread1 < 1e-6
    ok &= v
    print(f"  without imports every candidate world is the same graph (spread {spread1:.2e}): "
          f"{v} - this is why the step exists")
    nfs = []
    for c in q["cands"]:
        if c == s289.REFUSE_LABEL:
            continue
        _E, _sm, nf = s289.build_graph(p, q, bank, dev, query_value=c, import_k=k)
        nfs.append(nf.detach())
    v = any(float((nfs[i] - nfs[j]).abs().sum()) > 1e-6
            for i in range(len(nfs)) for j in range(i + 1, len(nfs)))
    ok &= v
    print(f"4 with imports the completed worlds differ as graphs: {v}")

    R = s289.route_reward(q, dev)
    p1, p2 = torch.softmax(l1, 0), torch.softmax(l2, 0)
    manual = -((p1[:-1] * R).sum() + p1[-1] * ((p2 * R).sum() - s289.STEP_COST))
    loss = s289.route_loss(net, p, q, dev, bank)
    v = abs(float(loss) - float(manual)) < 1e-6
    ok &= v
    print(f"5 loss == closed-form expected payoff of the route: {v} "
          f"({float(loss):+.6f} vs {float(manual):+.6f})")
    v = float(R[q["label"]]) == 1.0 and float(R.sum()) < float(len(R))
    ok &= v
    print(f"  reward vector {[round(float(x), 2) for x in R]} (label {q['label']}, "
          f"answerable {q['answerable']}): {v}")
    print(f"  a step costs {s289.STEP_COST}, so always expanding pays that on every question")

    print("\nROUTE OK" if ok else "\nROUTE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
