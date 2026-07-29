"""SOTE V2 metrics harness — role + codebook hit@1/@5.

Import from train digs / eval scripts. Does not train.
Candidates for @k are always closed codebook surfaces.
"""

from __future__ import annotations

from collections import defaultdict

import torch

RELS = ("on", "to")
ROLE_ORDER = ("verb_ing", "rel", "right", "leftish", "other")


def target_role(ex, line_words) -> str:
    tw = ex["target_word"]
    ws = line_words
    if tw.endswith("ing") and tw != "ing":
        return "verb_ing"
    if tw in RELS:
        return "rel"
    pl = int(ex["prefix_len"])
    if pl >= 1 and pl <= len(ws) and ws[pl - 1] in RELS:
        return "right"
    if tw in ws and ws.index(tw) == 0:
        return "leftish"
    return "other"


@torch.no_grad()
def eval_hitk_by_role(model, pairs, lines, word_fps, surfaces, stoi, k: int = 5, min_prefix: int = 2):
    buckets = defaultdict(lambda: {"n": 0, "h1": 0, "h5": 0, "rank_sum": 0.0})
    V = len(surfaces)
    kk = min(k, V)
    chance5 = kk / max(V, 1)

    for ex in pairs:
        if int(ex["prefix_len"]) < min_prefix:
            continue
        line = lines[ex["line_i"]]
        role = target_role(ex, line["words"])
        gold = ex["target_word"]
        if gold not in stoi:
            continue
        ids = list(ex["prefix_word_ids"][-model.max_len :])
        pred = model.pred_from_prefix(word_fps[ids])
        sims = word_fps @ pred
        top_idx = torch.topk(sims, k=kk).indices.tolist()
        top_labs = [surfaces[int(i)] for i in top_idx]
        order = torch.argsort(sims, descending=True)
        rank = int((order == stoi[gold]).nonzero()[0]) + 1
        for name in (role, "ALL"):
            b = buckets[name]
            b["n"] += 1
            b["h1"] += int(top_labs[0] == gold)
            b["h5"] += int(gold in top_labs)
            b["rank_sum"] += rank

    out = {"chance_at_k": chance5, "k": kk, "V": V, "roles": {}}
    for name, b in buckets.items():
        n = max(b["n"], 1)
        out["roles"][name] = {
            "n": b["n"],
            "hit1": b["h1"] / n if b["n"] else 0.0,
            "hit5": b["h5"] / n if b["n"] else 0.0,
            "mean_rank": b["rank_sum"] / n if b["n"] else 0.0,
        }
    return out


def fmt_role_table(block, title: str) -> list[str]:
    lines = [
        f"=== {title} ===",
        f"  V={block['V']}  k={block['k']}  chance@k={block['chance_at_k']*100:.2f}%",
        f"  {'role':10s} {'n':>5} {'hit@1':>8} {'hit@5':>8} {'mean_rk':>8}",
    ]
    order = ["ALL"] + [r for r in ROLE_ORDER if r in block["roles"]]
    for extra in sorted(block["roles"]):
        if extra not in order:
            order.append(extra)
    for name in order:
        if name not in block["roles"]:
            continue
        r = block["roles"][name]
        if r["n"] == 0:
            continue
        lines.append(
            f"  {name:10s} {r['n']:5d} {r['hit1']*100:7.1f}% {r['hit5']*100:7.1f}% {r['mean_rank']:8.1f}"
        )
    return lines
