"""
Probe five candidate SOTE variants against the 7.8c checkpoint.

1) Morphology discovery — unsupervised cluster of morph deltas
2) Residual as feature — relatedness / retrieval with R_span vs compose
3) Curiosity-driven stabilize — novelty vs always-EMA
4) Hierarchical attractors — readiness scores from residual/morph/compound
5) Dynamic context shift — how much prev-word fp pollutes compose

Writes:
  results/variant_probes_report.txt
  results/variant_probes.json

Run:
  python variant_probes.py
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

from residual_audit import (
    EXTRA_PAIRS,
    COMPOUND_PROBES,
    build_pair_list,
    compose_word,
    cos,
    load_78c,
    project_onto_span,
    residuals_for_word,
)
from train import (
    ALL_WORDS,
    HOLD_WORDS,
    LETTER2ID,
    RELATED_FORMS,
    RESULTS_DIR,
    TRAIN_WORDS,
    pick_device,
)


def kmeans(x: torch.Tensor, k: int, iters: int = 40, seed: int = 0):
    """x: (n, d) normalized rows. Returns labels (n,), centers (k, d)."""
    n, d = x.shape
    g = torch.Generator().manual_seed(seed)
    # init: random distinct points
    perm = torch.randperm(n, generator=g)[:k]
    centers = x[perm].clone()
    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        sims = x @ centers.T
        labels = sims.argmax(dim=1)
        new = []
        for j in range(k):
            mask = labels == j
            if mask.any():
                c = F.normalize(x[mask].mean(0), dim=-1)
            else:
                c = x[torch.randint(0, n, (1,), generator=g)].squeeze(0)
            new.append(c)
        centers = torch.stack(new, 0)
    return labels, centers


def cluster_purity(labels, tags):
    # majority tag per cluster
    by_c = defaultdict(list)
    for lab, tag in zip(labels.tolist(), tags):
        by_c[lab].append(tag)
    correct = 0
    for labs in by_c.values():
        correct += Counter(labs).most_common(1)[0][1]
    return correct / max(len(tags), 1)


def nmi_approx(labels, tags):
    """Simple NMI between cluster ids and tag strings."""
    n = len(tags)
    tag_ids = {t: i for i, t in enumerate(sorted(set(tags)))}
    y = [tag_ids[t] for t in tags]
    x = labels.tolist()
    cx, cy = Counter(x), Counter(y)
    cxy = Counter(zip(x, y))

    def H(counts):
        s = 0.0
        for c in counts.values():
            p = c / n
            s -= p * math.log(p + 1e-12)
        return s

    mi = 0.0
    for (a, b), c in cxy.items():
        pxy = c / n
        mi += pxy * math.log(pxy / ((cx[a] / n) * (cy[b] / n) + 1e-12) + 1e-12)
    hx, hy = H(cx), H(cy)
    return float(2 * mi / (hx + hy + 1e-12))


@torch.no_grad()
def morph_discovery(encoder, composer, basins, device):
    pairs = build_pair_list()
    # keep tags with enough mass + all for clustering matrix
    deltas = []
    tags = []
    meta = []
    for stem, form, tag in pairs:
        cs = compose_word(composer, encoder, stem, device)
        cf = compose_word(composer, encoder, form, device)
        d = F.normalize(cf - cs, dim=-1).cpu()
        deltas.append(d)
        tags.append(tag)
        meta.append({"stem": stem, "form": form, "tag": tag})
    x = torch.stack(deltas, 0)
    # focus k on frequent morph classes
    major = [t for t in tags if tags.count(t) >= 2]
    k = len(set(major))
    labels, centers = kmeans(x, k=max(k, 2), seed=1)
    purity = cluster_purity(labels, tags)
    nmi = nmi_approx(labels, tags)

    # per discovered cluster: dominant tag + mean intra cos
    clusters = []
    for j in range(int(labels.max().item()) + 1):
        idx = (labels == j).nonzero(as_tuple=False).view(-1)
        members = [meta[i] for i in idx.tolist()]
        tag_counts = Counter(m["tag"] for m in members)
        dom, dom_n = tag_counts.most_common(1)[0]
        vecs = x[idx]
        intra = float((vecs @ vecs.T).fill_diagonal_(0).sum() / max(len(idx) * (len(idx) - 1), 1))
        clusters.append(
            {
                "id": j,
                "size": len(members),
                "dominant_tag": dom,
                "dominant_frac": dom_n / len(members),
                "intra_cos": intra,
                "members": [f"{m['stem']}->{m['form']}" for m in members],
            }
        )

    # automatic rule readiness: tags with purity-friendly cluster
    ready = []
    for tag in sorted(set(tags)):
        if tags.count(tag) < 2:
            continue
        # best cluster recall for this tag
        best = 0.0
        for j in range(int(labels.max().item()) + 1):
            idx = (labels == j).nonzero(as_tuple=False).view(-1).tolist()
            if not idx:
                continue
            hit = sum(1 for i in idx if tags[i] == tag)
            rec = hit / tags.count(tag)
            prec = hit / len(idx)
            best = max(best, 2 * prec * rec / (prec + rec + 1e-12))
        ready.append({"tag": tag, "n": tags.count(tag), "best_f1": best})

    # Pairwise within-tag cohesion (imbalance-aware); unsupervised purity alone
    # is dominated by frequent -s pairs.
    by_tag_vecs = defaultdict(list)
    for d, tag in zip(deltas, tags):
        by_tag_vecs[tag].append(d)
    pairwise = []
    for tag, vecs in by_tag_vecs.items():
        if len(vecs) < 2:
            continue
        m = torch.stack(vecs)
        intra = float((m @ m.T).fill_diagonal_(0).sum() / (len(vecs) * (len(vecs) - 1)))
        pairwise.append({"tag": tag, "n": len(vecs), "intra_cos": intra})
    pairwise.sort(key=lambda r: -r["intra_cos"])

    strong = [p for p in pairwise if p["n"] >= 3 and p["intra_cos"] >= 0.25]
    verdict = (
        f"seeded prior OK for: {', '.join(p['tag'] for p in strong)}; "
        "full unsupervised discovery weak (class imbalance)"
        if strong
        else "weak — keep diagnostic only"
    )

    return {
        "n_pairs": len(pairs),
        "k": int(labels.max().item()) + 1,
        "purity": purity,
        "nmi": nmi,
        "clusters": clusters,
        "tag_readiness": sorted(ready, key=lambda r: -r["best_f1"]),
        "pairwise_intra": pairwise,
        "verdict": verdict,
    }


@torch.no_grad()
def residual_as_feature(encoder, composer, basins, device, rng: random.Random):
    cache = {w: residuals_for_word(composer, encoder, basins, w, device) for w in ALL_WORDS}
    related = [(a, b) for a, b in RELATED_FORMS if a in cache and b in cache]
    related += [(a, b) for a, b, _ in EXTRA_PAIRS if a in cache and b in cache]
    related = list(dict.fromkeys(related))

    def mean_pair_cos(pairs, key):
        return sum(cos(cache[a][key], cache[b][key]) for a, b in pairs) / max(len(pairs), 1)

    random_pairs = []
    while len(random_pairs) < len(related):
        a, b = rng.sample(ALL_WORDS, 2)
        if (a, b) not in related and (b, a) not in related:
            random_pairs.append((a, b))

    # retrieval: given stem, find form among candidates by stem+proto vs by compose alone
    # proto = mean delta of other pairs with same tag
    pairs = build_pair_list()
    by_tag = defaultdict(list)
    for stem, form, tag in pairs:
        by_tag[tag].append((stem, form))

    def retrieve(use_residual: bool):
        ok = 0
        n = 0
        for tag, group in by_tag.items():
            if len(group) < 2:
                continue
            for i, (stem, form) in enumerate(group):
                others = [g for j, g in enumerate(group) if j != i]
                deltas = []
                for s, f in others:
                    if use_residual:
                        d = cache[f]["r_span"] - cache[s]["r_span"]
                    else:
                        d = cache[f]["compose"] - cache[s]["compose"]
                    deltas.append(F.normalize(d, dim=-1))
                proto = F.normalize(torch.stack(deltas).mean(0), dim=-1)
                base = cache[stem]["r_span" if use_residual else "compose"]
                pred = F.normalize(base + proto, dim=-1)
                # candidates: all forms in ALL_WORDS with same length band / all related forms
                cands = list({f for _, f in group}) + [
                    w for w in ALL_WORDS if w != stem and w not in {f for _, f in group}
                ]
                cands = list(dict.fromkeys(cands))[:40]
                # score candidates
                best_w, best_s = None, -2.0
                key = "r_span" if use_residual else "compose"
                for w in cands:
                    s = cos(pred, cache[w][key])
                    if s > best_s:
                        best_s, best_w = s, w
                ok += int(best_w == form)
                n += 1
        return ok / max(n, 1)

    # relatedness separation
    rel_c = mean_pair_cos(related, "compose")
    rnd_c = mean_pair_cos(random_pairs, "compose")
    rel_r = mean_pair_cos(related, "r_span")
    rnd_r = mean_pair_cos(random_pairs, "r_span")
    compose_gap = rel_c - rnd_c
    rspan_gap = rel_r - rnd_r

    return {
        "related_compose_cos": rel_c,
        "random_compose_cos": rnd_c,
        "compose_gap": compose_gap,
        "related_rspan_cos": rel_r,
        "random_rspan_cos": rnd_r,
        "rspan_gap": rspan_gap,
        "retrieve_compose_acc": retrieve(False),
        "retrieve_rspan_acc": retrieve(True),
        "verdict": (
            "residual ≈ compose for relatedness — use as aux loss / morph proto, not replace compose"
            if abs(rspan_gap - compose_gap) < 0.15
            else (
                "USE residual in loss / morph proto"
                if rspan_gap > compose_gap * 0.5
                else "compose still dominates — residual auxiliary only"
            )
        ),
    }


@torch.no_grad()
def curiosity_probe(encoder, composer, memory, device):
    """Compare always-write EMA vs novelty-gated create."""
    mem_words = list(memory.entries.keys())
    if not mem_words:
        return {"verdict": "no memory — skip", "n_memory": 0}

    bank = []
    names = []
    for w in mem_words:
        v = memory.get(w)
        if v is None:
            continue
        bank.append(F.normalize(v.to(device), dim=-1))
        names.append(w)
    bank_t = torch.stack(bank, 0)  # (M, d)

    rows = []
    for w in ALL_WORDS:
        c = compose_word(composer, encoder, w, device)
        sims = bank_t @ c
        # exclude self slot if present
        if w in names:
            sims = sims.clone()
            sims[names.index(w)] = -1.0
        max_sim = float(sims.max())
        novelty = 1.0 - max_sim
        # curiosity gate proposals
        create_if = novelty >= 0.45  # would create attractor
        stabilize_if = max_sim >= 0.70  # close enough to update existing
        ambiguous = (not create_if) and (not stabilize_if)
        rows.append(
            {
                "word": w,
                "set": "hold" if w in HOLD_WORDS else "train",
                "max_sim_other": max_sim,
                "novelty": novelty,
                "gate_create": create_if,
                "gate_stabilize": stabilize_if,
                "ambiguous": ambiguous,
                "mem_count": memory.count(w),
            }
        )

    hold = [r for r in rows if r["set"] == "hold"]
    train = [r for r in rows if r["set"] == "train"]

    def rate(xs, key):
        return sum(1 for r in xs if r[key]) / max(len(xs), 1)

    return {
        "n_memory": len(names),
        "hold_mean_novelty": sum(r["novelty"] for r in hold) / max(len(hold), 1),
        "train_mean_novelty": sum(r["novelty"] for r in train) / max(len(train), 1),
        "hold_create_rate": rate(hold, "gate_create"),
        "train_create_rate": rate(train, "gate_create"),
        "hold_stabilize_rate": rate(hold, "gate_stabilize"),
        "hold_ambiguous_rate": rate(hold, "ambiguous"),
        "always_ema_updates_per_word": {
            "hold_mean_count": sum(r["mem_count"] for r in hold) / max(len(hold), 1),
            "note": "7.8c writes every exposure — not selective",
        },
        "examples": sorted(hold, key=lambda r: -r["novelty"])[:5]
        + sorted(hold, key=lambda r: r["novelty"])[:3],
        "verdict": (
            "implement curiosity gate next train — always-EMA overwrites "
            f"(~{sum(r['mem_count'] for r in hold) / max(len(hold), 1):.0f} updates/HOLD word); "
            "novelty signal mid but selectivity still needed"
        ),
    }


@torch.no_grad()
def hierarchical_readiness(encoder, composer, basins, device, morph, residual):
    compounds = []
    for left, right, whole in COMPOUND_PROBES:
        if whole not in ALL_WORDS:
            continue
        cw = compose_word(composer, encoder, whole, device)
        parts = []
        for p in (left, right):
            if p in ALL_WORDS:
                parts.append(compose_word(composer, encoder, p, device))
            else:
                codebook = basins.normalized()
                fps = torch.stack([codebook[LETTER2ID[c]] for c in p], 0).unsqueeze(0)
                lengths = torch.tensor([len(p)], device=device)
                parts.append(F.normalize(composer(fps, lengths).squeeze(0), dim=-1))
        mean_p = F.normalize(torch.stack(parts).mean(0), dim=-1)
        compounds.append(cos(cw, mean_p))

    ing_pair = next((p for p in morph.get("pairwise_intra", []) if p["tag"] == "ing"), None)
    s_pair = next((p for p in morph.get("pairwise_intra", []) if p["tag"] == "s"), None)
    levels = {
        "letter": {
            "status": "frozen",
            "evidence": "alphabet_full_frozen.pt",
            "ready": True,
        },
        "subword": {
            "status": "partial",
            "ing_intra": None if ing_pair is None else ing_pair["intra_cos"],
            "s_intra": None if s_pair is None else s_pair["intra_cos"],
            "ready": bool(ing_pair and ing_pair["intra_cos"] >= 0.25),
            "note": "seed soft -ing only; unsupervised -s mixed; no full subword layer yet",
        },
        "word": {
            "status": "soft memory (7.8c)",
            "ready": True,
            "note": "create+EMA works; freeze only after cold-OOV ok",
        },
        "phrase": {
            "status": "not ready",
            "compound_mean_part_cos": sum(compounds) / max(len(compounds), 1),
            "ready": False,
            "note": "compounds not additive; need word contract first",
        },
    }
    return {
        "levels": levels,
        "verdict": "letter OK → soft subword(-ing) only → word soft → phrase later",
    }


@torch.no_grad()
def context_shift_probe(encoder, composer, basins, device, rng: random.Random):
    """word imprint = compose + alpha * prev; measure trace drop & identity drift."""
    words = TRAIN_WORDS[:]
    alphas = [0.0, 0.1, 0.25, 0.5]
    codebook = basins.normalized()
    rows = []
    for alpha in alphas:
        trace_keep = []
        self_cos = []
        for w in words:
            prev = rng.choice([x for x in words if x != w])
            c = compose_word(composer, encoder, w, device)
            p = compose_word(composer, encoder, prev, device)
            imprint = F.normalize(c + alpha * p, dim=-1)
            # letter traces for chars in w
            traces = [float((imprint * codebook[LETTER2ID[ch]]).sum()) for ch in set(w)]
            base = [float((c * codebook[LETTER2ID[ch]]).sum()) for ch in set(w)]
            trace_keep.append(min(traces) - min(base))
            self_cos.append(cos(imprint, c))
        rows.append(
            {
                "alpha": alpha,
                "mean_self_cos": sum(self_cos) / len(self_cos),
                "mean_min_trace_delta": sum(trace_keep) / len(trace_keep),
            }
        )

    # recommendation: max alpha with self_cos>=0.9 and trace_delta > -0.05
    safe = [r for r in rows if r["mean_self_cos"] >= 0.9 and r["mean_min_trace_delta"] >= -0.05]
    max_safe = max((r["alpha"] for r in safe), default=0.0)
    return {
        "curves": rows,
        "max_safe_alpha": max_safe,
        "verdict": (
            f"allow weak context alpha<={max_safe}"
            if max_safe > 0
            else "keep context tiny / residual-only — letter traces fragile"
        ),
    }


def format_report(payload: dict) -> str:
    m, r, c, h, x = (
        payload["morphology_discovery"],
        payload["residual_as_feature"],
        payload["curiosity"],
        payload["hierarchical"],
        payload["context_shift"],
    )
    lines = [
        "SOTE variant probes (against 7.8c)",
        f"timestamp: {payload['timestamp']}",
        "",
        "=== 1. Morphology discovery ===",
        f"purity={m['purity']:.2f}  NMI={m['nmi']:.2f}  k={m['k']}  pairs={m['n_pairs']}",
        f"verdict: {m['verdict']}",
        "pairwise intra-tag delta cos (better than k-means under imbalance):",
    ]
    for t in m.get("pairwise_intra", []):
        lines.append(f"  {t['tag']:8s} n={t['n']:2d}  intra={t['intra_cos']:.3f}")
    lines.append("unsupervised tag readiness (F1 of best cluster):")
    for t in m["tag_readiness"]:
        lines.append(f"  {t['tag']:8s} n={t['n']:2d}  F1={t['best_f1']:.2f}")
    lines.append("clusters:")
    for cl in m["clusters"]:
        lines.append(
            f"  c{cl['id']}: dom={cl['dominant_tag']}({cl['dominant_frac']:.0%}) "
            f"n={cl['size']} intra={cl['intra_cos']:.2f}  {', '.join(cl['members'][:6])}"
        )

    lines += [
        "",
        "=== 2. Residual as feature ===",
        f"related/random compose cos: {r['related_compose_cos']:.3f} / {r['random_compose_cos']:.3f} "
        f"(gap {r['compose_gap']:.3f})",
        f"related/random R_span cos:  {r['related_rspan_cos']:.3f} / {r['random_rspan_cos']:.3f} "
        f"(gap {r['rspan_gap']:.3f})",
        f"morph retrieve acc compose/rspan: {r['retrieve_compose_acc']:.0%} / {r['retrieve_rspan_acc']:.0%}",
        f"verdict: {r['verdict']}",
        "",
        "=== 3. Curiosity-driven stabilization ===",
        f"memory slots: {c.get('n_memory')}",
        f"HOLD/train mean novelty: {c.get('hold_mean_novelty', 0):.3f} / {c.get('train_mean_novelty', 0):.3f}",
        f"HOLD create/stabilize/ambiguous rates: "
        f"{c.get('hold_create_rate', 0):.0%}/{c.get('hold_stabilize_rate', 0):.0%}/{c.get('hold_ambiguous_rate', 0):.0%}",
        f"always-EMA hold mean count: {c.get('always_ema_updates_per_word', {}).get('hold_mean_count', 0):.0f}",
        f"verdict: {c.get('verdict')}",
        "",
        "=== 4. Hierarchical attractors ===",
        f"verdict: {h['verdict']}",
    ]
    for name, lvl in h["levels"].items():
        lines.append(f"  {name:8s} ready={lvl.get('ready')}  {lvl.get('status')}  {lvl.get('note', '')}")

    lines += [
        "",
        "=== 5. Dynamic context shift ===",
        f"max_safe_alpha: {x['max_safe_alpha']}",
        f"verdict: {x['verdict']}",
    ]
    for row in x["curves"]:
        lines.append(
            f"  alpha={row['alpha']:.2f}  self_cos={row['mean_self_cos']:.3f}  "
            f"d(min_trace)={row['mean_min_trace_delta']:+.3f}"
        )

    lines += [
        "",
        "=== Priority recommendation ===",
        "1. Residual-as-feature (aux) + seeded -ing morph prior (pairwise evidence)",
        "2. Curiosity gate on SoftWordMemory (stop always-EMA)",
        "3. Soft subword only for high-intra tags (-ing); not full hierarchy",
        "4. Context shift alpha<=0.25 after traces stronger; keep weak",
        "5. Phrase level after word contract freezes",
        "",
    ]
    return "\n".join(lines)


@torch.no_grad()
def main():
    device = pick_device()
    rng = random.Random(0)
    encoder, composer, basins, memory, _ckpt = load_78c(device)
    print("Running variant probes on", device)

    morph = morph_discovery(encoder, composer, basins, device)
    residual = residual_as_feature(encoder, composer, basins, device, rng)
    curiosity = curiosity_probe(encoder, composer, memory, device)
    hier = hierarchical_readiness(encoder, composer, basins, device, morph, residual)
    context = context_shift_probe(encoder, composer, basins, device, rng)

    payload = {
        "stage": "variant_probes",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "morphology_discovery": morph,
        "residual_as_feature": residual,
        "curiosity": curiosity,
        "hierarchical": hier,
        "context_shift": context,
    }
    report = format_report(payload)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "variant_probes_report.txt").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "variant_probes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report)
    print("Saved results/variant_probes_report.txt")


if __name__ == "__main__":
    main()
