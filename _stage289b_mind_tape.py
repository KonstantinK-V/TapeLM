"""
Stage 289b — Is there a mind tape? Is the space of situations smooth enough to interpolate in?

A second tape was proposed: not facts, but PATTERNS. Not "Alan Kay was born in Springfield" but
"a coalition of three agreeing mentions with high context rank, plus one outlier nobody
corroborates". Today that knowledge sits smeared across 4,417 weights. On a mind tape it would
sit as POINTS, and inference would be neighbourhood and interpolation - the continuous space a
discrete tape does not have, and the reason a transformer exceeds its dataset while a lookup
table cannot.

Three things it would buy, and they are not small:
  - the repertoire grows WITHOUT retraining, the way facts already do, so the invariant "the
    mind does not grow with knowledge" holds where it should - the READER stays constant while
    experience accumulates;
  - judgment gets provenance: "why did you flag this?" - "because it resembles these three
    stored cases". The project bought that for knowledge; this buys it for reasoning;
  - between two stored patterns there IS a point, so a novel situation can be answered by
    blending. That is the machinery a conjecture needs.

And the condition without which it degenerates immediately: the key must be STRUCTURE ONLY.
Four arms measured the alternative - frozen trunk, rank-8, anonymised text, delta channel - and
every continuous store keyed on identity became a lookup table. So the pattern vector here is
built from ranks, ratios and indicators, is fixed-length regardless of how many mentions the
address has, and cannot name anyone.

This probe trains NOTHING. It fills a memory with patterns from one tape and asks four
questions of the space itself:

  1. SIGNAL. Does 1-nearest-neighbour beat the counting detector and the majority-class floor
     on a held-out tape? k=1 so there is no k to choose.
  2. SMOOTHNESS. Is being right related to being CLOSE? If correctness is independent of
     distance, the space is not a space, it is a bag, and interpolation in it means nothing.
     Measured as the AUC of distance separating wrong neighbours from right ones.
  3. INTERPOLATION - the one that matters for conjecture. Take two stored patterns with the
     SAME label and look up their midpoint: does it land near that label too? Against the
     control of midpoints between DIFFERENT labels, which should not. A space where midpoints
     are meaningless can store patterns but cannot blend them, and blending is the whole
     reason to want continuity.
  4. GROWTH. Does accuracy rise as the memory fills? That is "repertoire grows without
     retraining", measured rather than assumed.

What this cannot answer: whether the mind tape beats the TRAINED head. That needs 288's
checkpoint and is the follow-up. Note though that a tie is already a win - 4,417 weights over
about a thousand examples is close to a memory already, and an explicit memory is auditable
and extensible where weights are neither.

  python _stage289b_mind_tape.py --smoke
  python _stage289b_mind_tape.py
  python _stage289b_mind_tape.py --holdout address
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage271_controller as s271
import _stage280_raw_exam as s280
import _stage286_evidence as s286
import _stage288_repair as s288
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank
from _tape_index import context_words

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 289
QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
LOG_PATH = RES / "_stage289b_log.txt"


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


# ----------------------------------------------------------------- the pattern, identity-free

def rank_pairs(v: np.ndarray) -> np.ndarray:
    """Within-example ranks in [0,1], ties sharing a rank. The only currency allowed here."""
    if v.size == 0:
        return v
    order = v.argsort()
    r = np.empty(len(v), dtype=float)
    r[order] = np.arange(len(v), dtype=float)
    uniq, inv = np.unique(v, return_inverse=True)
    if len(uniq) > 1:
        s = np.zeros(len(uniq))
        c = np.zeros(len(uniq))
        np.add.at(s, inv, r)
        np.add.at(c, inv, 1.0)
        return (s / c)[inv] / max(1, len(v) - 1)
    return np.zeros(len(v))


def pattern_of(same: np.ndarray, cos: np.ndarray, shared: np.ndarray,
               share: np.ndarray, ext: np.ndarray) -> np.ndarray:
    """One example as a fixed-length point, whatever its number of mentions.

    Every entry is a rank, a ratio or a count normalised by n, so two addresses about different
    subjects with the same shape land in the same place - which is the entire premise. Fixed
    length is what makes it a POINT rather than a set, and points are what a continuous space
    is made of.
    """
    n = len(share)
    iu = np.triu_indices(n, 1)
    feats = [1.0 / n, float(np.mean(same[iu])) if iu[0].size else 0.0]
    for M, ranked in ((same, False), (cos, True), (shared, True)):
        v = M[iu] if iu[0].size else np.zeros(1)
        v = rank_pairs(v) if ranked else v
        feats += [float(np.mean(v))] + [float(np.quantile(v, q)) for q in QUANTILES]
    for col in (share, rank_pairs(ext)):
        feats += [float(np.mean(col)), float(np.min(col)), float(np.max(col)),
                  float(np.std(col))]
    return np.asarray(feats, dtype=float)


def knn_predict(mem: np.ndarray, lab: list, q: np.ndarray, exclude: set[int] | None = None):
    """1-NN. k=1 because any other k is a constant somebody chose."""
    d = np.linalg.norm(mem - q, axis=1)
    if exclude:
        d = d.copy()
        d[list(exclude)] = np.inf
    i = int(d.argmin())
    return lab[i], float(d[i]), i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--addresses", type=int, default=0)
    ap.add_argument("--min-mentions", type=int, default=2)
    ap.add_argument("--tapes", type=int, default=8,
                    help="how many resampled training tapes fill the memory. The memory is the "
                         "point of the stage, so it gets more than one tape's worth.")
    ap.add_argument("--address-tau", type=float, default=0.90)
    ap.add_argument("--address-overlap", type=int, default=2)
    ap.add_argument("--addr-key", choices=("two", "set", "mean"), default="two")
    ap.add_argument("--holdout", choices=("corpus", "address"), default="corpus")
    ap.add_argument("--run-tag", type=str, default="")
    args = ap.parse_args()

    global LOG_PATH
    tag = (args.run_tag and f"_{args.run_tag}") or ""
    tag += "_addrholdout" if args.holdout == "address" else ""
    LOG_PATH = RES / f"_stage289b_log{tag}.txt"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    t0 = time.time()
    n_addr = args.addresses or (300 if args.smoke else 400)

    log(f"Stage289b mind tape start {datetime.now(timezone.utc).isoformat()} "
        f"device={device} holdout={args.holdout} tapes={args.tapes}")

    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    pad_id = tok.token_to_id(PAD) or 0
    can = SelfModelXL(n_char, tok.get_vocab_size()).to(device)
    can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    can.eval()
    for p in can.parameters():
        p.requires_grad_(False)
    bank = FpBank(can, stoi, device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(4_000_000 if args.smoke else 30_000_000)
    all_lines = [l.strip() for l in wtext.split("\n") if 80 <= len(l.strip()) <= 400]
    cut = int(0.7 * len(all_lines))
    train_lines = all_lines[:cut][: (3000 if args.smoke else 25000)]
    eval_lines = all_lines[cut:][: (1500 if args.smoke else 12000)]
    if args.holdout == "address":
        eval_lines = train_lines

    def side(address: str) -> int:
        a = address.split(":", 1)[-1].split("|")[0]
        return int(hashlib.sha1(a.encode("utf-8")).hexdigest(), 16) & 1

    def new_pack(r, lines, want):
        p = s280.pack_from_corpus(lines, bank=bank, tok=tok, pad_id=pad_id, device=device,
                                  rng=r, n_addr=n_addr, min_mentions=args.min_mentions,
                                  tau=args.address_tau, overlap=args.address_overlap,
                                  soft_match=0.0, min_per_family=8, addr_key=args.addr_key)
        if args.holdout == "address":
            p = dict(p)
            p["items"] = [it for it in p["items"] if side(it["address"]) == want]
        return p

    def graph(p, ev, item):
        """The three channels and the two node columns, as plain arrays."""
        slots, vals_e = ev["slots"], ev["vals"]
        n = len(slots)
        ck, ws = p.setdefault("_ctx", {}), p.setdefault("_words", {})
        for sl in set(slots):
            if sl not in ck:
                c = bank.ctx_fp(p["texts"][sl], exclude=p["tape"].values[sl])
                ck[sl] = F.normalize(c, dim=-1) if c is not None else None
                ws[sl] = set(context_words(p["texts"][sl], exclude=p["tape"].values[sl]))
        med = p.get("_median")
        if med is None:
            lens = sorted(len(v) for v in p["postings"].values())
            med = lens[len(lens) // 2] if lens else 1
            p["_median"] = med
        same = np.zeros((n, n))
        cos = np.zeros((n, n))
        shared = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                si, sj = slots[i], slots[j]
                same[i, j] = same[j, i] = float(vals_e[i] == vals_e[j])
                if ck[si] is not None and ck[sj] is not None:
                    cos[i, j] = cos[j, i] = float(ck[si] @ ck[sj])
                rare = sum(1 for w in (ws[si] & ws[sj]) if len(p["postings"].get(w, ())) < med)
                shared[i, j] = shared[j, i] = rare / max(1, min(len(ws[si]), len(ws[sj])))
        cnt = Counter(vals_e)
        own = set(item["slots"])
        ex_s = {v: s286.ext_support(p, item["S"], v, own) for v in cnt}
        share = np.asarray([cnt[v] / n for v in vals_e])
        ext = np.asarray([float(ex_s[v]) for v in vals_e])
        return same, cos, shared, share, ext

    def collect(p, r):
        """Every (address, corruption) pair as a point plus its label."""
        X, y = [], []
        for it in p["items"]:
            if len(it["slots"]) < 2:
                continue
            for sl in it["slots"]:
                for op in s288.OPS:
                    ev = s288.corrupt(p, it, r, op, sl)
                    if ev is None:
                        continue
                    X.append(pattern_of(*graph(p, ev, it)))
                    y.append(op)
        return X, y

    # ---------------------------------------------------------------- fill the memory
    memX, memY = [], []
    for t in range(args.tapes):
        p = new_pack(random.Random(SEED + t), train_lines, 0)
        x, yy = collect(p, random.Random(SEED + 100 + t))
        memX += x
        memY += yy
        log(f"  tape {t + 1}/{args.tapes}: +{len(x)} patterns (memory {len(memX)})")
    if len(memX) < 8 * s286.MIN_ANSWERED:
        log(f"  memory too small: {len(memX)}")
        return 1
    M = np.stack(memX)
    # scale each dimension by its own spread, computed on the memory alone and without labels,
    # so no single feature dominates the distance by accident of units
    sd = M.std(0)
    sd[sd < 1e-9] = 1.0
    mu = M.mean(0)
    M = (M - mu) / sd
    log(f"  memory {M.shape[0]} patterns x {M.shape[1]} dims, "
        f"labels {json.dumps(dict(Counter(memY)))}")

    held = new_pack(random.Random(SEED + 99), eval_lines, 1)
    hx, hy = collect(held, random.Random(SEED + 999))
    if len(hx) < 4 * s286.MIN_ANSWERED:
        log(f"  held-out too small: {len(hx)}")
        return 1
    H = (np.stack(hx) - mu) / sd
    log(f"  held out {H.shape[0]} patterns, labels {json.dumps(dict(Counter(hy)))}")

    # ---------------------------------------------------------------- 1. signal
    preds, dists = [], []
    for q in H:
        lab, d, _ = knn_predict(M, memY, q)
        preds.append(lab)
        dists.append(d)
    ok = [int(a == b) for a, b in zip(preds, hy)]
    knn_acc = float(np.mean(ok))
    major = Counter(memY).most_common(1)[0][0]
    floor_major = float(np.mean([int(major == b) for b in hy]))
    # the counting rival, on the same three-way question
    count_ok = []
    for i, lab in enumerate(hy):
        # a counter sees only "is there a minority value" - it cannot tell replace from dup
        count_ok.append(int(lab == "none"))
    floor_count = float(np.mean(count_ok))

    # ---------------------------------------------------------------- 2. smoothness
    d_wrong = [d for d, o in zip(dists, ok) if not o]
    d_right = [d for d, o in zip(dists, ok) if o]
    sm_auc = s286.auc(d_wrong, d_right)          # far neighbours should be the wrong ones
    sm_z = s286.auc_z(sm_auc, len(d_wrong), len(d_right))

    # ---------------------------------------------------------------- 3. interpolation
    irng = random.Random(SEED + 5)
    by_lab = {}
    for i, l in enumerate(memY):
        by_lab.setdefault(l, []).append(i)
    same_hit, diff_hit = [], []
    for _ in range(400):
        l = irng.choice(list(by_lab))
        if len(by_lab[l]) < 2:
            continue
        a, b = irng.sample(by_lab[l], 2)
        lab, _, _ = knn_predict(M, memY, (M[a] + M[b]) / 2.0, exclude={a, b})
        same_hit.append(int(lab == l))
        l2 = irng.choice([k for k in by_lab if k != l] or [l])
        if not by_lab[l2]:
            continue
        c = irng.choice(by_lab[l2])
        lab2, _, _ = knn_predict(M, memY, (M[a] + M[c]) / 2.0, exclude={a, c})
        diff_hit.append(int(lab2 == l))
    interp_same = float(np.mean(same_hit)) if same_hit else float("nan")
    interp_diff = float(np.mean(diff_hit)) if diff_hit else float("nan")

    # ---------------------------------------------------------------- 4. growth
    growth = {}
    for frac in (0.125, 0.25, 0.5, 1.0):
        k = max(1, int(frac * len(M)))
        sub, subY = M[:k], memY[:k]
        acc = float(np.mean([int(knn_predict(sub, subY, q)[0] == t)
                             for q, t in zip(H, hy)]))
        growth[f"{frac:.3f}"] = acc

    g_signal = bool(knn_acc > floor_major and knn_acc > floor_count)
    g_smooth = bool(not math.isnan(sm_z) and sm_z > 1.645)
    g_interp = bool(not math.isnan(interp_same) and not math.isnan(interp_diff)
                    and interp_same > interp_diff)
    g_growth = bool(growth["1.000"] > growth["0.125"])

    overall = ("MIND_TAPE_REAL" if (g_signal and g_smooth and g_interp and g_growth)
               else "MIND_TAPE_STORES_BUT_CANNOT_BLEND" if (g_signal and not g_interp)
               else "MIND_TAPE_PARTIAL" if g_signal
               else "MIND_TAPE_NO")

    out = {
        "stage": "289b", "overall": overall, "seed": SEED, "smoke": args.smoke,
        "holdout": args.holdout, "run_tag": args.run_tag, "trained_parameters": 0,
        "memory": {"n": int(M.shape[0]), "dims": int(M.shape[1]),
                   "labels": dict(Counter(memY)), "tapes": args.tapes},
        "held_out": {"n": int(H.shape[0]), "labels": dict(Counter(hy))},
        "signal": {"knn_1_accuracy": knn_acc, "majority_class_floor": floor_major,
                   "counting_floor": floor_count},
        "smoothness": {"auc_distance_separates_wrong_from_right": sm_auc, "auc_z": sm_z,
                       "mean_distance_when_right": float(np.mean(d_right)) if d_right else
                       float("nan"),
                       "mean_distance_when_wrong": float(np.mean(d_wrong)) if d_wrong else
                       float("nan")},
        "interpolation": {"midpoint_same_label": interp_same,
                          "midpoint_different_label_control": interp_diff,
                          "n_same": len(same_hit), "n_diff": len(diff_hit)},
        "growth": growth,
        "gates": {
            "G_space_has_signal": g_signal,
            "G_space_is_smooth": g_smooth,
            "G_interpolation_holds": g_interp,
            "G_grows_without_training": g_growth,
        },
        "fp_version": s271.fp_version(),
        "note": (
            "Whether a mind tape can exist: not facts but PATTERNS, stored as points in a "
            "continuous space where a novel situation is answered by neighbourhood and "
            "blending. Nothing is trained. The pattern is built from ranks, ratios and "
            "indicators and is fixed-length whatever the address size, so it cannot name "
            "anyone - four measured arms showed that any continuous store keyed on identity "
            "becomes a lookup table, so structure-only is the condition, not a preference. "
            "Four questions of the space itself: does 1-NN beat the majority-class and "
            "counting floors held out; is being right related to being CLOSE, without which "
            "the space is a bag and interpolation is meaningless; does the midpoint between "
            "two same-label patterns land near that label, against the control of midpoints "
            "between different labels - the property a conjecture would rest on; and does "
            "accuracy rise as the memory fills, which is 'the repertoire grows without "
            "retraining' measured rather than assumed. What it cannot say is whether the mind "
            "tape beats 288's trained head; that needs the checkpoint. A tie there is already "
            "a win, because 4,417 weights over a thousand examples is close to a memory "
            "already and an explicit memory is auditable and extensible where weights are "
            "neither."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(), "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"stage289b_decision{tag}.json").write_text(json.dumps(out, indent=2),
                                                       encoding="utf-8")
    log(json.dumps({"overall": overall, "gates": out["gates"],
                    "signal": out["signal"], "interpolation": out["interpolation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
