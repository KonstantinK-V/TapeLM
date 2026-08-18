"""
Stage 281 — What counts as an assertion, decided by the corpus rather than by a regex.

280 stopped at a wall its own gate caught: the teacher ceiling on raw text came out at -0.189,
below the 0.750 that unconditional silence scores, and the score-gap cut changed the held-out
numbers by nothing at all - byte for byte the same precision and the same ceiling. The candidates
were not separable by vote score.

The subjects say why. The held-out exam asked about `behind`, `curious`, `experience`, `hot`,
`comedy`, `fantastic`, `united`, `coast`. "The behind was ..." has no answer, so no rule of
aggregation could be right about it. 279's write decision was extracting ADJACENT CAPITALISED
WORDS, not assertions, and every measurement downstream inherited that.

The missing piece is a criterion for "is this an assertion at all", and writing one by hand is
just a bigger regex. So this measures it instead. Every extracted pair carries a FRAME - the
words standing between the anchor and the value, which is the relation the sentence was using -
and a frame can be judged by three statistics of the write journal, none of which reads a label:

  YIELD          a real relation gets independently restated, so its assertions reach CONFIRM.
                 An accidental adjacency is stated once by one source and never corroborated.
                 confirm_rate is that difference, and it is the core of the stage.

  GENERALITY     a relation applies to many subjects. A frame seen with one anchor is that
                 anchor's phrasing, not a relation.

  FUNCTIONALITY  a relation mostly maps one subject to one value. A frame where every anchor
                 carries five different values is an enumeration - "X , the Y ," - and this is
                 how a knowledge base tells a relation from a co-occurrence. Non-obvious and
                 cheap: mean distinct values per anchor, counted.

SKIP is then executable: write everything once, score the frames, drop the assertions whose frame
failed, write again. Nothing is fitted and nothing is labelled.

The fourth thing the frame buys is the one GOAL.md left open. Two assertions can only DISPUTE
each other if they share a frame. "Michael was born in X" and "Michael was appointed Y" are not a
disagreement, they are two facts, and counting them as witnesses of one address is exactly what
made 280's teacher answer everywhere and be wrong. --frame-in-address turns "sources disagree"
and "sources are talking about different things" into different situations mechanically.

The gate is cheap because it needs no training: the teacher runs alone on the rebuilt tape, and
if its ceiling clears half of silence then 280 is worth running again and otherwise it is not.

  python _stage281_frames.py --smoke
  python _stage281_frames.py --smoke --frame-in-address
  python _stage281_frames.py --addresses 400
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage279_write_decision as s279
import _stage280_raw_exam as s280
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 281
FAMILIES = s280.FAMILIES
LOG_PATH = RES / "_stage281_log.txt"


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def frame_of(a: dict) -> str:
    return (a["address"].split("|", 1) + [""])[1].strip()


def score_frames(asserts, bank, *, soft_match=0.0):
    """Three statistics per frame, all counted from the write journal and none from a label."""
    by_frame = defaultdict(list)
    for a in asserts:
        by_frame[frame_of(a)].append(a)

    stats = {}
    for fr, items in by_frame.items():
        # yield: write each frame's assertions on their own and see how often a second source
        # says the same thing at the same address. A relation is corroborated; an adjacency is not
        t = s279.Tape(bank, soft_match)
        for a in items:
            t.decide(a["address"], a["value"], a["source"])
        n = max(1, len(items))
        by_anchor = defaultdict(set)
        for a in items:
            by_anchor[a["address"].split("|", 1)[0]].add(a["value"])
        vpa = float(np.mean([len(v) for v in by_anchor.values()])) if by_anchor else 0.0
        stats[fr] = {
            "n": len(items),
            "confirm_rate": t.counts[s279.CONFIRM] / n,
            "dispute_rate": t.counts[s279.DISPUTE] / n,
            "anchors": len(by_anchor),
            "values_per_anchor": vpa,
            "empty": fr == "",
        }
    return stats


def cluster_frames(asserts, stats, *, min_jaccard: float, min_shared: int):
    """Two frames are the same relation if they connect the same pairs, not if they share words.

    Paraphrase, jargon and dialect give one relation many surface forms - "was born in", "b.",
    "a native of" - and a criterion that reads the words fragments the statistics across all of
    them and then throws each fragment away for being too small. The extensional test needs no
    word list and no encoder: collect the (subject, value) pairs each frame produces and merge
    frames whose pair sets overlap. Nothing here is English.
    """
    pairs = defaultdict(set)
    for a in asserts:
        anchor = a["address"].split("|", 1)[0]
        pairs[frame_of(a)].add((anchor, a["value"]))

    frames = sorted(pairs, key=lambda f: -len(pairs[f]))
    parent = {f: f for f in frames}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, f in enumerate(frames):
        for g in frames[:i]:
            inter = len(pairs[f] & pairs[g])
            if inter < min_shared:
                continue
            if inter / max(1, len(pairs[f] | pairs[g])) >= min_jaccard:
                parent[find(f)] = find(g)
                break

    groups = defaultdict(list)
    for f in frames:
        groups[find(f)].append(f)
    # a merged relation is judged as one, so its statistics are pooled rather than fragmented
    merged = {}
    for root, members in groups.items():
        n = sum(stats[m]["n"] for m in members)
        merged[root] = {
            "members": members, "n": n,
            "confirm_rate": sum(stats[m]["confirm_rate"] * stats[m]["n"] for m in members) / max(1, n),
            "anchors": len({p[0] for m in members for p in pairs[m]}),
            "values_per_anchor": float(np.mean([stats[m]["values_per_anchor"] for m in members])),
            "empty": all(stats[m]["empty"] for m in members),
        }
    return merged, {f: find(f) for f in frames}


def keep_set(stats, *, min_n, min_confirm, min_anchors, max_vpa, allow_empty):
    keep = set()
    for fr, s in stats.items():
        if s["empty"] and not allow_empty:
            continue                     # anchor and value adjacent: apposition, a NAME not a fact
        if s["n"] < min_n or s["anchors"] < min_anchors:
            continue
        if s["confirm_rate"] < min_confirm or s["values_per_anchor"] > max_vpa:
            continue
        keep.add(fr)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--addresses", type=int, default=0)
    ap.add_argument("--min-mentions", type=int, default=3)
    ap.add_argument("--min-n", type=int, default=4, help="assertions a frame needs to be judged")
    ap.add_argument("--min-confirm", type=float, default=0.15,
                    help="a frame whose assertions are never corroborated is an adjacency")
    ap.add_argument("--min-anchors", type=int, default=2,
                    help="a frame seen with one subject is that subject's phrasing")
    ap.add_argument("--max-values-per-anchor", type=float, default=2.5,
                    help="a relation is roughly functional; an enumeration is not")
    ap.add_argument("--cluster-frames", action="store_true",
                    help="merge frames that connect the same (subject, value) pairs before "
                         "judging them, so one relation stated three ways is judged once. Uses "
                         "no word list and no encoder.")
    ap.add_argument("--cluster-jaccard", type=float, default=0.30)
    ap.add_argument("--cluster-shared", type=int, default=2)
    ap.add_argument("--allow-empty-frame", action="store_true",
                    help="keep pairs where anchor and value are adjacent (apposition)")
    ap.add_argument("--frame-in-address", action="store_true",
                    help="two assertions may only dispute each other if they share a frame - the "
                         "difference between sources disagreeing and sources discussing "
                         "different things")
    ap.add_argument("--address-tau", type=float, default=0.90)
    ap.add_argument("--address-overlap", type=int, default=2)
    ap.add_argument("--soft-match", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--max-reads", type=int, default=7)
    ap.add_argument("--k-gap", type=float, default=0.35)
    ap.add_argument("--hop", choices=("none", "fp"), default="fp")
    args = ap.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()
    n_addr = args.addresses or (60 if args.smoke else 400)

    log(f"Stage281 frames start {datetime.now(timezone.utc).isoformat()} device={device} "
        f"frame_in_address={args.frame_in_address}")

    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    char_table = s185.build_char_table(tok, stoi, pad_id, V).to(device)
    model = SelfModelXL(n_char, V).to(device)
    model.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    bank = FpBank(model, stoi, device)

    with WIKI.open("r", encoding="utf-8", errors="ignore") as f:
        wtext = f.read(4_000_000 if args.smoke else 30_000_000)
    all_lines = [l.strip() for l in wtext.split("\n") if 80 <= len(l.strip()) <= 400]
    cut = int(0.7 * len(all_lines))
    train_lines = all_lines[:cut][: (3000 if args.smoke else 25000)]
    eval_lines = all_lines[cut:][: (1500 if args.smoke else 12000)]

    # ---- pass one: write everything, then judge the frames -----------------------------------
    common = s279.common_nouns(train_lines)
    asserts, _ = s279.corpus_assertions(train_lines, rng, n_addr, args.min_mentions,
                                        "anchor_rel", common=common)
    stats = score_frames(asserts, bank, soft_match=args.soft_match)
    cluster_of = None
    if args.cluster_frames:
        merged, cluster_of = cluster_frames(asserts, stats, min_jaccard=args.cluster_jaccard,
                                            min_shared=args.cluster_shared)
        log(f"  clustering: {len(stats)} frames -> {len(merged)} relations")
        kept_roots = keep_set(merged, min_n=args.min_n, min_confirm=args.min_confirm,
                              min_anchors=args.min_anchors,
                              max_vpa=args.max_values_per_anchor,
                              allow_empty=args.allow_empty_frame)
        keep = {f for f, root in cluster_of.items() if root in kept_roots}
    else:
        keep = keep_set(stats, min_n=args.min_n, min_confirm=args.min_confirm,
                        min_anchors=args.min_anchors, max_vpa=args.max_values_per_anchor,
                        allow_empty=args.allow_empty_frame)
    kept_asserts = sum(s["n"] for fr, s in stats.items() if fr in keep)
    log(f"  pass 1: {len(asserts)} assertions over {len(stats)} frames "
        f"-> keep {len(keep)} frames / {kept_asserts} assertions "
        f"({kept_asserts / max(1, len(asserts)):.3f}) ({time.time()-t0:.0f}s)")
    top = sorted(((s["confirm_rate"], fr, s) for fr, s in stats.items() if s["n"] >= args.min_n),
                 reverse=True)
    # label by the actual keep set, not by rank: the first version printed the top eight as
    # KEEP whatever their fate, and 'lder' with one anchor was shown as kept while it was dropped
    for cr, fr, s in top[:12]:
        log(f"    {'KEEP' if fr in keep else 'drop'}  '{fr}' n={s['n']} confirm={cr:.2f} "
            f"anchors={s['anchors']} vpa={s['values_per_anchor']:.2f}")
    if not keep:
        log("  no frame survived; loosen --min-confirm or --min-n")
        return 1

    # ---- pass two: rebuild the tape from surviving frames only -------------------------------
    def build(keep_frames):
        # same held-out lines and the same seed on both arms, so the only difference between
        # before and after is which frames were allowed to write
        return s280.pack_from_corpus(
            eval_lines, bank=bank, tok=tok, pad_id=pad_id, device=device,
            rng=random.Random(SEED + 9), n_addr=n_addr, min_mentions=args.min_mentions,
            tau=args.address_tau, overlap=args.address_overlap,
            soft_match=args.soft_match, keep_frames=keep_frames)

    packs = {"before": build(None), "after": build(keep)}

    # ---- the gate: the teacher alone, no training at all -------------------------------------
    @torch.no_grad()
    def ceiling(pack):
        if len(pack["items"]) < 6:
            return {"n_items": len(pack["items"]), "reward": float("nan")}
        per = {f: defaultdict(list) for f in FAMILIES}
        common_kw = dict(k=args.topk, max_steps=args.max_steps, max_reads=args.max_reads,
                         read_cost=0.02, wrong_cost=1.0, abstain_reward=0.75,
                         subject_filter=True, hop=args.hop, hop_min=1.0, k_gap=args.k_gap)
        rewards = []
        for it in pack["items"]:
            t = s280.rollout(None, model, char_table, tok, pack, it, pad_id, device,
                             teacher_only=True, **common_kw)
            per[it["kind"]]["correct"].append(t["correct"])
            per[it["kind"]]["abstain"].append(int(t["abstained"]))
            rewards.append(t["reward"])
        m = lambda xs: float(np.mean(xs)) if xs else float("nan")
        out = {"n_items": len(pack["items"]), "reward": m(rewards),
               "addresses": pack["n_addresses"], "slots": pack["n_slots"],
               "families": dict(Counter(i["kind"] for i in pack["items"]))}
        for f in FAMILIES:
            out[f] = {"n": len(per[f]["abstain"]), "teacher_acc": m(per[f]["correct"]),
                      "teacher_abstain": m(per[f]["abstain"])}
        return out

    before = ceiling(packs["before"])
    after = ceiling(packs["after"])
    log("  ceiling BEFORE frames: " + json.dumps(before))
    log("  ceiling AFTER  frames: " + json.dumps(after))

    silence = 0.75
    g_frames = len(keep) > 0
    g_shrinks = kept_asserts < len(asserts)
    g_functional = all(stats[fr]["values_per_anchor"] <= args.max_values_per_anchor
                       for fr in keep)
    g_ceiling = after["reward"] >= 0.5 * silence
    g_improves = after["reward"] > before["reward"] + 0.10
    g_tie_abstain = after.get("tie", {}).get("teacher_abstain", 0.0) >= 0.5

    if not g_frames:
        overall = "NO_FRAME_SURVIVES"
    elif g_ceiling and g_tie_abstain:
        overall = "FRAMES_MAKE_THE_EXAM_SOUND"    # 280 is worth running again
    elif g_improves:
        overall = "FRAMES_HELP_NOT_ENOUGH"
    else:
        overall = "FRAMES_DO_NOT_HELP"            # the wall is elsewhere; say so plainly

    out = {
        "stage": 281, "overall": overall, "trained_parameters": 0, "smoke": args.smoke,
        "seed": SEED, "frame_in_address": args.frame_in_address,
        "clustered": bool(args.cluster_frames),
        "thresholds": {"min_n": args.min_n, "min_confirm": args.min_confirm,
                       "min_anchors": args.min_anchors,
                       "max_values_per_anchor": args.max_values_per_anchor,
                       "allow_empty_frame": args.allow_empty_frame},
        "frames": {"total": len(stats), "kept": len(keep),
                   "assertions_total": len(asserts), "assertions_kept": kept_asserts,
                   "kept_fraction": kept_asserts / max(1, len(asserts)),
                   "top": [{"frame": fr, **s} for _, fr, s in top[:20]],
                   "worst": [{"frame": fr, **s} for _, fr, s in top[-10:]]},
        "gates": {
            "G_frames_survive": g_frames, "G_tape_shrinks": g_shrinks,
            "G_kept_frames_functional": g_functional,
            "G_ceiling_clears_silence": g_ceiling,
            "G_ceiling_improves": g_improves,
            "G_teacher_abstains_on_tie": g_tie_abstain,
        },
        "ceiling_before": before, "ceiling_after": after,
        "reference_280": {"teacher_ceiling_reward": -0.18888888888888897,
                          "held_out_precision": 0.3412698412698412},
        "note": (
            "280's gate caught a wall its own numbers explain: the held-out exam asked about "
            "behind, curious, experience and coast, and no rule of aggregation can be right "
            "about a question with no answer. 279 was extracting adjacent capitalised words "
            "rather than assertions. Writing a better regex would only move the problem, so the "
            "criterion is measured instead. Each pair carries the frame that stood between "
            "anchor and value, and a frame is judged by three statistics of the write journal, "
            "none of which reads a label: how often its assertions are corroborated by a second "
            "source, how many distinct subjects it applies to, and how nearly functional it is - "
            "a relation maps one subject to about one value, an enumeration does not. SKIP is "
            "then executable. --frame-in-address additionally requires two assertions to share a "
            "frame before they may dispute, which is the distinction GOAL.md left open between "
            "sources disagreeing and sources discussing different things. The gate needs no "
            "training at all: the teacher runs alone on the rebuilt tape, and 280 is worth "
            "repeating only if its ceiling clears half the value of unconditional silence."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(), "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    tag = "_fia" if args.frame_in_address else ""
    (RES / f"stage281_decision{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (RES / f"stage281_mini{tag}.md").write_text(
        f"# Stage 281 what counts as an assertion\n\n**{overall}**"
        f"{' · SMOKE' if args.smoke else ''} · trained parameters **0**\n\n"
        f"- frames {len(keep)}/{len(stats)} kept, assertions {kept_asserts}/{len(asserts)} "
        f"({kept_asserts / max(1, len(asserts)):.1%})\n"
        f"- teacher ceiling **{before['reward']:.3f} -> {after['reward']:.3f}** "
        f"(silence pays 0.750; 280 measured -0.189)\n"
        f"- tie abstention by the teacher: "
        f"{before.get('tie', {}).get('teacher_abstain', float('nan')):.2f} -> "
        f"{after.get('tie', {}).get('teacher_abstain', float('nan')):.2f}\n\n"
        f"| frame | n | confirm | anchors | values/anchor |\n|---|---:|---:|---:|---:|\n"
        + "".join(f"| `{fr}` | {s['n']} | {s['confirm_rate']:.2f} | {s['anchors']} | "
                  f"{s['values_per_anchor']:.2f} |\n" for _, fr, s in top[:10])
        + "\n## Gates\n\n"
        + "".join(f"- {k}: **{v}**\n" for k, v in out["gates"].items()),
        encoding="utf-8",
    )
    log(json.dumps({"overall": overall, "gates": out["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
