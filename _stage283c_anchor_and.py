"""
Stage 283c — Anchor-block write with AND + relation overlap.

283b scoped soft merge to one anchor and killed cross-entity foreign (0.0, flat), but OR
linking on full context glued distinct relations of the same name → tie ≈ 0.70.

283c keeps the hard block. Inside the block the edge is conjunction: relation-tail overlap
>= min_overlap AND two-channel cos >= tau=0.90. Same numbers as write2; no tau sweep. Overlap
reads the address relation (`anchor|rel`), not ambient context English.

  python _stage283c_anchor_and.py --smoke
  python _stage283c_anchor_and.py --rungs 4M:400 30M:400 120M:2000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer

import _stage177_curve_bpe as s177
import _stage185_tape_read as s185
import _stage271_controller as s271
import _stage280_raw_exam as s280
import _stage283_scale as s283
from _stage191_night import PAD, SelfModelXL, load_data
from _stage194_fp_fact_memory import FpBank

RES = Path("results")
CKPT_P1 = Path("checkpoints/stage191_p1_curve.pt")
WIKI = Path("data/_wikitext103_train.txt")
SEED = 283
LOG_PATH = RES / "_stage283c_log.txt"
MERGE_MODE = "anchor_block_and"


def log(m: str) -> None:
    line = m if m.endswith("\n") else m + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rungs", nargs="*", default=[])
    ap.add_argument("--corpus", type=Path, default=WIKI)
    ap.add_argument("--min-mentions", type=int, default=2)
    ap.add_argument("--address-tau", type=float, default=0.90,
                    help="fixed; 283c does not sweep this")
    ap.add_argument("--address-overlap", type=int, default=2)
    ap.add_argument("--soft-match", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=7)
    ap.add_argument("--hop", choices=("none", "fp"), default="fp")
    ap.add_argument("--hop-min", type=float, default=1.0)
    ap.add_argument("--k-gap", type=float, default=0.35)
    ap.add_argument("--subject-filter", choices=("off", "on"), default="on")
    ap.add_argument("--addr-key", choices=("two", "set", "mean"), default="two")
    ap.add_argument("--min-precision", type=float, default=0.60)
    ap.add_argument("--min-separation", type=float, default=0.30)
    ap.add_argument("--max-crowding", type=float, default=0.20)
    ap.add_argument("--max-foreign", type=float, default=0.10)
    args = ap.parse_args()

    global LOG_PATH
    LOG_PATH = RES / "_stage283c_log.txt"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    rungs = s283.parse_rungs(args.rungs) if args.rungs else (
        [(2_000_000, 100), (4_000_000, 200)] if args.smoke
        else [(4_000_000, 400), (30_000_000, 400), (120_000_000, 2000)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    _, _, stoi, n_char = load_data()
    tok = Tokenizer.from_file(str(s177.TOK_PATH))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id(PAD) or 0
    s185.build_char_table(tok, stoi, pad_id, V)
    can = SelfModelXL(n_char, V).to(device)
    can.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=False)["model"])
    can.eval()
    for p in can.parameters():
        p.requires_grad_(False)
    bank = FpBank(can, stoi, device)
    mind_params = sum(p.numel() for p in can.parameters())

    log(f"Stage283c anchor-block AND start {datetime.now(timezone.utc).isoformat()} "
        f"device={device} rungs={rungs} mind_params={mind_params} merge_mode={MERGE_MODE}")

    if not args.corpus.exists():
        log(f"  corpus not found: {args.corpus}")
        return 1
    with args.corpus.open("r", encoding="utf-8", errors="ignore") as f:
        full = f.read(max(c for c, _ in rungs))
    log(f"  corpus {len(full)/1e6:.1f}M chars read ({time.time()-t0:.0f}s)")

    out_rungs = []
    for chars, n_addr in rungs:
        t1 = time.time()
        rng = random.Random(SEED)
        lines = [l.strip() for l in full[:chars].split("\n") if 80 <= len(l.strip()) <= 400]
        pack = s280.pack_from_corpus(
            lines, bank=bank, tok=tok, pad_id=pad_id, device=device, rng=rng, n_addr=n_addr,
            min_mentions=args.min_mentions, tau=args.address_tau,
            overlap=args.address_overlap, soft_match=args.soft_match,
            min_per_family=0, addr_key=args.addr_key, merge_mode=MERGE_MODE)
        build_s = time.time() - t1
        if not pack["items"]:
            log(f"  rung {chars}:{n_addr} produced no items, skipped")
            continue
        r = s283.measure(pack, bank, rng, args.topk, args.hop, args.hop_min, args.k_gap,
                         args.subject_filter == "on", args.address_tau)
        r.update({"chars": chars, "n_addr_requested": n_addr, "build_s": build_s,
                  "mind_params": mind_params, "lines": len(lines),
                  "merge_mode": MERGE_MODE})
        out_rungs.append(r)
        log(f"  rung {chars/1e6:.0f}M:{n_addr} -> {r['n_addresses']} addr / {r['n_slots']} slots"
            f" | precision {r['retrieval_precision']:.3f}"
            f" | return true {r['return_path_true']:.3f} other {r['return_path_other']:.3f}"
            f" (sep {r['return_path_separation']:+.3f})"
            f" | crowding {r['address_crowding']:.3f}"
            f" (members {r['member_crowding']:.3f} two {r['two_channel_crowding']:.3f}"
            f" anchors alone {r['anchor_only_crowding']:.3f})"
            f" foreign {r['foreign_member_rate']:.3f}"
            f" | k covers {r['k_covers_address']:.3f}"
            f" | ties {r['families_natural']['tie']:.3f}"
            f" clean {r['families_natural']['clean']:.3f} ({build_s:.0f}s)")

    if not out_rungs:
        log("  nothing measured")
        return 1
    last = out_rungs[-1]
    first = out_rungs[0]
    g_precision = last["retrieval_precision"] >= args.min_precision
    g_separation = last["return_path_separation"] >= args.min_separation
    g_crowding = last["foreign_member_rate"] <= args.max_foreign
    g_ties = last["families_natural"]["tie"] > 0.0
    g_mind_constant = all(r["mind_params"] == mind_params for r in out_rungs)

    overall = ("SCALE_OK" if all((g_precision, g_separation, g_crowding, g_ties))
               else "SCALE_PARTIAL" if (g_precision and g_ties) else "SCALE_NO")

    out = {
        "stage": "283c", "overall": overall, "seed": SEED, "smoke": args.smoke,
        "corpus": str(args.corpus), "merge_mode": MERGE_MODE,
        "link": "and", "overlap_from": "rel",
        "addr_key": args.addr_key,
        "address": {"tau": args.address_tau, "overlap": args.address_overlap,
                    "soft_match": args.soft_match, "min_mentions": args.min_mentions},
        "retrieval": {"topk": args.topk, "hop": args.hop, "hop_min": args.hop_min,
                      "k_gap": args.k_gap, "subject_filter": args.subject_filter},
        "thresholds": {"min_precision": args.min_precision,
                       "min_separation": args.min_separation,
                       "max_crowding": args.max_crowding,
                       "max_foreign": args.max_foreign},
        "mind_params": mind_params,
        "gates": {
            "G_precision_holds": g_precision,
            "G_return_path_separates": g_separation,
            "G_addresses_stay_distinct": g_crowding,
            "G_k_covers_the_address": last["k_covers_address"] >= 0.90,
            "G_ties_exist_naturally": g_ties,
            "G_mind_does_not_grow": g_mind_constant,
        },
        "slopes": {
            "slots": [r["n_slots"] for r in out_rungs],
            "precision": [r["retrieval_precision"] for r in out_rungs],
            "return_path_separation": [r["return_path_separation"] for r in out_rungs],
            "address_crowding": [r["address_crowding"] for r in out_rungs],
            "member_crowding": [r["member_crowding"] for r in out_rungs],
            "two_channel_crowding": [r["two_channel_crowding"] for r in out_rungs],
            "foreign_member_rate": [r["foreign_member_rate"] for r in out_rungs],
            "distinct_anchors": [r["distinct_anchors"] for r in out_rungs],
            "anchor_only_crowding": [r["anchor_only_crowding"] for r in out_rungs],
            "anchor_growth_exponent": (
                math.log(max(1, last["distinct_anchors"]) / max(1, first["distinct_anchors"]))
                / math.log(last["chars"] / max(1, first["chars"]))
                if len(out_rungs) > 1 and last["chars"] != first["chars"] else float("nan")),
            "k_covers_address": [r["k_covers_address"] for r in out_rungs],
            "tie_share": [r["families_natural"]["tie"] for r in out_rungs],
            "clean_share": [r["families_natural"]["clean"] for r in out_rungs],
            "slots_growth": last["n_slots"] / max(1, first["n_slots"]),
        },
        "rungs": out_rungs,
        "fp_version": s271.fp_version(),
        "reference_283b": {
            "n_addresses": 2000, "n_slots": 11074,
            "foreign_member_rate": 0.0, "tie": 0.6955, "clean": 0.126,
        },
        "reference_write2": {
            "n_addresses": 877, "n_slots": 2765,
            "foreign_member_rate": 0.00860178354884333,
            "tie": 0.34891676168757124, "clean": 0.5245153933865451,
        },
        "reference_geom": {
            "n_addresses": 936, "n_slots": 4730,
            "foreign_member_rate": 0.05161396505477364,
        },
        "note": (
            "Letter step 283c: hard anchor block as in 283b; inside the block, edge = "
            "relation-tail overlap AND two-channel cos>=tau. Fixes 283b's tie explosion from "
            "OR+ctx overlap without retuning tau. Pass if foreign stays flat/near-zero, tie "
            "falls toward write2, slots stay above write2."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(), "wall_s": time.time() - t0,
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "stage283c_decision.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(json.dumps({"overall": overall, "gates": out["gates"], "slopes": out["slopes"]},
                   indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
