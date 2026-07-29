"""
Stage 134 — codebook-as-tokenizer + hops-as-assist (eval-only).

Architecture under test (practical compromise post-133):
  text → word_fp / codebook id   (= tokenizer)
       → frozen CE next-id head  (= single GPT-like trunk)
       → decode D
  hops / SoftPhraseMemory        (= optional help + separate fact API)

NOT tested here:
  - hops as primary arbiter among top5 (that was 133; PARITY / worse)
  - co-training hops into CE (117/118/120)

Modes:
  A           atom argmax always
  B_override  on RIGHT only: if hop confident → override; else atom
  B_bias      on RIGHT only: soft logit boost for hop cands; else atom
  C           fact API (lookup + retrieve) — separate metric, not STORY gate

Gate (exact@1):
  PRIMARY_HELD  if |B_obj − A_obj| ≤ 1.5pp and atom rel ≥ 0.70
  ASSIST_HARM   if best B drops SEEN obj by > 3pp
  FACT_API_OK   if bank retrieve@1 ≥ 0.80 on known (left,rel) cues
  Verdict PASS  if PRIMARY_HELD and FACT_API_OK
        PARTIAL if PRIMARY_HELD xor FACT_API_OK (or mild assist swing)
        PARITY  otherwise (incl. assist flat + weak facts)

Run:
  python _stage134_codebook_tok_hops_assist.py
"""
from __future__ import annotations

import json
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    CKPT,
    FACT_BANK_EXP_F,
    RELS,
    RES,
    Config,
    WordIdTransformer,
    _role,
    build_ts_repeat_mix,
    eval_id_capacity_suite,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
)
from _stage109_110_slot_baseline import ensure_100k, _subsample  # noqa: E402
from _stage116_117_hops_morph_follow import build_path_memory  # noqa: E402

ATOM125 = CKPT / "stage125_same_role_rank.pt"
ATOM100 = CKPT / "stage100_scale_100k.pt"
LOG = RES / "_stage134_log.txt"
DEC = RES / "stage134_codebook_tok_hops_assist_decision.json"

# assist knobs (eval-only; not a grid search)
TAU_MEM = 0.55          # min SoftPhraseMemory sim to fire override
MARGIN = 0.05           # best − 2nd mem sim among hop rights
BIAS_ALPHA = 4.0        # logit add = alpha * mem_sim
MAX_HOP_RIGHTS = 8


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def safe_phrase_fp(stack, ws):
    mw = max(1, int(getattr(stack.phrase, "max_len", 5)) - 1)
    ws = list(ws)[-mw:]
    if not ws:
        return None
    try:
        return stack.phrase_fp(ws)
    except Exception:
        return None


@torch.no_grad()
def mem_score_right(mem, cue_fp, right_word):
    """Best SoftPhraseMemory sim among slots whose meta.right == right_word."""
    if cue_fp is None or not mem.slots:
        return None
    best = None
    for name, sim in mem.topk(cue_fp, k=min(32, len(mem.slots))):
        meta = mem.fact_meta.get(name) or {}
        if meta.get("right") == right_word:
            s = float(sim)
            best = s if best is None else max(best, s)
    return best


def hop_rights_for_prefix(ws_prefix, by_lr, by_rel):
    """Bank rights for (left,rel); fall back to by_rel if empty."""
    if not ws_prefix or ws_prefix[-1] not in RELS:
        return []
    rel = ws_prefix[-1]
    rights = []
    if len(ws_prefix) >= 2:
        rights = sorted(by_lr.get((ws_prefix[-2], rel), set()))
    if not rights:
        rights = sorted(by_rel.get(rel, set()))[:MAX_HOP_RIGHTS]
    return rights[:MAX_HOP_RIGHTS]


def assist_decision(stack, mem, by_lr, by_rel, ws_prefix, stoi, words):
    """
    Returns dict:
      confident: bool
      pick_id: int | None   (vocab id if confident)
      scores: {wid: mem_sim}
      reason: str
    Only meaningful when prefix ends in REL (object slot).
    """
    empty = {"confident": False, "pick_id": None, "scores": {}, "reason": "no_rel"}
    if not ws_prefix or ws_prefix[-1] not in RELS:
        return empty
    rights = hop_rights_for_prefix(ws_prefix, by_lr, by_rel)
    if not rights:
        return {**empty, "reason": "no_bank"}

    left = ws_prefix[-2] if len(ws_prefix) >= 2 else ""
    rel = ws_prefix[-1]
    cue = safe_phrase_fp(stack, [left, rel] if left else [rel])
    if cue is None:
        cue = safe_phrase_fp(stack, ws_prefix[-3:])

    scores = {}
    for r in rights:
        if r not in stoi:
            continue
        s = mem_score_right(mem, cue, r)
        if s is None:
            # exact bank membership without retrieve hit — weak prior
            s = 0.35 if (left, rel) in by_lr and r in by_lr[(left, rel)] else 0.0
        scores[stoi[r]] = float(s)

    if not scores:
        return {**empty, "reason": "rights_oov"}

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_id, best_s = ranked[0]
    second_s = ranked[1][1] if len(ranked) > 1 else -1.0

    # unique exact (left,rel) → strong confidence even if mem soft
    unique = (
        len(ws_prefix) >= 2
        and len(by_lr.get((left, rel), set())) == 1
        and words[best_id] in by_lr.get((left, rel), set())
    )
    mem_ok = best_s >= TAU_MEM and (best_s - second_s) >= MARGIN
    confident = bool(unique or mem_ok)
    reason = "unique_lr" if unique else ("mem_tau" if mem_ok else "below_tau")
    return {
        "confident": confident,
        "pick_id": best_id if confident else None,
        "scores": scores,
        "reason": reason,
        "best_sim": best_s,
    }


@torch.no_grad()
def eval_lm_modes(model, stack, mem, by_lr, by_rel, hold, words, stoi, device):
    """
    A / B_override / B_bias on same hold.
    Assist only on role==right; elsewhere always atom.
    """
    modes = ("A", "B_override", "B_bias")
    stats = {
        m: {
            "n": 0,
            "h": 0,
            "n_right": 0,
            "h_right": 0,
            "assist_fire": 0,
            "assist_correct": 0,
            "assist_disagree_atom": 0,
        }
        for m in modes
    }

    pairs = lines_to_pairs(hold, stoi)
    for ex in pairs:
        pl = int(ex["prefix_len"])
        if pl < 1:
            continue
        ws = hold[ex["line_i"]]["words"]
        gold = ex["target_word"]
        if gold not in stoi:
            continue
        role = _role(ex, ws)
        gid = stoi[gold]
        ids = list(ex["prefix_word_ids"][-model.max_len :])
        logits = model.logits_from_prefix(ids)
        atom_id = int(logits.argmax())

        preds = {"A": atom_id}

        if role == "right":
            dec = assist_decision(stack, mem, by_lr, by_rel, ws[:pl], stoi, words)
            # B_override
            if dec["confident"] and dec["pick_id"] is not None:
                preds["B_override"] = int(dec["pick_id"])
                stats["B_override"]["assist_fire"] += 1
                if preds["B_override"] == gid:
                    stats["B_override"]["assist_correct"] += 1
                if preds["B_override"] != atom_id:
                    stats["B_override"]["assist_disagree_atom"] += 1
            else:
                preds["B_override"] = atom_id

            # B_bias
            logits_b = logits.clone()
            if dec["scores"]:
                for wid, sim in dec["scores"].items():
                    logits_b[wid] = logits_b[wid] + BIAS_ALPHA * float(sim)
                stats["B_bias"]["assist_fire"] += 1
            pred_b = int(logits_b.argmax())
            preds["B_bias"] = pred_b
            if dec["scores"] and pred_b == gid:
                stats["B_bias"]["assist_correct"] += 1
            if pred_b != atom_id:
                stats["B_bias"]["assist_disagree_atom"] += 1
        else:
            preds["B_override"] = atom_id
            preds["B_bias"] = atom_id

        for m in modes:
            ok = int(preds[m] == gid)
            stats[m]["n"] += 1
            stats[m]["h"] += ok
            if role == "right":
                stats[m]["n_right"] += 1
                stats[m]["h_right"] += ok

    out = {}
    for m, d in stats.items():
        n = max(d["n"], 1)
        nr = max(d["n_right"], 1)
        af = d["assist_fire"]
        out[m] = {
            "all_hit1": d["h"] / n,
            "n": d["n"],
            "obj_hit1": d["h_right"] / nr,
            "n_right": d["n_right"],
            "assist_fire_on_right": af / nr if m != "A" else 0.0,
            "assist_precision_when_fire": (d["assist_correct"] / af) if af else None,
            "assist_disagree_atom_frac": (d["assist_disagree_atom"] / nr) if m != "A" else 0.0,
        }
    return out


@torch.no_grad()
def eval_fact_api(stack, mem, by_lr, device, max_keys=400):
    """
    C — separate fact/continuity API (not LM STORY gate).
      lookup: exact (left,rel) → rights non-empty
      retrieve@1: cue=phrase_fp([left,rel]) → top1 slot right matches any gold right
    """
    keys = list(by_lr.keys())
    if len(keys) > max_keys:
        # stable subsample
        keys = sorted(keys, key=lambda k: (k[0], k[1]))[:max_keys]

    n = 0
    lookup_ok = 0
    ret1 = 0
    ret5 = 0
    for left, rel in keys:
        rights = by_lr[(left, rel)]
        if not rights:
            continue
        n += 1
        lookup_ok += 1  # key exists by construction
        cue = safe_phrase_fp(stack, [left, rel])
        if cue is None or not mem.slots:
            continue
        hits = mem.topk(cue, k=min(5, len(mem.slots)))
        tops = []
        for name, _sim in hits:
            meta = mem.fact_meta.get(name) or {}
            r = meta.get("right")
            if r:
                tops.append(r)
        if tops and tops[0] in rights:
            ret1 += 1
        if any(r in rights for r in tops):
            ret5 += 1

    return {
        "n_keys": n,
        "lookup_coverage": lookup_ok / max(n, 1),
        "retrieve_hit1": ret1 / max(n, 1),
        "retrieve_hit5": ret5 / max(n, 1),
        "n_mem_slots": len(mem.slots),
        "tau_mem": TAU_MEM,
        "bias_alpha": BIAS_ALPHA,
    }


def verdict_of(a_obj, b_best_obj, rel, fact):
    held = abs(b_best_obj - a_obj) <= 0.015 and rel >= 0.70
    harm = (b_best_obj < a_obj - 0.03) and rel >= 0.0
    fact_ok = fact["retrieve_hit1"] >= 0.80
    if harm:
        return "ASSIST_HARM"
    if held and fact_ok:
        return "PASS"
    if held or fact_ok:
        return "PARTIAL"
    return "PARITY"


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"134 codebook-tok + hops-assist start {datetime.now(timezone.utc).isoformat()}")
    log("Protocol: A=atom | B_override/B_bias=hops help on RIGHT | C=fact API")
    log("Train: none (eval-only). Codebook=tokenizer; CE trunk frozen.")
    try:
        phrases = ensure_100k()
        cfg = Config()
        cfg.c87_n_fat, cfg.c87_n_rare = 200, 80
        cfg.c87_fat_copies, cfg.c87_rare_copies = 40, 2
        cfg.c87_seen_hold_frac, cfg.c87_story_keep_frac, cfg.c87_hold_frac = 0.2, 0.5, 0.15
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        parent = CKPT / "stage85_foundation_FROZEN.pt"
        if not parent.exists():
            parent = CKPT / "stage85_foundation.pt"
        stack, _ = load_foundation_85(device, cfg, path=parent)
        for mod in (stack.encoder, stack.composer, stack.morph, stack.phrase, stack.binder):
            for p in mod.parameters():
                p.requires_grad_(False)
            mod.eval()

        train, hold_seen, _, hold_story, _ = build_ts_repeat_mix(phrases, cfg, seed=272)
        for ln in train + hold_seen + hold_story:
            ln.setdefault("subkind", line_subkind(ln))
            ln["split"] = ln.get("bucket", "x")
        words = sorted({w for ln in train + hold_seen + hold_story for w in ln["words"]})
        stoi = {s: i for i, s in enumerate(words)}

        model = WordIdTransformer(len(words), 256, 4, 2, 16, 0.1).to(device)
        path = ATOM125 if ATOM125.exists() else ATOM100
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert list(ck["surfaces"]) == words, "atom ckpt surfaces mismatch — need same 100k mix seed"
        model.load_state_dict(ck["word_tf"], strict=True)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        log(f"[init] frozen atom (CE trunk) {path.name} | codebook V={len(words)}")

        log("[mem] build SoftPhraseMemory + (left,rel) index ...")
        mem, by_lr, by_rel = build_path_memory(
            stack, cfg, train, device,
            fact_path=FACT_BANK_EXP_F if FACT_BANK_EXP_F.exists() else None,
        )

        ev_seen = _subsample(hold_seen, 600, 1341)
        ev_story = _subsample(hold_story, 400, 1343)

        log("\n======== A / B_override / B_bias (LM exact@1) ========")
        seen = eval_lm_modes(model, stack, mem, by_lr, by_rel, ev_seen, words, stoi, device)
        story = eval_lm_modes(model, stack, mem, by_lr, by_rel, ev_story, words, stoi, device)
        for tag, blk in (("SEEN", seen), ("STORY", story)):
            log(
                f"  [{tag}] A_obj={100*blk['A']['obj_hit1']:.1f}% "
                f"B_ov={100*blk['B_override']['obj_hit1']:.1f}% "
                f"B_bias={100*blk['B_bias']['obj_hit1']:.1f}% | "
                f"fire_ov={100*(blk['B_override']['assist_fire_on_right'] or 0):.1f}% "
                f"fire_bias={100*(blk['B_bias']['assist_fire_on_right'] or 0):.1f}%"
            )

        log("\n======== C fact API (not STORY gate) ========")
        fact = eval_fact_api(stack, mem, by_lr, device)
        log(
            f"  retrieve@1={100*fact['retrieve_hit1']:.1f}% "
            f"retrieve@5={100*fact['retrieve_hit5']:.1f}% "
            f"keys={fact['n_keys']} slots={fact['n_mem_slots']}"
        )

        suite = eval_id_capacity_suite(model, ev_seen, train, words, stoi, device)
        rel = suite["roles"].get("rel", {}).get("hit1", 0.0)

        a_obj = seen["A"]["obj_hit1"]
        b_best = max(seen["B_override"]["obj_hit1"], seen["B_bias"]["obj_hit1"])
        lift = b_best - a_obj
        verdict = verdict_of(a_obj, b_best, rel, fact)

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "codebook_tokenizer_CE_trunk_hops_assist",
            "atom_ckpt": path.name,
            "knobs": {"tau_mem": TAU_MEM, "margin": MARGIN, "bias_alpha": BIAS_ALPHA},
            "seen": seen,
            "story": story,
            "fact_api": fact,
            "rel_atom": rel,
            "seen_obj": {
                "A": a_obj,
                "B_override": seen["B_override"]["obj_hit1"],
                "B_bias": seen["B_bias"]["obj_hit1"],
                "best_B": b_best,
                "lift_vs_A_pp": lift,
            },
            "story_obj": {
                "A": story["A"]["obj_hit1"],
                "B_override": story["B_override"]["obj_hit1"],
                "B_bias": story["B_bias"]["obj_hit1"],
            },
            "verdict": verdict,
            "note": (
                "Primary train signal = single CE trunk (frozen). "
                "Hops = conditional assist on RIGHT + separate fact API. "
                "STORY lift is NOT the success claim for hops."
            ),
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        (RES / f"stage134_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")

        log(
            f"[134] {verdict} SEEN A={100*a_obj:.1f}% best_B={100*b_best:.1f}% "
            f"lift={100*lift:+.1f}pp | fact@1={100*fact['retrieve_hit1']:.1f}% rel={100*rel:.1f}%"
        )

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stage 134 codebook-tok + hops-assist (eval):** {verdict} "
                f"SEEN A={100*a_obj:.1f}% best_B={100*b_best:.1f}% "
                f"fact@1={100*fact['retrieve_hit1']:.1f}%. "
                f"`stage134_codebook_tok_hops_assist_decision.json`.\n"
            )
            if "Stage 134 codebook" not in txt:
                if "**F85 dual-channel FREEZE:**" in txt:
                    txt = txt.replace(
                        "**F85 dual-channel FREEZE:**",
                        block + "\n**F85 dual-channel FREEZE:**",
                    )
                else:
                    txt = txt.rstrip() + "\n" + block
                replay.write_text(txt, encoding="utf-8")
        log("DONE 134")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
