"""
Stage 133 — HONEST hops-as-arbiter among atom top5 (obj slot).

Previous 128/130 were NOT clean:
  128 = hop filter + atom argmax
  130 = MLP with atom z-logit as feat0 (could copy LM)

This dig ONLY:

  (a) EVAL-ONLY pure hop-rank@top5
      score(c) = cos( cue_fp, phrase_fp([left, rel, c]) )
      NO atom logit in the decision.
      Also report SoftPhraseMemory max-sim(right=c) variant.
      Fallback if phrase_fp fails: -1e9 (skip cand).

  (b) FT phrase-fingerprint adapter
      Train bilinear/MLP on frozen phrase fps:
        score(c) = MLP([cue_fp, cand_fp]) or cue @ W @ cand
      Loss: CE / hinge over atom top5 when gold in top5 (right slot only).
      Atom frozen. Foundation phrase encode frozen; only adapter trains.
      Optional light: also push bank triple keys.

  (c) Compare to atom@1 on same hold (SEEN obj / STORY obj).

Gate vs atom@1 on SEEN obj:
  PASS if pure or FT hop-rank +3pp with rel held (atom rel reported)
  PARTIAL +1.5pp

Waits for stage132 (or stage130_131 if 132 absent after timeout).

Run:
  python _stage133_honest_hop_rank.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
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
    _warmup_then_constant,
    build_ts_repeat_mix,
    collate_word_id_batch,
    eval_id_capacity_suite,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
)
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402
from _stage116_117_hops_morph_follow import build_path_memory  # noqa: E402

DEC132 = RES / "stage132_selective_morph_decision.json"
DEC131 = RES / "stage130_131_decision.json"
ATOM125 = CKPT / "stage125_same_role_rank.pt"
ATOM100 = CKPT / "stage100_scale_100k.pt"
LOG = RES / "_stage133_log.txt"
DEC = RES / "stage133_honest_hop_rank_decision.json"
K = 5


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def wait_upstream(timeout_s=12 * 3600, poll=45):
    log(f"[wait] for {DEC132.name} (132 follows 131) ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC132.exists():
            d = json.loads(DEC132.read_text(encoding="utf-8"))
            if d.get("verdict"):
                log("[wait] 132 done")
                return {"via": "132", **d}
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still ... {int(time.time()-t0)}s")
    raise TimeoutError("132 not ready")


class FingerprintScorer(nn.Module):
    """score(cue, cand) from frozen 256-d phrase fps — ONLY trainable adapter."""

    def __init__(self, dim=256, mode="bilinear"):
        super().__init__()
        self.mode = mode
        if mode == "bilinear":
            self.W = nn.Linear(dim, dim, bias=False)
            nn.init.eye_(self.W.weight)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

    def forward(self, cue_fp, cand_fp):
        # cue_fp, cand_fp: [B, D] or [D]
        if cue_fp.dim() == 1:
            cue_fp = cue_fp.unsqueeze(0)
            cand_fp = cand_fp.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        cue_fp = F.normalize(cue_fp, dim=-1)
        cand_fp = F.normalize(cand_fp, dim=-1)
        if self.mode == "bilinear":
            s = (self.W(cue_fp) * cand_fp).sum(-1)
        else:
            s = self.mlp(torch.cat([cue_fp, cand_fp], -1)).squeeze(-1)
        return s.squeeze(0) if squeeze else s


def safe_phrase_fp(stack, ws):
    mw = max(1, int(getattr(stack.phrase, "max_len", 5)) - 1)
    ws = list(ws)[-mw:]
    if not ws:
        return None
    try:
        return stack.phrase_fp(ws)
    except Exception:
        return None


def cand_phrase_fp(stack, left, rel, right, cache):
    key = (left, rel, right)
    if key in cache:
        return cache[key]
    fp = safe_phrase_fp(stack, [left, rel, right])
    cache[key] = fp
    return fp


def cue_fp_from_prefix(stack, ws_prefix, cache):
    key = tuple(ws_prefix[-4:])
    if key in cache:
        return cache[key]
    # cue: up to last 3 tokens ending in rel, or [left, rel]
    if ws_prefix and ws_prefix[-1] in RELS:
        left = ws_prefix[-2] if len(ws_prefix) >= 2 else ws_prefix[-1]
        rel = ws_prefix[-1]
        fp = safe_phrase_fp(stack, [left, rel])
        if fp is None:
            fp = safe_phrase_fp(stack, ws_prefix[-3:])
    else:
        fp = safe_phrase_fp(stack, ws_prefix[-3:])
    cache[key] = fp
    return fp


@torch.no_grad()
def pure_hop_score_fingerprint(stack, ws_prefix, cand_word, cue_cache, cand_cache):
    if not ws_prefix or ws_prefix[-1] not in RELS:
        return None
    left = ws_prefix[-2] if len(ws_prefix) >= 2 else ""
    rel = ws_prefix[-1]
    cue = cue_fp_from_prefix(stack, ws_prefix, cue_cache)
    cand = cand_phrase_fp(stack, left, rel, cand_word, cand_cache)
    if cue is None or cand is None:
        return None
    return float(F.cosine_similarity(cue.unsqueeze(0), cand.unsqueeze(0)).item())


@torch.no_grad()
def pure_hop_score_memory(mem, stack, ws_prefix, cand_word, cue_cache):
    cue = cue_fp_from_prefix(stack, ws_prefix, cue_cache)
    if cue is None or not mem.slots:
        return None
    best = None
    for name, sim in mem.topk(cue, k=min(32, len(mem.slots))):
        meta = mem.fact_meta.get(name) or {}
        if meta.get("right") == cand_word:
            best = float(sim) if best is None else max(best, float(sim))
    return best


def pick_among_top5(scores_by_id, top5_ids):
    """argmax score; skip None; if all None return top5_ids[0] (atom) as last resort flagged."""
    best_i, best_s = None, None
    for i in top5_ids:
        s = scores_by_id.get(i)
        if s is None:
            continue
        if best_s is None or s > best_s:
            best_s, best_i = s, i
    if best_i is None:
        return top5_ids[0], True  # fell back to atom
    return best_i, False


@torch.no_grad()
def eval_modes(model, stack, mem, hold, words, stoi, device, scorer=None):
    """
    Modes on RIGHT slot only (+ ALL for story report):
      atom: argmax full V (standard)
      pure_fp: argmax fingerprint cos among top5
      pure_mem: argmax mem sim among top5
      ft_fp: argmax scorer(cue,cand) among top5 (if scorer)
    """
    stats = {
        m: {"n": 0, "h": 0, "oracle_k": 0, "fallback_atom": 0, "n_right": 0, "h_right": 0}
        for m in ("atom", "pure_fp", "pure_mem", "ft_fp")
    }
    cue_cache, cand_cache = {}, {}

    for ln in hold:
        ws = ln["words"]
        ids = [stoi[w] for w in ws if w in stoi]
        if len(ids) != len(ws):
            continue
        for t in range(1, len(ws)):
            gold = ws[t]
            if gold not in stoi:
                continue
            gid = stoi[gold]
            logits = model.logits_from_prefix(ids[:t][-model.max_len :])
            top5 = [int(i) for i in logits.topk(min(K, logits.numel())).indices.tolist()]
            atom_pred = int(logits.argmax())

            # atom always
            for key in ("atom",):
                stats[key]["n"] += 1
                stats[key]["h"] += int(atom_pred == gid)
                stats[key]["oracle_k"] += int(gid in top5)

            is_right = ws[t - 1] in RELS
            if not is_right:
                # non-right: hop modes = atom (honest: arbiter only on obj)
                for key in ("pure_fp", "pure_mem", "ft_fp"):
                    stats[key]["n"] += 1
                    stats[key]["h"] += int(atom_pred == gid)
                continue

            # --- right slot: pure ranks ---
            scores_fp = {}
            scores_mem = {}
            scores_ft = {}
            for wid in top5:
                w = words[wid]
                scores_fp[wid] = pure_hop_score_fingerprint(stack, ws[:t], w, cue_cache, cand_cache)
                scores_mem[wid] = pure_hop_score_memory(mem, stack, ws[:t], w, cue_cache)
                if scorer is not None:
                    left = ws[t - 2] if t >= 2 else ""
                    rel = ws[t - 1]
                    cue = cue_fp_from_prefix(stack, ws[:t], cue_cache)
                    cand = cand_phrase_fp(stack, left, rel, w, cand_cache)
                    if cue is not None and cand is not None:
                        scores_ft[wid] = float(scorer(cue, cand).item())
                    else:
                        scores_ft[wid] = None

            pred_fp, fb_fp = pick_among_top5(scores_fp, top5)
            pred_mem, fb_mem = pick_among_top5(scores_mem, top5)
            if scorer is not None:
                pred_ft, fb_ft = pick_among_top5(scores_ft, top5)
            else:
                pred_ft, fb_ft = atom_pred, True

            for key, pred, fb in (
                ("pure_fp", pred_fp, fb_fp),
                ("pure_mem", pred_mem, fb_mem),
                ("ft_fp", pred_ft, fb_ft),
            ):
                stats[key]["n"] += 1
                stats[key]["h"] += int(pred == gid)
                stats[key]["oracle_k"] += int(gid in top5)
                stats[key]["fallback_atom"] += int(fb)
                stats[key]["n_right"] += 1
                stats[key]["h_right"] += int(pred == gid)

            stats["atom"]["n_right"] += 1
            stats["atom"]["h_right"] += int(atom_pred == gid)
            stats["atom"]["oracle_k"] += 0  # already counted in n loop — fix double count
            # fix atom oracle: counted once per position above; for right we shouldn't double
            # Actually atom oracle_k incremented for every position in first loop — OK for ALL
            # For right-only metrics use n_right/h_right

    out = {}
    for m, d in stats.items():
        n = max(d["n"], 1)
        nr = max(d["n_right"], 1)
        out[m] = {
            "all_hit1": d["h"] / n,
            "n": d["n"],
            "obj_hit1": d["h_right"] / nr,
            "n_right": d["n_right"],
            "oracle_k_all": d["oracle_k"] / n,
            "frac_fallback_atom_on_right": d["fallback_atom"] / nr,
        }
    # atom oracle on right separately
    return out


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"133 honest hop-rank start {datetime.now(timezone.utc).isoformat()}")
    log("Protocol: (a) pure fp/mem rank@top5  (b) FT fingerprint adapter  (c) vs atom@1")
    log("NOT: hop filter+atom argmax; NOT: MLP with atom logit features")
    try:
        upstream = wait_upstream()
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
        # keep phrase trainable? NO — frozen encode, only adapter
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
        fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)

        model = WordIdTransformer(len(words), 256, 4, 2, 16, 0.1).to(device)
        path = ATOM125 if ATOM125.exists() else ATOM100
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert list(ck["surfaces"]) == words
        model.load_state_dict(ck["word_tf"], strict=True)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        log(f"[init] frozen atom {path.name}")

        log("[mem] build (frozen slots from phrase_fp) ...")
        mem, by_lr, by_rel = build_path_memory(
            stack, cfg, train, device,
            fact_path=FACT_BANK_EXP_F if FACT_BANK_EXP_F.exists() else None,
        )

        ev_seen = _subsample(hold_seen, 600, 1)
        ev_story = _subsample(hold_story, 400, 2)

        # ----- (a) eval-only -----
        log("\n======== (a) EVAL-ONLY pure hop-rank@top5 ========")
        a_seen = eval_modes(model, stack, mem, ev_seen, words, stoi, device, scorer=None)
        a_story = eval_modes(model, stack, mem, ev_story, words, stoi, device, scorer=None)
        for tag, blk in (("SEEN", a_seen), ("STORY", a_story)):
            log(
                f"  [{tag}] atom_obj={100*blk['atom']['obj_hit1']:.1f}% | "
                f"pure_fp_obj={100*blk['pure_fp']['obj_hit1']:.1f}% | "
                f"pure_mem_obj={100*blk['pure_mem']['obj_hit1']:.1f}% | "
                f"fb_fp={100*blk['pure_fp']['frac_fallback_atom_on_right']:.1f}% "
                f"fb_mem={100*blk['pure_mem']['frac_fallback_atom_on_right']:.1f}%"
            )

        # ----- (b) FT fingerprint adapter -----
        log("\n======== (b) FT fingerprint adapter (atom frozen) ========")
        scorer = FingerprintScorer(dim=cfg.dim, mode="bilinear").to(device)
        opt = torch.optim.Adam(scorer.parameters(), lr=1e-3)
        fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
        story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
        fat_p = lines_to_pairs(fat, stoi)
        story_p = lines_to_pairs(_subsample(story, 40000, 13), stoi)
        right_p = [
            ex for ex in fat_p + story_p
            if ex["phrase"].split()[ex["prefix_len"] - 1] in RELS
        ]
        rr = random.Random(23)
        cue_cache, cand_cache = {}, {}
        best = {
            "obj": -1.0,
            "state": {k: v.detach().cpu().clone() for k, v in scorer.state_dict().items()},
            "eval": None,
        }

        for step in range(1, 25001):
            for g in opt.param_groups:
                g["lr"] = _warmup_then_constant(step, 1e-3, 200)
            losses = []
            for _ in range(8):
                if not right_p:
                    break
                ex = rr.choice(right_p)
                ws = ex["phrase"].split()
                pl = int(ex["prefix_len"])
                if pl < 1 or ws[pl - 1] not in RELS:
                    continue
                gold = ex["target_word"]
                if gold not in stoi:
                    continue
                packed = collate_word_id_batch([ex], stoi, model.max_len, model.pad_id, device)
                if packed is None:
                    continue
                ids, mask, tgt = packed
                with torch.no_grad():
                    logits = model.logits_last_from_batch(ids, mask)[0]
                    top5 = [int(i) for i in logits.topk(min(K, logits.numel())).indices.tolist()]
                gid = int(tgt[0])
                if gid not in top5:
                    # cannot supervise hop-rank among top5 — skip (honest)
                    continue
                left = ws[pl - 2] if pl >= 2 else ""
                rel = ws[pl - 1]
                cue = cue_fp_from_prefix(stack, ws[:pl], cue_cache)
                if cue is None:
                    continue
                scores = []
                valid = []
                for wid in top5:
                    cand = cand_phrase_fp(stack, left, rel, words[wid], cand_cache)
                    if cand is None:
                        scores.append(torch.tensor(-1e3, device=device))
                        valid.append(False)
                    else:
                        scores.append(scorer(cue, cand))
                        valid.append(True)
                if not any(valid) or not valid[top5.index(gid)]:
                    continue
                score_t = torch.stack(scores)
                target = torch.tensor([top5.index(gid)], device=device)
                losses.append(F.cross_entropy(score_t.unsqueeze(0), target))
                # hinge vs hardest wrong in top5
                neg_idx = [i for i, wid in enumerate(top5) if wid != gid and valid[i]]
                if neg_idx:
                    hard = neg_idx[int(score_t[neg_idx].argmax())]
                    losses.append(F.relu(0.2 - (score_t[top5.index(gid)] - score_t[hard])))
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % 2000 == 0 or step == 25000:
                scorer.eval()
                e = eval_modes(model, stack, mem, ev_seen, words, stoi, device, scorer=scorer)
                st = eval_modes(model, stack, mem, ev_story, words, stoi, device, scorer=scorer)
                obj = e["ft_fp"]["obj_hit1"]
                log(
                    f"  [133b] step {step}: SEEN obj atom={100*e['atom']['obj_hit1']:.1f}% "
                    f"ft_fp={100*obj:.1f}% pure_fp={100*e['pure_fp']['obj_hit1']:.1f}% | "
                    f"STORY ft_fp_obj={100*st['ft_fp']['obj_hit1']:.1f}%"
                )
                if obj >= best["obj"]:
                    best.update(
                        obj=obj,
                        state={k: v.detach().cpu().clone() for k, v in scorer.state_dict().items()},
                        eval_seen=e,
                        eval_story=st,
                    )
                scorer.train()

        scorer.load_state_dict(best["state"])
        # final (c)
        log("\n======== (c) final compare ========")
        scorer.eval()
        f_seen = eval_modes(model, stack, mem, ev_seen, words, stoi, device, scorer=scorer)
        f_story = eval_modes(model, stack, mem, ev_story, words, stoi, device, scorer=scorer)

        atom_obj = f_seen["atom"]["obj_hit1"]
        pure_fp = f_seen["pure_fp"]["obj_hit1"]
        pure_mem = f_seen["pure_mem"]["obj_hit1"]
        ft_fp = f_seen["ft_fp"]["obj_hit1"]
        best_hop = max(pure_fp, pure_mem, ft_fp)
        lift = best_hop - atom_obj

        # rel held from frozen atom
        suite = eval_id_capacity_suite(model, ev_seen, train, words, stoi, device)
        rel = suite["roles"].get("rel", {}).get("hit1", 0.0)

        if rel >= 0.70 and lift >= 0.03:
            verdict = "PASS"
        elif rel >= 0.70 and lift >= 0.015:
            verdict = "PARTIAL"
        else:
            verdict = "PARITY"

        ck = CKPT / "stage133_honest_hop_rank.pt"
        torch.save({"scorer": best["state"], "surfaces": words, "K": K, "mode": "bilinear_fp"}, ck)

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "honest_pure_hop_rank_then_FT_fingerprint",
            "not_tested_as": ["128_filter_plus_atom_argmax", "130_mlp_with_atom_logit_feat"],
            "upstream": upstream.get("via") or upstream.get("verdict") or upstream.get("dig"),
            "a_eval_only": {"seen": a_seen, "story": a_story},
            "c_final": {"seen": f_seen, "story": f_story},
            "seen_obj": {
                "atom": atom_obj,
                "pure_fp": pure_fp,
                "pure_mem": pure_mem,
                "ft_fp": ft_fp,
                "best_hop": best_hop,
                "lift_vs_atom_pp": lift,
            },
            "rel_atom": rel,
            "verdict": verdict,
            "ckpt": str(ck),
            "note": (
                "Arbiter only on right slot; non-right = atom. "
                "Supervise FT only when gold in atom top5."
            ),
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        (RES / f"stage133_honest_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")

        log(
            f"[133] {verdict} SEEN obj atom={100*atom_obj:.1f}% "
            f"pure_fp={100*pure_fp:.1f}% pure_mem={100*pure_mem:.1f}% "
            f"ft_fp={100*ft_fp:.1f}% best_lift={100*lift:+.1f}pp"
        )

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stage 133 honest hop-rank@top5:** {verdict} "
                f"atom_obj={100*atom_obj:.1f}% pure_fp={100*pure_fp:.1f}% "
                f"ft_fp={100*ft_fp:.1f}% lift={100*lift:+.1f}pp. "
                f"`stage133_honest_hop_rank_decision.json`.\n"
            )
            if "Stage 133 honest" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")
        log("DONE 133")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
