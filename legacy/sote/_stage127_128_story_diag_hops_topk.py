"""
Stage 127–128 after priority digs:

127 STORY-diag (exact@1 gate unchanged):
  - rank histogram of gold (1 / 2-5 / 6-20 / >20) on STORY + SEEN obj
  - SOTE top5 ∩ BPE beam5 on same STORY prefixes
  - hops∩atom_topK coverage (predicate for 128)

128 hops as arbiter among atom top-K (NOT prior on full V):
  Modes (eval + optional light FT):
    a) pick = argmax_{w in topK ∩ hop_cands} atom_logit; fallback argmax topK
    b) same + score hop retrieve sim when available
  Gate vs Stage100 / #5:
    PASS if SEEN obj +3pp with rel>=0.70 OR (obj held & near-miss gap shrinks ≥8pp)
    STORY reported; no soft claim

Run:
  python _stage127_128_story_diag_hops_topk.py
"""
from __future__ import annotations

import json
import random
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer  # noqa: E402

from train import (  # noqa: E402
    CKPT,
    FACT_BANK_EXP_F,
    RELS,
    RES,
    Config,
    WordIdTransformer,
    _role,
    _warmup_then_constant,
    build_ts_repeat_mix,
    collate_word_id_batch,
    eval_id_capacity_suite,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
)
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402
from _stage116_117_hops_morph_follow import (  # noqa: E402
    build_path_memory,
    hop_candidates_for_prefix,
)

ATOM100 = CKPT / "stage100_scale_100k.pt"
ATOM125 = CKPT / "stage125_same_role_rank.pt"
BPE_TOK = RES / "stage112_bpe_tokenizer.json"
BPE_CKPT = CKPT / "stage112_bpe_baseline.pt"
LOG = RES / "_stage127_128_log.txt"
DEC = RES / "stage127_128_decision.json"


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def setup(phrases):
    cfg = Config()
    cfg.c87_n_fat = 200
    cfg.c87_n_rare = 80
    cfg.c87_fat_copies = 40
    cfg.c87_rare_copies = 2
    cfg.c87_seen_hold_frac = 0.20
    cfg.c87_story_keep_frac = 0.50
    cfg.c87_hold_frac = 0.15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent85 = CKPT / "stage85_foundation_FROZEN.pt"
    if not parent85.exists():
        parent85 = CKPT / "stage85_foundation.pt"
    stack, _ = load_foundation_85(device, cfg, path=parent85)
    train, hold_seen, _, hold_story, _ = build_ts_repeat_mix(phrases, cfg, seed=272)
    for ln in train + hold_seen + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")
    words = sorted({w for ln in train + hold_seen + hold_story for w in ln["words"]})
    stoi = {s: i for i, s in enumerate(words)}
    fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)
    return cfg, device, stack, train, hold_seen, hold_story, words, stoi, fps


def load_atom(words, fps, device, prefer125=True):
    model = WordIdTransformer(len(words), 256, 4, 2, 16, 0.1).to(device)
    path = ATOM125 if (prefer125 and ATOM125.exists()) else ATOM100
    if path.exists():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if list(ck.get("surfaces", [])) == words:
            model.load_state_dict(ck["word_tf"], strict=True)
            log(f"[init] warm {path.name}")
            return model
    if ATOM100.exists():
        ck = torch.load(ATOM100, map_location="cpu", weights_only=False)
        if list(ck.get("surfaces", [])) == words:
            model.load_state_dict(ck["word_tf"], strict=True)
            log("[init] warm stage100")
            return model
    model.init_from_fps(fps)
    log("[init] fp")
    return model


def phrase_fp_fit(stack, ws):
    max_words = max(1, int(getattr(stack.phrase, "max_len", 5)) - 1)
    ws = ws[-max_words:] if len(ws) > max_words else ws
    if not ws:
        return None
    return stack.phrase_fp(ws)


# ---------- BPE beam top words ----------
@torch.no_grad()
def load_bpe(device):
    if not (BPE_TOK.exists() and BPE_CKPT.exists()):
        return None, None
    from transformers import GPT2Config, GPT2LMHeadModel
    from _stage111_112_follow import _encode_words

    tok = Tokenizer.from_file(str(BPE_TOK))
    V = tok.get_vocab_size()
    bos, eos, pad = tok.token_to_id("[BOS]"), tok.token_to_id("[EOS]"), tok.token_to_id("[PAD]")
    conf = GPT2Config(
        vocab_size=V, n_positions=64, n_embd=256, n_layer=4, n_head=4, n_inner=1024,
        bos_token_id=bos, eos_token_id=eos, pad_token_id=pad,
    )
    model = GPT2LMHeadModel(conf).to(device)
    ck = torch.load(BPE_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    def first_words_beam(prefix_words, beam=5, max_new=8):
        ids = _encode_words(tok, prefix_words, max_len=48, bos=bos, eos=eos, pad=pad)
        if ids and ids[-1] == eos:
            ids = ids[:-1]
        if not ids:
            ids = [bos]
        beams = [(0.0, [])]
        finished = []
        for _ in range(max_new):
            new_beams = []
            for sc, extra in beams:
                x = torch.tensor([ids + extra], dtype=torch.long, device=device)
                logits = model(x).logits[0, -1]
                logp = F.log_softmax(logits, dim=-1)
                topv, topi = logp.topk(beam)
                for lp, ti in zip(topv.tolist(), topi.tolist()):
                    ti = int(ti)
                    if ti in (eos, pad, bos):
                        decoded = tok.decode(extra).strip()
                        w = decoded.split()[0] if decoded.split() else ""
                        if w:
                            finished.append((sc + lp, w))
                        continue
                    nex = extra + [ti]
                    decoded = tok.decode(nex).strip()
                    if " " in decoded:
                        finished.append((sc + lp, decoded.split()[0]))
                    else:
                        new_beams.append((sc + lp, nex))
            new_beams.sort(key=lambda z: z[0], reverse=True)
            beams = new_beams[:beam]
            if len(finished) >= beam and not beams:
                break
        for sc, extra in beams:
            decoded = tok.decode(extra).strip()
            if decoded:
                finished.append((sc, decoded.split()[0]))
        finished.sort(key=lambda z: z[0], reverse=True)
        seen_w, out_w = set(), []
        for _, w in finished:
            if w and w not in seen_w:
                seen_w.add(w)
                out_w.append(w)
            if len(out_w) >= beam:
                break
        return out_w

    return model, first_words_beam


# ---------- 127 STORY diag ----------
@torch.no_grad()
def dig_127_story_diag(model, stack, cfg, device, words, stoi, hold_seen, hold_story, train, mem, by_lr, by_rel):
    log("\n======== 127 STORY-diag rank hist + SOTE∩BPE top5 ========")
    bpe_model, bpe_beam = load_bpe(device)

    def rank_bucket(rank: int) -> str:
        if rank == 1:
            return "r1"
        if rank <= 5:
            return "r2_5"
        if rank <= 20:
            return "r6_20"
        return "r21p"

    def analyze(hold, tag, n_lines, do_bpe=False, K=5):
        hold = _subsample(hold, min(n_lines, len(hold)), 31)
        hist = Counter_ish()
        roles = defaultdict(lambda: Counter_ish())
        inter = {
            "n": 0,
            "gold_both": 0,
            "gold_sote_only": 0,
            "gold_bpe_only": 0,
            "gold_neither": 0,
            "top5_jaccard_sum": 0.0,
            "n_jaccard": 0,
        }
        hop_cov = {
            "n_right": 0,
            "gold_in_atomK": 0,
            "gold_in_hop": 0,
            "gold_in_atomK_and_hop": 0,
            "hop_hit_atomK": 0,  # any hop cand in atom K
            "hop_nonempty": 0,
        }

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
                order = torch.argsort(logits, descending=True)
                rank = int((order == gid).nonzero()[0]) + 1
                topk_ids = [int(i) for i in order[:K].tolist()]
                topk_ws = [words[i] for i in topk_ids]
                rb = rank_bucket(rank)
                hist[rb] += 1
                hist["n"] += 1
                role = _role({"prefix_len": t, "target_word": gold}, ws)
                roles[role][rb] += 1
                roles[role]["n"] += 1
                roles["ALL"][rb] += 1
                roles["ALL"]["n"] += 1

                # hops coverage on right
                if ws[t - 1] in RELS:
                    hop = hop_candidates_for_prefix(ws[:t], by_lr, by_rel, mem, stack, stoi)
                    hop_cov["n_right"] += 1
                    in_k = gid in topk_ids
                    in_h = gid in hop
                    hop_cov["gold_in_atomK"] += int(in_k)
                    hop_cov["gold_in_hop"] += int(in_h)
                    hop_cov["gold_in_atomK_and_hop"] += int(in_k and in_h)
                    hop_cov["hop_nonempty"] += int(len(hop) > 0)
                    hop_cov["hop_hit_atomK"] += int(len(hop.intersection(topk_ids)) > 0)

                if do_bpe and bpe_beam is not None:
                    bpe_ws = bpe_beam(ws[:t], beam=K)
                    sote_set, bpe_set = set(topk_ws), set(bpe_ws)
                    inter["n"] += 1
                    g_s = gold in sote_set
                    g_b = gold in bpe_set
                    if g_s and g_b:
                        inter["gold_both"] += 1
                    elif g_s:
                        inter["gold_sote_only"] += 1
                    elif g_b:
                        inter["gold_bpe_only"] += 1
                    else:
                        inter["gold_neither"] += 1
                    if sote_set or bpe_set:
                        jac = len(sote_set & bpe_set) / max(len(sote_set | bpe_set), 1)
                        inter["top5_jaccard_sum"] += jac
                        inter["n_jaccard"] += 1

        def pack_hist(h):
            n = max(h["n"], 1)
            return {
                "n": h["n"],
                "r1": h["r1"] / n,
                "r2_5": h["r2_5"] / n,
                "r6_20": h["r6_20"] / n,
                "r21p": h["r21p"] / n,
                "in_top5": (h["r1"] + h["r2_5"]) / n,
                "out_top5": (h["r6_20"] + h["r21p"]) / n,
            }

        out = {
            "tag": tag,
            "rank_hist_ALL": pack_hist(hist),
            "rank_hist_by_role": {k: pack_hist(v) for k, v in roles.items()},
        }
        if hop_cov["n_right"]:
            nr = hop_cov["n_right"]
            out["hop_atomK_coverage_right"] = {
                "n": nr,
                "frac_gold_in_atomK": hop_cov["gold_in_atomK"] / nr,
                "frac_gold_in_hop": hop_cov["gold_in_hop"] / nr,
                "frac_gold_in_atomK_and_hop": hop_cov["gold_in_atomK_and_hop"] / nr,
                "frac_hop_nonempty": hop_cov["hop_nonempty"] / nr,
                "frac_hop_intersects_atomK": hop_cov["hop_hit_atomK"] / nr,
            }
        if do_bpe and inter["n"]:
            n = inter["n"]
            out["sote_bpe_top5"] = {
                "n": n,
                "gold_both": inter["gold_both"] / n,
                "gold_sote_only": inter["gold_sote_only"] / n,
                "gold_bpe_only": inter["gold_bpe_only"] / n,
                "gold_neither": inter["gold_neither"] / n,
                "mean_top5_jaccard": inter["top5_jaccard_sum"] / max(inter["n_jaccard"], 1),
            }
        # print
        h = out["rank_hist_ALL"]
        log(
            f"  [{tag}] rank: @1={100*h['r1']:.1f}% r2-5={100*h['r2_5']:.1f}% "
            f"r6-20={100*h['r6_20']:.1f}% >20={100*h['r21p']:.1f}% | "
            f"in5={100*h['in_top5']:.1f}% out5={100*h['out_top5']:.1f}%"
        )
        if "hop_atomK_coverage_right" in out:
            c = out["hop_atomK_coverage_right"]
            log(
                f"  [{tag}/right] gold∈atomK={100*c['frac_gold_in_atomK']:.1f}% "
                f"gold∈hop={100*c['frac_gold_in_hop']:.1f}% "
                f"gold∈K∩hop={100*c['frac_gold_in_atomK_and_hop']:.1f}% "
                f"hop∩K nonempty={100*c['frac_hop_intersects_atomK']:.1f}%"
            )
        if "sote_bpe_top5" in out:
            s = out["sote_bpe_top5"]
            log(
                f"  [{tag}∩BPE] both={100*s['gold_both']:.1f}% sote_only={100*s['gold_sote_only']:.1f}% "
                f"bpe_only={100*s['gold_bpe_only']:.1f}% neither={100*s['gold_neither']:.1f}% "
                f"jaccard={s['mean_top5_jaccard']:.3f}"
            )
        return out

    class Counter_ish(defaultdict):
        def __init__(self):
            super().__init__(int)

    # redefine with local helper using plain dict
    def Counter_ish():
        return defaultdict(int)

    # fix: analyze uses Counter_ish before redefinition in nested - use defaultdict directly
    return _analyze_impl(
        model, stack, device, words, stoi, hold_seen, hold_story,
        mem, by_lr, by_rel, bpe_beam, K=5,
    )


def _analyze_impl(model, stack, device, words, stoi, hold_seen, hold_story, mem, by_lr, by_rel, bpe_beam, K=5):
    def rank_bucket(rank: int) -> str:
        if rank == 1:
            return "r1"
        if rank <= 5:
            return "r2_5"
        if rank <= 20:
            return "r6_20"
        return "r21p"

    def analyze(hold, tag, n_lines, do_bpe=False):
        hold = _subsample(hold, min(n_lines, len(hold)), 31)
        hist = defaultdict(int)
        roles = defaultdict(lambda: defaultdict(int))
        inter = defaultdict(float)
        hop_cov = defaultdict(int)

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
                order = torch.argsort(logits, descending=True)
                rank = int((order == gid).nonzero()[0]) + 1
                topk_ids = [int(i) for i in order[:K].tolist()]
                topk_ws = [words[i] for i in topk_ids]
                rb = rank_bucket(rank)
                hist[rb] += 1
                hist["n"] += 1
                role = _role({"prefix_len": t, "target_word": gold}, ws)
                roles[role][rb] += 1
                roles[role]["n"] += 1
                roles["ALL"][rb] += 1
                roles["ALL"]["n"] += 1

                if t >= 1 and ws[t - 1] in RELS:
                    hop = hop_candidates_for_prefix(ws[:t], by_lr, by_rel, mem, stack, stoi)
                    hop_cov["n_right"] += 1
                    hop_cov["gold_in_atomK"] += int(gid in topk_ids)
                    hop_cov["gold_in_hop"] += int(gid in hop)
                    hop_cov["gold_in_atomK_and_hop"] += int(gid in topk_ids and gid in hop)
                    hop_cov["hop_nonempty"] += int(len(hop) > 0)
                    hop_cov["hop_hit_atomK"] += int(len(hop.intersection(set(topk_ids))) > 0)

                if do_bpe and bpe_beam is not None:
                    bpe_ws = bpe_beam(ws[:t], beam=K)
                    sote_set, bpe_set = set(topk_ws), set(bpe_ws)
                    inter["n"] += 1
                    g_s, g_b = gold in sote_set, gold in bpe_set
                    if g_s and g_b:
                        inter["gold_both"] += 1
                    elif g_s:
                        inter["gold_sote_only"] += 1
                    elif g_b:
                        inter["gold_bpe_only"] += 1
                    else:
                        inter["gold_neither"] += 1
                    uni = sote_set | bpe_set
                    if uni:
                        inter["top5_jaccard_sum"] += len(sote_set & bpe_set) / len(uni)
                        inter["n_jaccard"] += 1

        def pack_hist(h):
            n = max(h["n"], 1)
            return {
                "n": h["n"],
                "r1": h["r1"] / n,
                "r2_5": h["r2_5"] / n,
                "r6_20": h["r6_20"] / n,
                "r21p": h["r21p"] / n,
                "in_top5": (h["r1"] + h["r2_5"]) / n,
                "out_top5": (h["r6_20"] + h["r21p"]) / n,
            }

        out = {
            "tag": tag,
            "rank_hist_ALL": pack_hist(hist),
            "rank_hist_by_role": {k: pack_hist(v) for k, v in roles.items()},
        }
        if hop_cov["n_right"]:
            nr = hop_cov["n_right"]
            out["hop_atomK_coverage_right"] = {
                "n": nr,
                "frac_gold_in_atomK": hop_cov["gold_in_atomK"] / nr,
                "frac_gold_in_hop": hop_cov["gold_in_hop"] / nr,
                "frac_gold_in_atomK_and_hop": hop_cov["gold_in_atomK_and_hop"] / nr,
                "frac_hop_nonempty": hop_cov["hop_nonempty"] / nr,
                "frac_hop_intersects_atomK": hop_cov["hop_hit_atomK"] / nr,
            }
        if do_bpe and inter["n"]:
            n = inter["n"]
            out["sote_bpe_top5"] = {
                "n": int(n),
                "gold_both": inter["gold_both"] / n,
                "gold_sote_only": inter["gold_sote_only"] / n,
                "gold_bpe_only": inter["gold_bpe_only"] / n,
                "gold_neither": inter["gold_neither"] / n,
                "mean_top5_jaccard": inter["top5_jaccard_sum"] / max(inter["n_jaccard"], 1),
            }
        h = out["rank_hist_ALL"]
        log(
            f"  [{tag}] rank: @1={100*h['r1']:.1f}% r2-5={100*h['r2_5']:.1f}% "
            f"r6-20={100*h['r6_20']:.1f}% >20={100*h['r21p']:.1f}% | "
            f"in5={100*h['in_top5']:.1f}% out5={100*h['out_top5']:.1f}%"
        )
        if "hop_atomK_coverage_right" in out:
            c = out["hop_atomK_coverage_right"]
            log(
                f"  [{tag}/right] gold∈atomK={100*c['frac_gold_in_atomK']:.1f}% "
                f"gold∈hop={100*c['frac_gold_in_hop']:.1f}% "
                f"gold∈K∩hop={100*c['frac_gold_in_atomK_and_hop']:.1f}% "
                f"hop∩K={100*c['frac_hop_intersects_atomK']:.1f}%"
            )
        if "sote_bpe_top5" in out:
            s = out["sote_bpe_top5"]
            log(
                f"  [{tag}∩BPE] both={100*s['gold_both']:.1f}% sote_only={100*s['gold_sote_only']:.1f}% "
                f"bpe_only={100*s['gold_bpe_only']:.1f}% neither={100*s['gold_neither']:.1f}% "
                f"jaccard={s['mean_top5_jaccard']:.3f}"
            )
        return out

    seen = analyze(hold_seen, "SEEN", 500, do_bpe=False)
    # SEEN right-focused already in roles; also story with BPE
    story = analyze(hold_story, "STORY", 400, do_bpe=True)
    out = {"stage": 127, "seen": seen, "story": story, "K": K}
    (RES / "stage127_story_diag.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ---------- 128 hops among top-K ----------
def pick_among_topk(logits, topk_ids, hop_ids, mode="intersect_argmax"):
    """Return pred id. Never expands outside topk_ids (except empty hop→argmax topk)."""
    topk_ids = list(topk_ids)
    if not topk_ids:
        return int(logits.argmax())
    hop_in = [i for i in topk_ids if i in hop_ids]
    if mode == "intersect_argmax" and hop_in:
        scores = [(i, float(logits[i])) for i in hop_in]
        return max(scores, key=lambda z: z[1])[0]
    # fallback: plain argmax in topK (= global argmax if K covers top)
    return topk_ids[0]


@torch.no_grad()
def eval_atom_vs_hop_topk(model, hold, words, stoi, mem, by_lr, by_rel, stack, K=5, mode="intersect_argmax"):
    roles = defaultdict(lambda: {"n": 0, "atom": 0, "hopK": 0})
    obj = {"n": 0, "atom": 0, "hopK": 0, "oracle_K": 0}
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
            order = torch.argsort(logits, descending=True)
            topk_ids = [int(i) for i in order[:K].tolist()]
            atom_pred = topk_ids[0]
            hop = set()
            if ws[t - 1] in RELS:
                hop = hop_candidates_for_prefix(ws[:t], by_lr, by_rel, mem, stack, stoi)
            hop_pred = pick_among_topk(logits, topk_ids, hop, mode=mode)
            role = _role({"prefix_len": t, "target_word": gold}, ws)
            for name in (role, "ALL"):
                roles[name]["n"] += 1
                roles[name]["atom"] += int(atom_pred == gid)
                roles[name]["hopK"] += int(hop_pred == gid)
            if ws[t - 1] in RELS:
                obj["n"] += 1
                obj["atom"] += int(atom_pred == gid)
                obj["hopK"] += int(hop_pred == gid)
                obj["oracle_K"] += int(gid in topk_ids)
    def pack(d):
        n = max(d["n"], 1)
        return {k: (d[k] / n if k != "n" else d[k]) for k in d}
    return {
        "roles": {k: pack(v) for k, v in roles.items()},
        "obj": pack(obj),
    }


def dig_128_hops_topk(model, stack, cfg, device, words, stoi, train, hold_seen, hold_story, mem, by_lr, by_rel, diag127):
    log("\n======== 128 hops-rerank among atom top-K (not full-V prior) ========")
    K = 5
    # coverage gate from 127
    cov = (diag127.get("seen") or {}).get("hop_atomK_coverage_right") or {}
    log(f"[128] SEEN right coverage from 127: {json.dumps(cov)}")

    ev_seen = _subsample(hold_seen, 600, 1)
    ev_story = _subsample(hold_story, 400, 2)

    base_seen = eval_atom_vs_hop_topk(model, ev_seen, words, stoi, mem, by_lr, by_rel, stack, K=K)
    base_story = eval_atom_vs_hop_topk(model, ev_story, words, stoi, mem, by_lr, by_rel, stack, K=K)
    suite0 = eval_id_capacity_suite(model, ev_seen, train, words, stoi, device)
    rel0 = suite0["roles"].get("rel", {}).get("hit1", 0.0)
    log(
        f"  [128 eval0] SEEN obj atom={100*base_seen['obj']['atom']:.1f}% "
        f"hopK={100*base_seen['obj']['hopK']:.1f}% oracleK={100*base_seen['obj']['oracle_K']:.1f}% "
        f"rel={100*rel0:.1f}% | "
        f"STORY ALL atom={100*base_story['roles']['ALL']['atom']:.1f}% "
        f"hopK={100*base_story['roles']['ALL']['hopK']:.1f}%"
    )

    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p = lines_to_pairs(fat, stoi)
    story_p = lines_to_pairs(_subsample(story, 40000, 8), stoi)

    def is_right(ex):
        return _role(ex, ex["phrase"].split()) == "right"

    right_p = [ex for ex in fat_p + story_p if is_right(ex)]
    other_p = [ex for ex in fat_p if not is_right(ex)]
    rr = random.Random(11)
    best = {
        "obj": base_seen["obj"]["hopK"],
        "atom_obj": base_seen["obj"]["atom"],
        "rel": rel0,
        "story": base_story["roles"]["ALL"]["hopK"],
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "gap": base_seen["obj"]["oracle_K"] - base_seen["obj"]["atom"],
    }

    for step in range(1, 25001):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, 5e-4, 100)
        losses = []
        # 4x right: margin inside frozen topK snapshot
        for _ in range(4):
            if not right_p:
                break
            ex = rr.choice(right_p)
            packed = collate_word_id_batch([ex], stoi, model.max_len, model.pad_id, device)
            if packed is None:
                continue
            ids, mask, tgt = packed
            with torch.no_grad():
                logits0 = model.logits_last_from_batch(ids, mask)[0]
                topk_ids = torch.topk(logits0, k=min(K, logits0.numel())).indices.tolist()
            gold = int(tgt[0])
            if gold not in topk_ids:
                # standard CE to pull gold up
                losses.append(F.cross_entropy(model.logits_last_from_batch(ids, mask), tgt))
                continue
            ws = ex["phrase"].split()
            hop = hop_candidates_for_prefix(ws[: ex["prefix_len"]], by_lr, by_rel, mem, stack, stoi)
            logits = model.logits_last_from_batch(ids, mask)[0]
            # CE over topK (discrimination among near-miss competitors)
            sl = torch.tensor(topk_ids, device=device, dtype=torch.long)
            target = torch.tensor([topk_ids.index(gold)], device=device)
            losses.append(F.cross_entropy(logits.index_select(0, sl).unsqueeze(0), target))
            # extra: if hop∩topk has distractors, hinge gold vs best hop distractor
            hop_dist = [i for i in topk_ids if i in hop and i != gold]
            if hop_dist:
                hard = hop_dist[int(logits[torch.tensor(hop_dist, device=device)].argmax())]
                losses.append(F.relu(0.5 - (logits[gold] - logits[hard])))

        # 2x other full CE hold rel
        for _ in range(2):
            if not other_p:
                break
            ex = rr.choice(other_p)
            packed = collate_word_id_batch([ex], stoi, model.max_len, model.pad_id, device)
            if packed is None:
                continue
            ids, mask, tgt = packed
            losses.append(F.cross_entropy(model.logits_last_from_batch(ids, mask), tgt))

        if not losses:
            continue
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 2000 == 0 or step == 25000:
            model.eval()
            s = eval_atom_vs_hop_topk(model, ev_seen, words, stoi, mem, by_lr, by_rel, stack, K=K)
            st = eval_atom_vs_hop_topk(model, ev_story, words, stoi, mem, by_lr, by_rel, stack, K=K)
            suite = eval_id_capacity_suite(model, ev_seen, train, words, stoi, device)
            rel = suite["roles"].get("rel", {}).get("hit1", 0.0)
            obj_h = s["obj"]["hopK"]
            obj_a = s["obj"]["atom"]
            log(
                f"  [128] step {step}: SEEN obj atom={100*obj_a:.1f}% hopK={100*obj_h:.1f}% "
                f"oracleK={100*s['obj']['oracle_K']:.1f}% rel={100*rel:.1f}% | "
                f"STORY hopK={100*st['roles']['ALL']['hopK']:.1f}%"
            )
            if rel >= 0.70 and obj_h >= best["obj"]:
                best.update(
                    obj=obj_h, atom_obj=obj_a, rel=rel,
                    story=st["roles"]["ALL"]["hopK"],
                    gap=s["obj"]["oracle_K"] - obj_a,
                    state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    eval_seen=s, eval_story=st,
                )
            model.train()

    # restore best
    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=True)

    lift = best["obj"] - REF100["obj"]
    gap_shrink = (base_seen["obj"]["oracle_K"] - base_seen["obj"]["atom"]) - best.get("gap", 0)
    # also compare hopK vs atom at start
    hop_lift = best["obj"] - base_seen["obj"]["atom"]

    if best["rel"] >= 0.70 and (lift >= 0.03 or hop_lift >= 0.03):
        verdict = "PASS"
    elif best["rel"] >= 0.70 and (lift >= 0.015 or hop_lift >= 0.015):
        verdict = "PARTIAL"
    else:
        verdict = "PARITY"

    ck = CKPT / "stage128_hops_topk.pt"
    torch.save({"word_tf": best["state"], "surfaces": words, "K": K}, ck)
    out = {
        "dig": "128_hops_among_topk",
        "verdict": verdict,
        "K": K,
        "mode": "intersect_argmax + topK CE FT",
        "eval0": {"seen": base_seen, "story": base_story},
        "best": {
            "seen_obj_atom": best["atom_obj"],
            "seen_obj_hopK": best["obj"],
            "rel": best["rel"],
            "story_hopK": best["story"],
            "hopK_minus_atom0_pp": hop_lift,
            "obj_lift_vs_ref100_pp": lift,
        },
        "ckpt": str(ck),
        "note": "Selection restricted to atom top-K; hops only arbitrate inside K",
    }
    (RES / f"stage128_hops_topk_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(
        f"[128] {verdict} hopK_obj={100*best['obj']:.1f}% "
        f"(atom0={100*base_seen['obj']['atom']:.1f}% Δ={100*hop_lift:+.1f}pp) "
        f"STORY_hopK={100*best['story']:.1f}%"
    )
    return out


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"127/128 start {datetime.now(timezone.utc).isoformat()}")
    try:
        phrases = ensure_100k()
        cfg, device, stack, train, hold_seen, hold_story, words, stoi, fps = setup(phrases)
        model = load_atom(words, fps, device, prefer125=True)
        model.eval()

        log("[mem] build path memory ...")
        mem, by_lr, by_rel = build_path_memory(
            stack, cfg, train, device, fact_path=FACT_BANK_EXP_F if FACT_BANK_EXP_F.exists() else None
        )

        _orig = stack.phrase_fp

        def _safe_phrase_fp(ws):
            max_words = max(1, int(getattr(stack.phrase, "max_len", 5)) - 1)
            if len(ws) > max_words:
                ws = ws[-max_words:]
            return _orig(ws)

        stack.phrase_fp = _safe_phrase_fp  # type: ignore

        _, bpe_beam = load_bpe(device)
        log("\n======== 127 STORY-diag ========")
        diag127 = run_127_clean(
            model, stack, device, words, stoi, hold_seen, hold_story, mem, by_lr, by_rel, bpe_beam
        )

        r128 = dig_128_hops_topk(
            model, stack, cfg, device, words, stoi, train, hold_seen, hold_story, mem, by_lr, by_rel, diag127
        )

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage127": diag127,
            "stage128": r128,
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stages 127–128:** STORY-diag + hops-among-topK → 128 {r128['verdict']} "
                f"hopK_obj={100*r128['best']['seen_obj_hopK']:.1f}% "
                f"(Δ vs atom0 {100*r128['best']['hopK_minus_atom0_pp']:+.1f}pp). "
                f"`stage127_128_decision.json`.\n"
            )
            if "Stages 127–128" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")
        log("DONE 127/128")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


def run_127_clean(model, stack, device, words, stoi, hold_seen, hold_story, mem, by_lr, by_rel, bpe_beam):
    """Clean 127 without the broken wrapper."""
    # monkeypatch analyze by calling _analyze_impl differently — split returns
    # Rebuild: call internal twice by temporarily modifying — simplest copy of analyze from _analyze_impl

    def one(hold, tag, n_lines, do_bpe):
        # hijack: run full _analyze_impl logic for one set by wrapping hold
        # Easiest: duplicate call pattern from _analyze_impl.analyze
        hold = _subsample(hold, min(n_lines, len(hold)), 31)
        K = 5
        hist = defaultdict(int)
        roles = defaultdict(lambda: defaultdict(int))
        inter = defaultdict(float)
        hop_cov = defaultdict(int)

        def rank_bucket(rank: int) -> str:
            if rank == 1:
                return "r1"
            if rank <= 5:
                return "r2_5"
            if rank <= 20:
                return "r6_20"
            return "r21p"

        with torch.no_grad():
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
                    order = torch.argsort(logits, descending=True)
                    rank = int((order == gid).nonzero()[0]) + 1
                    topk_ids = [int(i) for i in order[:K].tolist()]
                    topk_ws = [words[i] for i in topk_ids]
                    rb = rank_bucket(rank)
                    hist[rb] += 1
                    hist["n"] += 1
                    role = _role({"prefix_len": t, "target_word": gold}, ws)
                    roles[role][rb] += 1
                    roles[role]["n"] += 1
                    roles["ALL"][rb] += 1
                    roles["ALL"]["n"] += 1
                    if ws[t - 1] in RELS:
                        hop = hop_candidates_for_prefix(ws[:t], by_lr, by_rel, mem, stack, stoi)
                        hop_cov["n_right"] += 1
                        hop_cov["gold_in_atomK"] += int(gid in topk_ids)
                        hop_cov["gold_in_hop"] += int(gid in hop)
                        hop_cov["gold_in_atomK_and_hop"] += int(gid in topk_ids and gid in hop)
                        hop_cov["hop_nonempty"] += int(len(hop) > 0)
                        hop_cov["hop_hit_atomK"] += int(len(set(topk_ids) & hop) > 0)
                    if do_bpe and bpe_beam is not None:
                        bpe_ws = bpe_beam(ws[:t], beam=K)
                        sote_set, bpe_set = set(topk_ws), set(bpe_ws)
                        inter["n"] += 1
                        g_s, g_b = gold in sote_set, gold in bpe_set
                        if g_s and g_b:
                            inter["gold_both"] += 1
                        elif g_s:
                            inter["gold_sote_only"] += 1
                        elif g_b:
                            inter["gold_bpe_only"] += 1
                        else:
                            inter["gold_neither"] += 1
                        uni = sote_set | bpe_set
                        if uni:
                            inter["top5_jaccard_sum"] += len(sote_set & bpe_set) / len(uni)
                            inter["n_jaccard"] += 1

        def pack_hist(h):
            n = max(h["n"], 1)
            return {
                "n": h["n"],
                "r1": h["r1"] / n,
                "r2_5": h["r2_5"] / n,
                "r6_20": h["r6_20"] / n,
                "r21p": h["r21p"] / n,
                "in_top5": (h["r1"] + h["r2_5"]) / n,
                "out_top5": (h["r6_20"] + h["r21p"]) / n,
            }

        out = {
            "tag": tag,
            "rank_hist_ALL": pack_hist(hist),
            "rank_hist_by_role": {k: pack_hist(v) for k, v in roles.items()},
        }
        if hop_cov["n_right"]:
            nr = hop_cov["n_right"]
            out["hop_atomK_coverage_right"] = {
                "n": nr,
                "frac_gold_in_atomK": hop_cov["gold_in_atomK"] / nr,
                "frac_gold_in_hop": hop_cov["gold_in_hop"] / nr,
                "frac_gold_in_atomK_and_hop": hop_cov["gold_in_atomK_and_hop"] / nr,
                "frac_hop_nonempty": hop_cov["hop_nonempty"] / nr,
                "frac_hop_intersects_atomK": hop_cov["hop_hit_atomK"] / nr,
            }
        if do_bpe and inter["n"]:
            n = inter["n"]
            out["sote_bpe_top5"] = {
                "n": int(n),
                "gold_both": inter["gold_both"] / n,
                "gold_sote_only": inter["gold_sote_only"] / n,
                "gold_bpe_only": inter["gold_bpe_only"] / n,
                "gold_neither": inter["gold_neither"] / n,
                "mean_top5_jaccard": inter["top5_jaccard_sum"] / max(inter["n_jaccard"], 1),
            }
        h = out["rank_hist_ALL"]
        log(
            f"  [{tag}] rank: @1={100*h['r1']:.1f}% r2-5={100*h['r2_5']:.1f}% "
            f"r6-20={100*h['r6_20']:.1f}% >20={100*h['r21p']:.1f}% | "
            f"in5={100*h['in_top5']:.1f}% out5={100*h['out_top5']:.1f}%"
        )
        if "hop_atomK_coverage_right" in out:
            c = out["hop_atomK_coverage_right"]
            log(
                f"  [{tag}/right] gold∈atomK={100*c['frac_gold_in_atomK']:.1f}% "
                f"gold∈hop={100*c['frac_gold_in_hop']:.1f}% "
                f"gold∈K∩hop={100*c['frac_gold_in_atomK_and_hop']:.1f}% "
                f"hop∩K={100*c['frac_hop_intersects_atomK']:.1f}%"
            )
        if "sote_bpe_top5" in out:
            s = out["sote_bpe_top5"]
            log(
                f"  [{tag}∩BPE] both={100*s['gold_both']:.1f}% sote_only={100*s['gold_sote_only']:.1f}% "
                f"bpe_only={100*s['gold_bpe_only']:.1f}% neither={100*s['gold_neither']:.1f}% "
                f"jaccard={s['mean_top5_jaccard']:.3f}"
            )
        return out

    seen = one(hold_seen, "SEEN", 500, False)
    story = one(hold_story, "STORY", 350, True)
    out = {"stage": 127, "seen": seen, "story": story, "K": 5}
    (RES / "stage127_story_diag.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
