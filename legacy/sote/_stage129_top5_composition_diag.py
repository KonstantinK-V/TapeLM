"""
Stage 129 — STORY top5 composition diag (tests local-freq vs content hypothesis).

For each STORY next-word position, build SOTE top5 + BPE beam5, slice:
  both / sote_only / bpe_only / neither (by gold membership)

Report per slice and overall:
  - frac function-like words in top5 (on,to,a,the,and,...)
  - frac content-like (rest)
  - whether top1 is function-like
  - mean unigram-freq rank of top5 words (head bias)
  - gold freq bucket

Also SEEN right: hop set size | hop∩topK | whether atom top1 ∈ hop
  (explains why hops@K failed to lift obj)

Run (CPU-friendly; can overlap 128 on GPU if needed):
  python _stage129_top5_composition_diag.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer  # noqa: E402

from train import (  # noqa: E402
    CKPT,
    RELS,
    RES,
    Config,
    WordIdTransformer,
    _role,
    build_ts_repeat_mix,
    line_subkind,
    load_foundation_85,
    target_freq_bucket,
    build_bigram_table,
)
from _stage109_110_slot_baseline import ensure_100k, _subsample  # noqa: E402
from _stage116_117_hops_morph_follow import (  # noqa: E402
    build_path_memory,
    hop_candidates_for_prefix,
)
from train import FACT_BANK_EXP_F  # noqa: E402

ATOM125 = CKPT / "stage125_same_role_rank.pt"
ATOM100 = CKPT / "stage100_scale_100k.pt"
BPE_TOK = RES / "stage112_bpe_tokenizer.json"
BPE_CKPT = CKPT / "stage112_bpe_baseline.pt"
DEC128 = RES / "stage127_128_decision.json"
LOG = RES / "_stage129_composition_log.txt"
OUT = RES / "stage129_top5_composition.json"

FUNC = {
    "a", "an", "the", "and", "or", "but", "to", "on", "in", "of", "at", "for",
    "with", "from", "by", "as", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "did", "does", "not", "no", "he", "she", "it",
    "they", "we", "you", "i", "his", "her", "their", "my", "your", "this",
    "that", "these", "those", "there", "here", "so", "if", "then", "when",
    "what", "who", "which", "up", "out", "into", "about", "him", "them", "me",
}


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def wait_128_optional(timeout_s=7200, poll=40):
    """If 128 running, wait so we can attach note; composition itself doesn't need it."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC128.exists():
            d = json.loads(DEC128.read_text(encoding="utf-8"))
            if d.get("stage128", {}).get("verdict"):
                log("[wait] 128 done — will include in report")
                return d
        # don't block forever if user only wants diag — max wait short if no process?
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] 128 ... {int(time.time()-t0)}s (diag can proceed without)")
        # after 2 min without 128, proceed anyway
        if time.time() - t0 > 120 and not DEC128.exists():
            log("[wait] proceed without 128 (not finished yet)")
            return {}
    return {}


def load_bpe_beam(device):
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

    return first_words_beam


def frac_func(ws):
    if not ws:
        return 0.0
    return sum(1 for w in ws if w in FUNC) / len(ws)


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"129 composition diag {datetime.now(timezone.utc).isoformat()}")
    try:
        # Don't hard-block on 128 — run diag now (128 holds GPU; use CPU)
        d128 = {}
        if DEC128.exists():
            d128 = json.loads(DEC128.read_text(encoding="utf-8"))

        device = torch.device("cpu")  # leave GPU to 128
        cfg = Config()
        cfg.c87_n_fat, cfg.c87_n_rare = 200, 80
        cfg.c87_fat_copies, cfg.c87_rare_copies = 40, 2
        cfg.c87_seen_hold_frac, cfg.c87_story_keep_frac, cfg.c87_hold_frac = 0.2, 0.5, 0.15
        parent = CKPT / "stage85_foundation_FROZEN.pt"
        if not parent.exists():
            parent = CKPT / "stage85_foundation.pt"
        stack, _ = load_foundation_85(device, cfg, path=parent)
        phrases = ensure_100k()
        train, hold_seen, _, hold_story, _ = build_ts_repeat_mix(phrases, cfg, seed=272)
        for ln in train + hold_seen + hold_story:
            ln.setdefault("subkind", line_subkind(ln))
            ln["split"] = ln.get("bucket", "x")
        words = sorted({w for ln in train + hold_seen + hold_story for w in ln["words"]})
        stoi = {s: i for i, s in enumerate(words)}
        _, unigram, _ = build_bigram_table(train)

        model = WordIdTransformer(len(words), 256, 4, 2, 16, 0.1)
        path = ATOM125 if ATOM125.exists() else ATOM100
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert list(ck["surfaces"]) == words
        model.load_state_dict(ck["word_tf"], strict=True)
        model.eval()
        log(f"[init] {path.name} on CPU")

        log("[bpe] load beam ...")
        bpe_beam = load_bpe_beam(device)

        log("[hopmem] ...")
        # path memory on CPU — may be slow; cap train observe via build_path_memory internal 5000
        mem, by_lr, by_rel = build_path_memory(
            stack, cfg, train, device,
            fact_path=FACT_BANK_EXP_F if FACT_BANK_EXP_F.exists() else None,
        )
        _orig = stack.phrase_fp

        def _safe(ws):
            mw = max(1, int(getattr(stack.phrase, "max_len", 5)) - 1)
            return _orig(ws[-mw:] if len(ws) > mw else ws)

        stack.phrase_fp = _safe  # type: ignore

        hold = _subsample(hold_story, min(350, len(hold_story)), 41)
        slices = defaultdict(lambda: {
            "n": 0,
            "sote_func_frac": [],
            "bpe_func_frac": [],
            "sote_top1_func": 0,
            "bpe_top1_func": 0,
            "sote_mean_uni": [],
            "bpe_mean_uni": [],
            "gold_bucket": Counter(),
            "role": Counter(),
        })
        overall = {"n": 0, "jaccard": []}

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
                    logits = model.logits_from_prefix(ids[:t][-model.max_len :])
                    top5_i = logits.topk(min(5, logits.numel())).indices.tolist()
                    sote5 = [words[int(i)] for i in top5_i]
                    bpe5 = bpe_beam(ws[:t], beam=5)
                    g_s, g_b = gold in sote5, gold in bpe5
                    if g_s and g_b:
                        sl = "both"
                    elif g_s:
                        sl = "sote_only"
                    elif g_b:
                        sl = "bpe_only"
                    else:
                        sl = "neither"
                    role = _role({"prefix_len": t, "target_word": gold}, ws)
                    b = slices[sl]
                    b["n"] += 1
                    b["sote_func_frac"].append(frac_func(sote5))
                    b["bpe_func_frac"].append(frac_func(bpe5))
                    b["sote_top1_func"] += int(sote5[0] in FUNC) if sote5 else 0
                    b["bpe_top1_func"] += int(bpe5[0] in FUNC) if bpe5 else 0
                    b["sote_mean_uni"].append(mean([unigram.get(w, 0) for w in sote5]) or 0)
                    b["bpe_mean_uni"].append(mean([unigram.get(w, 0) for w in bpe5]) or 0)
                    b["gold_bucket"][target_freq_bucket(unigram.get(gold, 0))] += 1
                    b["role"][role] += 1
                    slices["ALL"]["n"] += 1
                    slices["ALL"]["sote_func_frac"].append(frac_func(sote5))
                    slices["ALL"]["bpe_func_frac"].append(frac_func(bpe5))
                    slices["ALL"]["sote_top1_func"] += int(sote5[0] in FUNC) if sote5 else 0
                    slices["ALL"]["bpe_top1_func"] += int(bpe5[0] in FUNC) if bpe5 else 0
                    uni = set(sote5) | set(bpe5)
                    if uni:
                        overall["jaccard"].append(len(set(sote5) & set(bpe5)) / len(uni))
                    overall["n"] += 1

        def pack_slice(name, b):
            n = max(b["n"], 1)
            return {
                "n": b["n"],
                "mean_sote_func_frac_in_top5": mean(b["sote_func_frac"]),
                "mean_bpe_func_frac_in_top5": mean(b["bpe_func_frac"]),
                "frac_sote_top1_func": b["sote_top1_func"] / n,
                "frac_bpe_top1_func": b["bpe_top1_func"] / n,
                "mean_sote_top5_unigram": mean(b["sote_mean_uni"]),
                "mean_bpe_top5_unigram": mean(b["bpe_mean_uni"]),
                "gold_bucket": dict(b["gold_bucket"]),
                "role": dict(b["role"]),
                "delta_func_sote_minus_bpe": (
                    (mean(b["sote_func_frac"]) or 0) - (mean(b["bpe_func_frac"]) or 0)
                ),
            }

        packed = {k: pack_slice(k, v) for k, v in slices.items() if v["n"] or k == "ALL"}

        # hops width diag on SEEN right — why obj not lifting
        hop_stats = {
            "n": 0,
            "hop_size": [],
            "hop_cap_left_rel": [],
            "intersect_K": [],
            "top1_in_hop": 0,
            "gold_in_hop": 0,
            "gold_in_K": 0,
            "narrow_would_keep_gold": 0,  # gold in by_left_rel only
        }
        ev = _subsample(hold_seen, 400, 3)
        with torch.no_grad():
            for ln in ev:
                ws = ln["words"]
                ids = [stoi[w] for w in ws if w in stoi]
                if len(ids) != len(ws):
                    continue
                for t in range(1, len(ws)):
                    if ws[t - 1] not in RELS:
                        continue
                    gold = ws[t]
                    if gold not in stoi:
                        continue
                    logits = model.logits_from_prefix(ids[:t][-model.max_len :])
                    topk = set(int(i) for i in logits.topk(5).indices.tolist())
                    top1 = int(logits.argmax())
                    hop = hop_candidates_for_prefix(ws[:t], by_lr, by_rel, mem, stack, stoi)
                    narrow = set()
                    if len(ws) >= 2:
                        narrow = {stoi[w] for w in by_lr.get((ws[t - 2], ws[t - 1]), set()) if w in stoi}
                    hop_stats["n"] += 1
                    hop_stats["hop_size"].append(len(hop))
                    hop_stats["hop_cap_left_rel"].append(len(narrow))
                    hop_stats["intersect_K"].append(len(hop & topk))
                    hop_stats["top1_in_hop"] += int(top1 in hop)
                    hop_stats["gold_in_hop"] += int(stoi[gold] in hop)
                    hop_stats["gold_in_K"] += int(stoi[gold] in topk)
                    hop_stats["narrow_would_keep_gold"] += int(stoi[gold] in narrow)

        n = max(hop_stats["n"], 1)
        hop_pack = {
            "n": hop_stats["n"],
            "mean_hop_set_size": mean(hop_stats["hop_size"]),
            "mean_left_rel_only_size": mean(hop_stats["hop_cap_left_rel"]),
            "mean_hop_intersect_top5": mean(hop_stats["intersect_K"]),
            "frac_atom_top1_in_hop": hop_stats["top1_in_hop"] / n,
            "frac_gold_in_hop": hop_stats["gold_in_hop"] / n,
            "frac_gold_in_top5": hop_stats["gold_in_K"] / n,
            "frac_gold_in_left_rel_only": hop_stats["narrow_would_keep_gold"] / n,
            "note": (
                "If top1∈hop ~100% and hop set huge, intersect_argmax≈atom argmax — "
                "explains ~0pp lift from hops@K"
            ),
        }

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hypothesis": (
                "If SOTE top5 more function-heavy / higher unigram than BPE especially on "
                "bpe_only → local-freq bias supported; else reject 'SOTE=func BPE=plot'"
            ),
            "slices": packed,
            "mean_jaccard": mean(overall["jaccard"]),
            "hop_width_seen_right": hop_pack,
            "stage128_snapshot": (d128.get("stage128") or {}).get("best") or d128.get("stage128"),
            "verdict_rules": {
                "SUPPORT_local_freq": "delta_func_sote_minus_bpe on bpe_only >= 0.10",
                "REJECT_local_freq": "abs(delta) < 0.05 on bpe_only and ALL",
                "hops_obj_unlikely": "frac_atom_top1_in_hop >= 0.90 and mean_hop_set_size >= 20",
            },
        }

        # auto verdict strings
        bo = packed.get("bpe_only", {})
        delta = bo.get("delta_func_sote_minus_bpe")
        if delta is None:
            hyp = "NA"
        elif delta >= 0.10:
            hyp = "SUPPORT_local_freq"
        elif abs(delta) < 0.05:
            hyp = "REJECT_local_freq"
        else:
            hyp = "WEAK_local_freq"
        hops_obj = (
            "UNLIKELY"
            if (hop_pack["frac_atom_top1_in_hop"] >= 0.90 and (hop_pack["mean_hop_set_size"] or 0) >= 20)
            else "MAYBE_IF_NARROW_left_rel"
        )
        out["auto"] = {"func_hypothesis": hyp, "hops_lift_obj": hops_obj, "bpe_only_delta_func": delta}

        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        log(f"\n=== 129 composition ===")
        for name in ("ALL", "both", "bpe_only", "sote_only", "neither"):
            if name not in packed:
                continue
            p = packed[name]
            log(
                f"  [{name}] n={p['n']} sote_func={100*(p['mean_sote_func_frac_in_top5'] or 0):.1f}% "
                f"bpe_func={100*(p['mean_bpe_func_frac_in_top5'] or 0):.1f}% "
                f"Δ={100*(p['delta_func_sote_minus_bpe'] or 0):+.1f}pp "
                f"top1_func sote={100*p['frac_sote_top1_func']:.1f}% bpe={100*p['frac_bpe_top1_func']:.1f}%"
            )
        log(
            f"  [hop width SEEN right] |hop|={hop_pack['mean_hop_set_size']:.1f} "
            f"|left+rel|={hop_pack['mean_left_rel_only_size']:.1f} "
            f"top1∈hop={100*hop_pack['frac_atom_top1_in_hop']:.1f}% "
            f"gold∈left+rel={100*hop_pack['frac_gold_in_left_rel_only']:.1f}%"
        )
        log(f"  auto: func_hyp={hyp}  hops_obj={hops_obj}")
        log(f"saved {OUT}")

        # append to replay
        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stage 129 composition diag:** func_hyp={hyp}; hops_obj={hops_obj}; "
                f"bpe_only Δfunc={None if delta is None else f'{100*delta:+.1f}pp'}. "
                f"`stage129_top5_composition.json`.\n"
            )
            if "Stage 129 composition" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
