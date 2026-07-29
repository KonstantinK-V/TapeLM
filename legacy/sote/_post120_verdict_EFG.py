"""
After Stage120 finishes:
  1) One-page verdict A–D (mem gain, speed, BPE toks/word, rare buckets)
  2) Dig H: soft top-5 + morph-tolerant diag (NOT a gate; exact@1 remains claim)
  3) Sequential digs E → F → G′ (morph-root positions preferred)

E: who/where entity slots memory (refine 120)
F: projection adapter after fp (256→256)
G: denser attn via morph stem/affix positions (char-pieces fallback)

Run (waits for stage120_decision.json):
  python _post120_verdict_EFG.py
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer  # noqa: E402

from train import (  # noqa: E402
    CKPT,
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
    target_freq_bucket,
    build_bigram_table,
)
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402

DEC120 = RES / "stage120_decision.json"
DEC119 = RES / "stage119_decision.json"
DEC118 = RES / "stage118_decision.json"
BPE_TOK = RES / "stage112_bpe_tokenizer.json"
BPE_CKPT = CKPT / "stage112_bpe_baseline.pt"
ATOM100 = CKPT / "stage100_scale_100k.pt"
LOG = RES / "_post120_verdict_EFG_log.txt"
VERDICT = RES / "POST120_VERDICT_ABCD.txt"
DEC_EFG = RES / "stage121_123_EFG_decision.json"
SIGNALS = RES / "post120_signals_ABCD.json"


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def wait_120(timeout_s: int = 14 * 3600, poll_s: int = 45) -> dict:
    log(f"[wait] for {DEC120} ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC120.exists():
            d = json.loads(DEC120.read_text(encoding="utf-8"))
            if d.get("verdict") is not None or d.get("story_mem_all") is not None:
                log("[wait] 120 done")
                return d
        time.sleep(poll_s)
        if int(time.time() - t0) % 300 < poll_s:
            log(f"[wait] still ... {int(time.time()-t0)}s")
    raise TimeoutError("120 not ready")


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


# ---------- C: BPE toks per word ----------
def measure_toks_per_word(phrases: list[str], n: int = 2000) -> dict:
    if not BPE_TOK.exists():
        return {"available": False}
    tok = Tokenizer.from_file(str(BPE_TOK))
    rng = random.Random(272)
    sample = phrases if len(phrases) <= n else [phrases[i] for i in rng.sample(range(len(phrases)), n)]
    ratios = []
    n_words = n_toks = 0
    for p in sample:
        ws = p.split()
        if not ws:
            continue
        ids = tok.encode(p).ids
        # strip BOS/EOS if present
        bos, eos = tok.token_to_id("[BOS]"), tok.token_to_id("[EOS]")
        ids = [i for i in ids if i not in (bos, eos)]
        ratios.append(len(ids) / max(len(ws), 1))
        n_words += len(ws)
        n_toks += len(ids)
    ratios.sort()
    mid = ratios[len(ratios) // 2] if ratios else 0.0
    return {
        "available": True,
        "n_windows": len(ratios),
        "mean_toks_per_word": (n_toks / max(n_words, 1)),
        "median_toks_per_word": mid,
        "p90": ratios[int(0.9 * (len(ratios) - 1))] if ratios else 0.0,
        "signal": "LIVE" if (n_toks / max(n_words, 1)) >= 2.0 else "WEAK",
    }


# ---------- D: rare buckets SOTE vs BPE ----------
@torch.no_grad()
def rare_bucket_sote(phrases: list[str]) -> dict:
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
    train, hold_seen, hold_rare, hold_story, _ = build_ts_repeat_mix(phrases, cfg, seed=272)
    for ln in train + hold_seen + hold_rare + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")
    words = sorted({w for ln in train + hold_story for w in ln["words"]})
    fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)
    stoi = {s: i for i, s in enumerate(words)}
    model = WordIdTransformer(len(words), 256, 4, 2, 16, 0.1).to(device)
    if ATOM100.exists():
        ck = torch.load(ATOM100, map_location="cpu", weights_only=False)
        if list(ck.get("surfaces", [])) == words:
            model.load_state_dict(ck["word_tf"], strict=True)
        else:
            model.init_from_fps(fps)
            # prefer stage120 if exists
            p120 = CKPT / "stage120_phrase_mem.pt"
            if p120.exists():
                ck2 = torch.load(p120, map_location="cpu", weights_only=False)
                if list(ck2.get("surfaces", [])) == words:
                    model.load_state_dict(ck2["word_tf"], strict=True)
    else:
        model.init_from_fps(fps)
    model.eval()
    hold = _subsample(hold_story, min(800, len(hold_story)), 11)
    suite = eval_id_capacity_suite(model, hold, train, words, stoi, device)
    return {"available": True, "freq": suite.get("freq", {}), "source": "sote_atom"}


@torch.no_grad()
def rare_bucket_bpe(phrases: list[str]) -> dict:
    """Word-level next exact@1 by train-freq bucket using Stage112 GPT if present."""
    if not (BPE_TOK.exists() and BPE_CKPT.exists()):
        return {"available": False}
    from transformers import GPT2Config, GPT2LMHeadModel
    from _stage111_112_follow import _encode_words, bpe_pred_next_word

    cfg = Config()
    cfg.c87_n_fat = 200
    cfg.c87_n_rare = 80
    cfg.c87_fat_copies = 40
    cfg.c87_rare_copies = 2
    cfg.c87_seen_hold_frac = 0.20
    cfg.c87_story_keep_frac = 0.50
    cfg.c87_hold_frac = 0.15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _, _, hold_story, _ = build_ts_repeat_mix(phrases, cfg, seed=272)
    for ln in train + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")
    majority, unigram, _ = build_bigram_table(train)
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
    hold = _subsample(hold_story, min(400, len(hold_story)), 11)
    buckets = defaultdict(lambda: {"n": 0, "h": 0})
    for ln in hold:
        ws = ln["words"]
        for t in range(1, len(ws)):
            gold = ws[t]
            pred = bpe_pred_next_word(model, tok, ws[:t], device)
            ok = int(pred == gold)
            b = target_freq_bucket(unigram.get(gold, 0))
            buckets[b]["n"] += 1
            buckets[b]["h"] += ok
            buckets["ALL"]["n"] += 1
            buckets["ALL"]["h"] += ok
    freq = {
        k: {"n": v["n"], "model": v["h"] / max(v["n"], 1)}
        for k, v in buckets.items()
    }
    return {"available": True, "freq": freq, "source": "bpe112"}


def write_verdict_abcd(d120, d119, d118, toks, rare_s, rare_b) -> dict:
    gain = float(d120.get("gain_pp") or (d120.get("story_mem_all", 0) - d120.get("story_local_all", 0)))
    mem = d120.get("story_mem_all")
    loc = d120.get("story_local_all")
    a_sig = "LIVE" if gain >= 0.03 else ("WEAK" if gain >= 0.015 else "NULL")

    speed = (d119 or {}).get("stage119a_speed") or {}
    b_ver = speed.get("verdict", "NA")
    b_sig = "LIVE" if b_ver in ("SPEED", "PARITY_SPEED") else ("NA" if b_ver == "NA" else "WEAK")

    c_sig = toks.get("signal", "NA") if toks.get("available") else "NA"
    mean_tpw = toks.get("mean_toks_per_word")

    # rare compare
    d_sig = "NA"
    rare_cmp = {}
    if rare_s.get("available") and rare_b.get("available"):
        for band in ("rare1", "low2_5", "mid6_20", "head21p", "ALL"):
            ss = (rare_s.get("freq") or {}).get(band, {})
            bb = (rare_b.get("freq") or {}).get(band, {})
            sm = ss.get("model", ss.get("hit1"))
            bm = bb.get("model", bb.get("hit1"))
            if sm is None or bm is None:
                continue
            rare_cmp[band] = {"sote": sm, "bpe": bm, "delta_pp": bm - sm}
        rare_delta = rare_cmp.get("rare1", rare_cmp.get("low2_5", {})).get("delta_pp")
        if rare_delta is not None:
            d_sig = "LIVE" if rare_delta >= 0.08 else ("WEAK" if rare_delta >= 0.03 else "NULL")

    lines = [
        "SOTE POST-120 VERDICT A–D",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        "",
        "=== A. Phrase-memory gain (Stage120) ===",
        f"  STORY_LOCAL={None if loc is None else f'{100*loc:.1f}%'}  "
        f"STORY_MEM={None if mem is None else f'{100*mem:.1f}%'}  "
        f"gain={100*gain:+.1f}pp  verdict120={d120.get('verdict')}",
        f"  SEEN obj={100*float((d120.get('seen') or {}).get('obj') or 0):.1f}% "
        f"rel={100*float((d120.get('seen') or {}).get('rel') or 0):.1f}%",
        f"  Signal: {a_sig}  (LIVE if gain>=3pp)",
        "",
        "=== B. Speed edge (Stage119a) ===",
        f"  {json.dumps(speed, indent=2) if speed else 'missing'}",
        f"  Signal: {b_sig}",
        "",
        "=== C. BPE toks / word (same windows) ===",
        f"  {json.dumps(toks, indent=2)}",
        f"  Signal: {c_sig}  (LIVE if mean>=2.0)",
        "",
        "=== D. Rare buckets word-level SOTE vs BPE ===",
        f"  SOTE: {json.dumps(rare_s.get('freq'), indent=2) if rare_s.get('available') else 'NA'}",
        f"  BPE:  {json.dumps(rare_b.get('freq'), indent=2) if rare_b.get('available') else 'NA'}",
        f"  cmp:  {json.dumps(rare_cmp, indent=2)}",
        f"  Signal: {d_sig}  (LIVE if BPE-SOTE rare delta>=8pp)",
        "",
        "=== Next digs (queued H → E→F→G′) ===",
        "  H soft top-5 + morph@1 diag (NOT gate)",
        "  E who/where slots — strongest if A LIVE/WEAK",
        "  F fp projection adapter — if geometry/adapt angle still open",
        "  G′ morph-root denser positions — strongest if C/D LIVE",
        "",
        "=== One-liner ===",
        f"  A={a_sig} B={b_sig} C={c_sig} D={d_sig}",
    ]
    text = "\n".join(lines) + "\n"
    VERDICT.write_text(text, encoding="utf-8")
    log("\n" + text)
    out = {
        "A": {"signal": a_sig, "gain_pp": gain, "local": loc, "mem": mem, "verdict120": d120.get("verdict")},
        "B": {"signal": b_sig, "speed": speed},
        "C": {"signal": c_sig, **{k: toks[k] for k in toks if k != "signal"}},
        "D": {"signal": d_sig, "cmp": rare_cmp},
    }
    SIGNALS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ---------- Dig H: top-5 + morph-tolerant (diagnostic only) ----------
@torch.no_grad()
def dig_H_top5_soft_diag(phrases: list[str], signals: dict) -> dict:
    """
    Report hit@1 / hit@5 / morph-in-top5 on Stage100 atom (+ BPE word-beam@5 if ckpt).
    GATE stays exact@1. Signal = size of (hit5-hit1) near-miss band on STORY.
    """
    from _stage116_117_hops_morph_follow import morph_hit

    log("\n======== DIG H soft top-5 + morph diag (NOT gate) ========")
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
    fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)
    stoi = {s: i for i, s in enumerate(words)}
    model = WordIdTransformer(len(words), 256, 4, 2, 16, 0.1).to(device)
    if ATOM100.exists():
        ck = torch.load(ATOM100, map_location="cpu", weights_only=False)
        if list(ck.get("surfaces", [])) == words:
            model.load_state_dict(ck["word_tf"], strict=True)
        else:
            model.init_from_fps(fps)
            log("[H] Stage100 vocab mismatch — fp init (diag degraded)")
    else:
        model.init_from_fps(fps)
    model.eval()

    def eval_soft(hold_lines, tag: str):
        hold = _subsample(hold_lines, min(600, len(hold_lines)), 17)
        buckets = defaultdict(lambda: {"n": 0, "h1": 0, "h5": 0, "morph": 0})
        obj = {"n": 0, "h1": 0, "h5": 0, "morph": 0}
        for ln in hold:
            ws = ln["words"]
            ids = [stoi[w] for w in ws if w in stoi]
            if len(ids) != len(ws):
                continue
            for t in range(1, len(ws)):
                gold = ws[t]
                gid = stoi[gold]
                logits = model.logits_from_prefix(ids[:t][-model.max_len :])
                pred_i = int(logits.argmax())
                top5 = [int(i) for i in logits.topk(min(5, logits.numel())).indices.tolist()]
                h1 = int(pred_i == gid)
                h5 = int(gid in top5)
                mo = int(any(morph_hit(words[i], gold) for i in top5))
                if t >= 1 and ws[t - 1] in ("on", "to"):
                    role = "right"
                elif gold in ("on", "to"):
                    role = "rel"
                else:
                    role = "other"
                for name in (role, "ALL"):
                    buckets[name]["n"] += 1
                    buckets[name]["h1"] += h1
                    buckets[name]["h5"] += h5
                    buckets[name]["morph"] += mo
                if role == "right":
                    obj["n"] += 1
                    obj["h1"] += h1
                    obj["h5"] += h5
                    obj["morph"] += mo

        def pack(b):
            n = max(b["n"], 1)
            return {
                "n": b["n"],
                "hit1": b["h1"] / n,
                "hit5": b["h5"] / n,
                "morph_in_top5": b["morph"] / n,
                "hit5_minus_hit1_pp": (b["h5"] - b["h1"]) / n,
            }

        out = {k: pack(v) for k, v in buckets.items()}
        out["obj"] = pack(obj)
        log(
            f"  [H/{tag}] ALL hit1={out['ALL']['hit1']*100:.1f}% "
            f"hit5={out['ALL']['hit5']*100:.1f}% "
            f"(+{out['ALL']['hit5_minus_hit1_pp']*100:.1f}pp) "
            f"morph@top5={out['ALL']['morph_in_top5']*100:.1f}% | "
            f"obj hit1={out['obj']['hit1']*100:.1f}% hit5={out['obj']['hit5']*100:.1f}%"
        )
        return out

    sote_seen = eval_soft(hold_seen, "SEEN")
    sote_story = eval_soft(hold_story, "STORY")

    bpe_out = {"available": False}
    if BPE_TOK.exists() and BPE_CKPT.exists():
        from transformers import GPT2Config, GPT2LMHeadModel
        from _stage111_112_follow import _encode_words

        tok = Tokenizer.from_file(str(BPE_TOK))
        V = tok.get_vocab_size()
        bos, eos, pad = tok.token_to_id("[BOS]"), tok.token_to_id("[EOS]"), tok.token_to_id("[PAD]")
        conf = GPT2Config(
            vocab_size=V, n_positions=64, n_embd=256, n_layer=4, n_head=4, n_inner=1024,
            bos_token_id=bos, eos_token_id=eos, pad_token_id=pad,
        )
        bpe = GPT2LMHeadModel(conf).to(device)
        ck = torch.load(BPE_CKPT, map_location="cpu", weights_only=False)
        bpe.load_state_dict(ck["model"], strict=False)
        bpe.eval()

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
                    logits = bpe(x).logits[0, -1]
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

        hold = _subsample(hold_story, min(300, len(hold_story)), 17)
        n = h1 = h5 = mo = 0
        for ln in hold:
            ws = ln["words"]
            for t in range(1, len(ws)):
                gold = ws[t]
                cands = first_words_beam(ws[:t], beam=5)
                pred = cands[0] if cands else ""
                n += 1
                h1 += int(pred == gold)
                h5 += int(gold in cands)
                mo += int(any(morph_hit(c, gold) for c in cands) if cands else False)
        bpe_out = {
            "available": True,
            "n": n,
            "hit1": h1 / max(n, 1),
            "hit5": h5 / max(n, 1),
            "morph_in_beam5": mo / max(n, 1),
            "hit5_minus_hit1_pp": (h5 - h1) / max(n, 1),
            "note": "word-level beam@5 first words; STORY hold",
        }
        log(
            f"  [H/BPE STORY] hit1={bpe_out['hit1']*100:.1f}% "
            f"hit5={bpe_out['hit5']*100:.1f}% "
            f"(+{bpe_out['hit5_minus_hit1_pp']*100:.1f}pp) "
            f"morph_beam={bpe_out['morph_in_beam5']*100:.1f}%"
        )

    gap = sote_story.get("ALL", {}).get("hit5_minus_hit1_pp", 0.0)
    if gap >= 0.12:
        verdict = "NEAR_MISS_LIVE"
    elif gap >= 0.06:
        verdict = "NEAR_MISS_WEAK"
    else:
        verdict = "NEAR_MISS_NULL"
    out = {
        "dig": "H",
        "tag": "top5_morph_diag",
        "verdict": verdict,
        "gate": "exact@1_only_unchanged",
        "sote_seen": sote_seen,
        "sote_story": sote_story,
        "bpe_story": bpe_out,
        "supported_by": {k: signals.get(k, {}).get("signal") for k in ("C", "D")},
    }
    (RES / f"stage124_H_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (RES / "stage124_H_top5_decision.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[H] {verdict} STORY hit5-hit1={100*gap:+.1f}pp (diag only)")
    with VERDICT.open("a", encoding="utf-8") as f:
        f.write("\n=== H soft top-5 / morph diag (NOT gate) ===\n")
        f.write(json.dumps({
            "verdict": verdict,
            "sote_story_ALL": sote_story.get("ALL"),
            "sote_story_obj": sote_story.get("obj"),
            "bpe_story": bpe_out,
        }, indent=2) + "\n")
    return out


# ---------- Dig E: who/where slots ----------
def dig_E_entity_slots(phrases: list[str], signals: dict) -> dict:
    """Refine memory: track who/where lexemes from history; stronger boost on those ids."""
    log("\n======== DIG E who/where entity slots ========")
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent85 = CKPT / "stage85_foundation_FROZEN.pt"
    if not parent85.exists():
        parent85 = CKPT / "stage85_foundation.pt"
    stack, _ = load_foundation_85(device, cfg, path=parent85)

    # reuse 120-style episodes lightly from 100k phrases as pseudo-history chains
    # Group consecutive phrases as fake story of length 3–4
    rng = random.Random(272)
    phrases = list(phrases)
    rng.shuffle(phrases)
    episodes = []
    i = 0
    while i + 2 < len(phrases) and len(episodes) < 5000:
        sents = [phrases[i].split(), phrases[i + 1].split(), phrases[i + 2].split()]
        episodes.append(sents)
        i += 3

    def entity_sets(hist):
        who, where = set(), set()
        for ws in hist:
            for j, w in enumerate(ws):
                if w in ("on", "to") and j + 1 < len(ws):
                    where.add(ws[j + 1])
                if j == 0:
                    who.add(w)
                # nouns after a/the
                if w in ("a", "the", "an") and j + 1 < len(ws):
                    who.add(ws[j + 1])
        return who, where

    # build flat examples
    ex_all = []
    for sents in episodes:
        for si in range(len(sents)):
            ex_all.append({
                "words": sents[si],
                "phrase": " ".join(sents[si]),
                "history": sents[:si],
                "bucket": "story",
                "subkind": line_subkind({"words": sents[si]}),
                "split": "x",
            })
    rng.shuffle(ex_all)
    n_h = max(1, int(0.15 * len(ex_all)))
    hold = [e for e in ex_all[:n_h] if e["history"]]
    train = ex_all[n_h:]
    hold = _subsample(hold, min(600, len(hold)), 3)

    words = sorted({w for e in train + hold for w in e["words"]})
    # include history words
    for e in train + hold:
        for h in e["history"]:
            for w in h:
                if w not in words:
                    words.append(w)
    words = sorted(set(words))
    fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)
    stoi = {s: i for i, s in enumerate(words)}
    model = WordIdTransformer(len(words), 256, 4, 2, 16, 0.1).to(device)
    model.init_from_fps(fps)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pairs = []
    for li, ln in enumerate(train):
        ln["split"] = "x"
        for ex in lines_to_pairs([ln], stoi):
            ex["line_i"] = li
            ex["history"] = ln["history"]
            ex["words_full"] = ln["words"]
            pairs.append(ex)

    def boost_from_entities(hist, prefix, alpha_who=4.0, alpha_where=5.0):
        who, where = entity_sets(hist)
        b = {}
        # if prefix ends with on/to prefer where
        if prefix and prefix[-1] in ("on", "to"):
            for w in where:
                if w in stoi:
                    b[stoi[w]] = max(b.get(stoi[w], 0), alpha_where)
        for w in who | where:
            if w in stoi:
                b[stoi[w]] = max(b.get(stoi[w], 0), alpha_who if w in who else alpha_where)
        return b

    @torch.no_grad()
    def eval_gain(use_ent: bool):
        roles = defaultdict(lambda: {"n": 0, "h": 0})
        for ln in hold:
            ws = ln["words"]
            ids = [stoi[w] for w in ws if w in stoi]
            if len(ids) != len(ws):
                continue
            for t in range(1, len(ws)):
                logits = model.logits_from_prefix(ids[:t][-model.max_len :])
                if use_ent and ln["history"]:
                    for wid, b in boost_from_entities(ln["history"], ws[:t]).items():
                        logits[wid] = logits[wid] + b
                pred = words[int(logits.argmax())]
                ok = int(pred == ws[t])
                roles["ALL"]["n"] += 1
                roles["ALL"]["h"] += ok
        return roles["ALL"]["h"] / max(roles["ALL"]["n"], 1)

    rr = random.Random(1)
    best = {"gain": -1.0, "local": 0.0, "mem": 0.0, "state": None}
    for step in range(1, 25001):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, 1e-3, 200)
        ex = rr.choice(pairs)
        packed = collate_word_id_batch([ex], stoi, model.max_len, model.pad_id, device)
        if packed is None:
            continue
        ids, mask, tgt = packed
        logits = model.logits_last_from_batch(ids, mask)
        if ex.get("history"):
            pref = ex["words_full"][: ex["prefix_len"]]
            for wid, b in boost_from_entities(ex["history"], pref).items():
                logits[0, wid] = logits[0, wid] + b
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 2000 == 0 or step == 25000:
            model.eval()
            loc = eval_gain(False)
            mem = eval_gain(True)
            gain = mem - loc
            log(f"  [E] step {step}: LOCAL={loc*100:.1f}% ENT={mem*100:.1f}% gain={gain*100:+.1f}pp")
            if gain >= best["gain"]:
                best.update(gain=gain, local=loc, mem=mem,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            model.train()

    verdict = "PASS" if best["gain"] >= 0.03 else ("PARTIAL" if best["gain"] >= 0.015 else "NULL")
    ck = CKPT / "stage121_E_entity_slots.pt"
    if best["state"] is not None:
        torch.save({"word_tf": best["state"], "surfaces": words}, ck)
    out = {
        "dig": "E", "tag": "entity_who_where", "verdict": verdict,
        "local": best["local"], "ent_mem": best["mem"], "gain_pp": best["gain"],
        "supported_by": signals.get("A", {}).get("signal"), "ckpt": str(ck),
    }
    (RES / f"stage121_E_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[E] {verdict} gain={100*best['gain']:+.1f}pp")
    return out


# ---------- Dig F: projection adapter ----------
def dig_F_proj_adapter(phrases: list[str], signals: dict) -> dict:
    log("\n======== DIG F fp projection adapter ========")
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent85 = CKPT / "stage85_foundation_FROZEN.pt"
    if not parent85.exists():
        parent85 = CKPT / "stage85_foundation.pt"
    stack, _ = load_foundation_85(device, cfg, path=parent85)
    cfg.c87_n_fat = 200
    cfg.c87_fat_copies = 40
    cfg.c87_n_rare = 80
    cfg.c87_rare_copies = 2
    cfg.c87_seen_hold_frac = 0.2
    cfg.c87_story_keep_frac = 0.5
    cfg.c87_hold_frac = 0.15
    train, hold_seen, _, hold_story, _ = build_ts_repeat_mix(phrases, cfg, seed=272)
    for ln in train + hold_seen + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")
    words = sorted({w for ln in train + hold_seen + hold_story for w in ln["words"]})
    raw_fps = torch.stack([stack.w(w).detach() for w in words], 0).to(device)
    stoi = {s: i for i, s in enumerate(words)}

    proj = torch.nn.Linear(256, 256, bias=False).to(device)
    torch.nn.init.eye_(proj.weight)
    model = WordIdTransformer(len(words), 256, 4, 2, 16, 0.1).to(device)

    def refresh_emb():
        with torch.no_grad():
            adapted = F.normalize(proj(raw_fps), dim=-1)
            model.init_from_fps(adapted)

    refresh_emb()
    opt = torch.optim.Adam(list(model.parameters()) + list(proj.parameters()), lr=1e-3)
    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p = lines_to_pairs(fat, stoi)
    story_p = lines_to_pairs(_subsample(story, 40000, 5), stoi)
    rr = random.Random(2)
    ev_seen = _subsample(hold_seen, 600, 1)
    ev_story = _subsample(hold_story, 400, 2)
    best = {"obj": 0.0, "rel": 0.0, "story": 0.0, "state": None, "proj": None}
    for step in range(1, 30001):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, 1e-3, 200)
        # periodically tie emb to proj(fp) so adapter stays meaningful
        if step % 50 == 1:
            adapted = F.normalize(proj(raw_fps), dim=-1)
            with torch.no_grad():
                w = torch.zeros_like(model.tok.weight)
                w[: len(words)] = adapted
                model.tok.weight.copy_(w)
            model.tok.weight.requires_grad_(True)
        ex = [rr.choice(fat_p) for _ in range(6)] + [rr.choice(story_p) for _ in range(2)]
        packed = collate_word_id_batch(ex, stoi, 16, model.pad_id, device)
        if packed is None:
            continue
        ids, mask, tgt = packed
        loss = F.cross_entropy(model.logits_last_from_batch(ids, mask), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 2000 == 0 or step == 30000:
            model.eval()
            seen = eval_id_capacity_suite(model, ev_seen, train, words, stoi, device)
            story_e = eval_id_capacity_suite(model, ev_story, train, words, stoi, device)
            obj = seen["obj"]["hit1"]
            rel = seen["roles"].get("rel", {}).get("hit1", 0.0)
            st = story_e["roles"].get("ALL", {}).get("hit1", 0.0)
            log(f"  [F] step {step}: obj={obj*100:.1f}% rel={rel*100:.1f}% STORY={st*100:.1f}%")
            if obj >= best["obj"] and rel >= 0.5:
                best.update(obj=obj, rel=rel, story=st,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            proj=proj.state_dict())
            model.train()

    lift = best["story"] - REF100["story_all"]
    verdict = "PASS" if lift >= 0.03 and best["rel"] >= 0.7 else (
        "PARTIAL" if lift >= 0.015 else "PARITY"
    )
    ck = CKPT / "stage122_F_proj_adapter.pt"
    torch.save({"word_tf": best["state"], "proj": best["proj"], "surfaces": words}, ck)
    out = {
        "dig": "F", "tag": "fp_proj_adapter", "verdict": verdict,
        "seen": {"obj": best["obj"], "rel": best["rel"]},
        "story_all": best["story"], "story_lift_pp": lift, "ckpt": str(ck),
        "supported_by_C_D": {"C": signals.get("C", {}).get("signal"), "D": signals.get("D", {}).get("signal")},
    }
    (RES / f"stage122_F_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[F] {verdict} STORY={100*best['story']:.1f}%")
    return out


# ---------- Dig G: denser steps without BPE ----------
def dig_G_dense_steps(phrases: list[str], signals: dict) -> dict:
    """
    Represent each word as [STEM_ID-ish surface char-ngram proxy]: actually
    split each word into overlapping pieces of len<=3 as extra tokens in sequence
    so attn has denser positions, while TARGET remains whole word at word boundaries.
    Simpler approach: duplicate each word id twice in the prefix (word, word) to
    double positions — crude density dig.
    Better: insert morph/char pieces via hashing into extended vocab rows init from fp.
    Here: for prefix words w1..wk predict wk+1 using sequence
      [w1, #a, #b, w2, #c, ...] where #x are char-trigram ids in extended emb.
    """
    log("\n======== DIG G denser attn steps (char pieces) ========")
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent85 = CKPT / "stage85_foundation_FROZEN.pt"
    if not parent85.exists():
        parent85 = CKPT / "stage85_foundation.pt"
    stack, _ = load_foundation_85(device, cfg, path=parent85)
    cfg.c87_n_fat = 200
    cfg.c87_fat_copies = 40
    cfg.c87_n_rare = 80
    cfg.c87_rare_copies = 2
    cfg.c87_seen_hold_frac = 0.2
    cfg.c87_story_keep_frac = 0.5
    cfg.c87_hold_frac = 0.15
    train, hold_seen, _, hold_story, _ = build_ts_repeat_mix(phrases, cfg, seed=272)
    for ln in train + hold_seen + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")
    words = sorted({w for ln in train + hold_seen + hold_story for w in ln["words"]})

    # char pieces
    pieces = set()
    for w in words:
        if len(w) <= 3:
            pieces.add(f"#{w}")
        else:
            for i in range(0, len(w) - 1):
                pieces.add("#" + w[i : i + 3] if i + 3 <= len(w) else "#" + w[i:])
    pieces = sorted(pieces)[:4000]
    surfaces = words + pieces
    stoi = {s: i for i, s in enumerate(surfaces)}
    # fps: words from stack; pieces from mean of chars via stack.w on piece body
    fps_list = []
    for s in surfaces:
        if s.startswith("#"):
            body = s[1:] or "a"
            try:
                fps_list.append(stack.w(body).detach())
            except Exception:
                fps_list.append(torch.zeros(cfg.dim, device=device))
        else:
            fps_list.append(stack.w(s).detach())
    fps = F.normalize(torch.stack(fps_list, 0), dim=-1).to(device)

    def expand_prefix(ws):
        ids = []
        for w in ws:
            ids.append(stoi[w])
            # add up to 2 pieces
            added = 0
            if len(w) <= 3:
                p = f"#{w}"
                if p in stoi:
                    ids.append(stoi[p])
            else:
                for i in range(0, len(w) - 1, 2):
                    p = "#" + w[i : i + 3]
                    if p not in stoi:
                        p = "#" + w[i : i + 2]
                    if p in stoi:
                        ids.append(stoi[p])
                        added += 1
                    if added >= 2:
                        break
        return ids

    model = WordIdTransformer(len(surfaces), 256, 4, 2, max_len=32, dropout=0.1).to(device)
    model.init_from_fps(fps)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    rr = random.Random(3)

    def sample_ex():
        ln = rr.choice(fat) if rr.random() < 0.75 else rr.choice(story)
        ws = ln["words"]
        if len(ws) < 2:
            return None
        t = rr.randrange(1, len(ws))
        pref = expand_prefix(ws[:t])[-model.max_len :]
        gold = ws[t]
        if gold not in stoi or not pref:
            return None
        return pref, stoi[gold]

    @torch.no_grad()
    def eval_hold(hold_lines):
        # evaluate word-level with expanded prefixes
        n = h = 0
        for ln in hold_lines:
            ws = ln["words"]
            for t in range(1, len(ws)):
                gold = ws[t]
                if gold not in stoi:
                    continue
                pref = expand_prefix(ws[:t])[-model.max_len :]
                if not pref:
                    continue
                pred = surfaces[model.pred_id(pref)]
                n += 1
                h += int(pred == gold)
        return h / max(n, 1)

    ev_story = _subsample(hold_story, 400, 2)
    ev_seen = _subsample(hold_seen, 400, 1)
    best = {"story": 0.0, "seen_all": 0.0, "state": None}
    for step in range(1, 30001):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, 1e-3, 200)
        batch_prefs, batch_tgt = [], []
        for _ in range(8):
            s = sample_ex()
            if s is None:
                continue
            batch_prefs.append(s[0])
            batch_tgt.append(s[1])
        if not batch_prefs:
            continue
        tmax = max(len(p) for p in batch_prefs)
        bsz = len(batch_prefs)
        ids = torch.full((bsz, tmax), model.pad_id, dtype=torch.long, device=device)
        mask = torch.ones((bsz, tmax), dtype=torch.bool, device=device)
        for i, p in enumerate(batch_prefs):
            ids[i, -len(p) :] = torch.tensor(p, device=device)
            mask[i, -len(p) :] = False
        tgt = torch.tensor(batch_tgt, device=device)
        loss = F.cross_entropy(model.logits_last_from_batch(ids, mask), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 2000 == 0 or step == 30000:
            model.eval()
            st = eval_hold(ev_story)
            sn = eval_hold(ev_seen)
            log(f"  [G] step {step}: STORY_ALL~{st*100:.1f}% SEEN_ALL~{sn*100:.1f}%")
            if st >= best["story"]:
                best.update(story=st, seen_all=sn,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            model.train()

    lift = best["story"] - REF100["story_all"]
    verdict = "PASS" if lift >= 0.03 else ("PARTIAL" if lift >= 0.015 else "PARITY")
    ck = CKPT / "stage123_G_dense_steps.pt"
    torch.save({"word_tf": best["state"], "surfaces": surfaces}, ck)
    out = {
        "dig": "G", "tag": "char_piece_dense_steps", "verdict": verdict,
        "story_all": best["story"], "seen_all": best["seen_all"],
        "story_lift_pp": lift, "mean_expand_hint": "up_to_+2_pieces_per_word",
        "supported_by_C": signals.get("C", {}).get("signal"), "ckpt": str(ck),
    }
    (RES / f"stage123_G_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[G] {verdict} STORY~{100*best['story']:.1f}%")
    return out


def main():
    RES.mkdir(exist_ok=True)
    resume = SIGNALS.exists() and DEC120.exists() and VERDICT.exists()
    # append mode on resume so A–D page is kept
    if not resume and LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"Post120 verdict+EFG start {datetime.now(timezone.utc).isoformat()} resume={resume}")
    try:
        if resume:
            log("[resume] skip wait/A–D — load signals; run H→E→F→G")
            signals = json.loads(SIGNALS.read_text(encoding="utf-8-sig"))
            phrases = ensure_100k()
        else:
            d120 = wait_120()
            d119 = load_json(DEC119)
            d118 = load_json(DEC118)
            phrases = ensure_100k()

            log("[C] toks/word ...")
            toks = measure_toks_per_word(phrases)
            log("[D] rare SOTE ...")
            rare_s = rare_bucket_sote(phrases)
            log("[D] rare BPE ...")
            rare_b = rare_bucket_bpe(phrases)

            signals = write_verdict_abcd(d120, d119, d118, toks, rare_s, rare_b)

        # H diag first (cheap), then E→F→G′
        rH = dig_H_top5_soft_diag(phrases, signals)
        rE = dig_E_entity_slots(phrases, signals)
        rF = dig_F_proj_adapter(phrases, signals)
        rG = dig_G_dense_steps(phrases, signals)

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals_ABCD": signals,
            "H": rH,
            "E": rE,
            "F": rF,
            "G": rG,
            "verdict_file": str(VERDICT),
        }
        DEC_EFG.write_text(json.dumps(out, indent=2), encoding="utf-8")

        with VERDICT.open("a", encoding="utf-8") as f:
            f.write("\n=== E/F/G results ===\n")
            f.write(f"H {rH['verdict']} (diag; gate=exact@1)\n")
            f.write(f"E {rE['verdict']} gain={100*rE['gain_pp']:+.1f}pp\n")
            f.write(f"F {rF['verdict']} STORY={100*rF['story_all']:.1f}% lift={100*rF['story_lift_pp']:+.1f}pp\n")
            f.write(f"G {rG['verdict']} STORY~{100*rG['story_all']:.1f}% lift={100*rG['story_lift_pp']:+.1f}pp\n")

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Post-120 A–D + H/E/F/G:** see `{VERDICT.name}`; "
                f"H {rH['verdict']}, E {rE['verdict']}, F {rF['verdict']}, G {rG['verdict']}. "
                f"`stage121_123_EFG_decision.json` / `stage124_H_top5_decision.json`.\n"
            )
            if "Post-120 A–D + H/E/F/G:" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")
        log("DONE verdict+H+EFG")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
