"""
Stage 130–131 after stopping membership hops@K (128).

130 Compare-layer on atom top-K + hop feats (NOT hop membership filter):
  For each c in atom top5:
    feats = [
      logit_z (z among K), rank_norm, 1{left+rel}, log1p(count_left_rel_c),
      max_mem_sim(right=c), 1{in_wide_hop}, log1p(unigram), 1{is_func}
    ]
  score = Linear/MLP(feats); pred = argmax_K score
  Train: CE on K when gold in K; else light full-V CE. Hold rel via non-right CE.
  Gate: SEEN obj +3pp vs atom@1 OR +3pp vs 128-eval0; rel>=0.70

131 Func-bias push (after 130):
  On STORY/other: margin / CE that penalizes function-word top1 when gold is content;
  optional upweight non-func targets. Report STORY + func% in top5.

Cross-attn atom↔memory (candidates outside topK) = later dig if 130 helps SEEN
but STORY out5 stays high — see plan note.

Run:
  python _stage130_131_compare_func.py
"""
from __future__ import annotations

import json
import math
import random
import sys
import traceback
from collections import Counter, defaultdict
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
    _role,
    _warmup_then_constant,
    build_bigram_table,
    build_ts_repeat_mix,
    collate_word_id_batch,
    eval_id_capacity_suite,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
)
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402
from _stage116_117_hops_morph_follow import build_path_memory  # noqa: E402

ATOM125 = CKPT / "stage125_same_role_rank.pt"
ATOM100 = CKPT / "stage100_scale_100k.pt"
LOG = RES / "_stage130_131_log.txt"
DEC = RES / "stage130_131_decision.json"

FUNC = {
    "a", "an", "the", "and", "or", "but", "to", "on", "in", "of", "at", "for",
    "with", "from", "by", "as", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "did", "does", "not", "no", "he", "she", "it",
    "they", "we", "you", "i", "his", "her", "their", "my", "your", "this",
    "that", "these", "those", "there", "here", "so", "if", "then", "when",
    "what", "who", "which", "up", "out", "into", "about", "him", "them", "me",
}
FEAT_DIM = 8
K = 5


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


class TopKCompare(nn.Module):
    def __init__(self, d_in=FEAT_DIM, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats):  # [K, d] -> [K]
        return self.net(feats).squeeze(-1)


def setup(phrases, device):
    cfg = Config()
    cfg.c87_n_fat, cfg.c87_n_rare = 200, 80
    cfg.c87_fat_copies, cfg.c87_rare_copies = 40, 2
    cfg.c87_seen_hold_frac, cfg.c87_story_keep_frac, cfg.c87_hold_frac = 0.2, 0.5, 0.15
    parent = CKPT / "stage85_foundation_FROZEN.pt"
    if not parent.exists():
        parent = CKPT / "stage85_foundation.pt"
    stack, _ = load_foundation_85(device, cfg, path=parent)
    train, hold_seen, _, hold_story, _ = build_ts_repeat_mix(phrases, cfg, seed=272)
    for ln in train + hold_seen + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")
    words = sorted({w for ln in train + hold_seen + hold_story for w in ln["words"]})
    stoi = {s: i for i, s in enumerate(words)}
    fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)
    return cfg, stack, train, hold_seen, hold_story, words, stoi, fps


def load_atom(words, fps, device):
    model = WordIdTransformer(len(words), 256, 4, 2, 16, 0.1).to(device)
    path = ATOM125 if ATOM125.exists() else ATOM100
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if list(ck.get("surfaces", [])) == words:
        model.load_state_dict(ck["word_tf"], strict=True)
        log(f"[init] {path.name}")
    else:
        model.init_from_fps(fps)
        log("[init] fp")
    return model


def build_triple_counts(train):
    cnt = Counter()
    by_lr = defaultdict(set)
    for ln in train:
        ws = ln["words"]
        for i, w in enumerate(ws):
            if w in RELS and i > 0 and i + 1 < len(ws):
                left, rel, right = ws[i - 1], w, ws[i + 1]
                cnt[(left, rel, right)] += 1
                by_lr[(left, rel)].add(right)
    return cnt, by_lr


def mem_sim_for_right(mem, stack, cue_ws, right_word, cache):
    """Max retrieve sim among top mem hits whose meta.right == right_word."""
    key = (tuple(cue_ws[-3:]), right_word)
    if key in cache:
        return cache[key]
    try:
        mw = max(1, int(getattr(stack.phrase, "max_len", 5)) - 1)
        cue = cue_ws[-mw:]
        fp = stack.phrase_fp(cue)
        best = 0.0
        for name, sim in mem.topk(fp, k=12):
            meta = mem.fact_meta.get(name) or {}
            if meta.get("right") == right_word:
                best = max(best, float(sim))
        cache[key] = best
        return best
    except Exception:
        cache[key] = 0.0
        return 0.0


def feats_for_topk(
    logits, topk_ids, words, stoi, ws_prefix, unigram, triple_cnt, by_lr, mem, stack, cache, device
):
    """Return tensor [K, FEAT_DIM] for candidates topk_ids (length K)."""
    Kloc = len(topk_ids)
    raw = logits[torch.tensor(topk_ids, device=device)]
    # z-score among K
    mu, sig = raw.mean(), raw.std().clamp(min=1e-6)
    z = (raw - mu) / sig
    left = ws_prefix[-2] if len(ws_prefix) >= 2 else ""
    rel = ws_prefix[-1] if ws_prefix else ""
    narrow = by_lr.get((left, rel), set()) if rel in RELS else set()
    # wide hop approx = narrow or by_rel filled outside — use narrow + mem rights from topk retrieve
    rows = []
    for r, wid in enumerate(topk_ids):
        w = words[wid]
        in_lr = 1.0 if w in narrow else 0.0
        cnt = float(triple_cnt.get((left, rel, w), 0)) if rel in RELS else 0.0
        sim = mem_sim_for_right(mem, stack, ws_prefix, w, cache) if rel in RELS else 0.0
        # membership in narrow∪{retrieved rights} — use sim>0 or in_lr as soft in_hop
        in_hop = 1.0 if (in_lr > 0 or sim > 0) else 0.0
        uni = math.log1p(float(unigram.get(w, 0)))
        is_f = 1.0 if w in FUNC else 0.0
        rows.append([
            float(z[r]),
            r / max(Kloc - 1, 1),
            in_lr,
            math.log1p(cnt),
            sim,
            in_hop,
            uni / 10.0,  # scale
            is_f,
        ])
    return torch.tensor(rows, dtype=torch.float32, device=device)


@torch.no_grad()
def eval_compare(model, scorer, hold, words, stoi, unigram, triple_cnt, by_lr, mem, stack, device, K=5):
    roles = defaultdict(lambda: {"n": 0, "atom": 0, "cmp": 0})
    obj = {"n": 0, "atom": 0, "cmp": 0, "oracle": 0}
    cache = {}
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
            topk_ids = [int(i) for i in logits.topk(min(K, logits.numel())).indices.tolist()]
            atom_pred = topk_ids[0]
            feats = feats_for_topk(
                logits, topk_ids, words, stoi, ws[:t], unigram, triple_cnt, by_lr, mem, stack, cache, device
            )
            cmp_pred = topk_ids[int(scorer(feats).argmax())]
            role = _role({"prefix_len": t, "target_word": gold}, ws)
            for name in (role, "ALL"):
                roles[name]["n"] += 1
                roles[name]["atom"] += int(atom_pred == gid)
                roles[name]["cmp"] += int(cmp_pred == gid)
            if ws[t - 1] in RELS:
                obj["n"] += 1
                obj["atom"] += int(atom_pred == gid)
                obj["cmp"] += int(cmp_pred == gid)
                obj["oracle"] += int(gid in topk_ids)

    def pack(d):
        n = max(d["n"], 1)
        return {k: (d[k] / n if k != "n" else d[k]) for k in d}

    return {"roles": {k: pack(v) for k, v in roles.items()}, "obj": pack(obj)}


def dig_130(phrases):
    log("\n======== 130 compare-layer topK + hop feats ========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, stack, train, hold_seen, hold_story, words, stoi, fps = setup(phrases, device)
    model = load_atom(words, fps, device)
    # freeze atom initially — train only scorer; then optional joint
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    log("[mem] ...")
    mem, by_lr_mem, by_rel = build_path_memory(
        stack, cfg, train, device, fact_path=FACT_BANK_EXP_F if FACT_BANK_EXP_F.exists() else None
    )
    _orig = stack.phrase_fp

    def _safe(ws):
        mw = max(1, int(getattr(stack.phrase, "max_len", 5)) - 1)
        return _orig(ws[-mw:] if len(ws) > mw else ws)

    stack.phrase_fp = _safe  # type: ignore

    triple_cnt, by_lr = build_triple_counts(train)
    _, unigram, _ = build_bigram_table(train)
    scorer = TopKCompare().to(device)
    opt = torch.optim.Adam(scorer.parameters(), lr=1e-3)

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p = lines_to_pairs(fat, stoi)
    story_p = lines_to_pairs(_subsample(story, 40000, 9), stoi)
    right_p = [ex for ex in fat_p + story_p if _role(ex, ex["phrase"].split()) == "right"]
    other_p = [ex for ex in fat_p if _role(ex, ex["phrase"].split()) != "right"]
    rr = random.Random(13)
    ev_seen = _subsample(hold_seen, 600, 1)
    ev_story = _subsample(hold_story, 400, 2)

    # identity-ish init eval with random scorer
    e0 = eval_compare(model, scorer, ev_seen, words, stoi, unigram, triple_cnt, by_lr, mem, stack, device)
    log(
        f"  [130 eval0] SEEN obj atom={100*e0['obj']['atom']:.1f}% cmp={100*e0['obj']['cmp']:.1f}% "
        f"oracle={100*e0['obj']['oracle']:.1f}%"
    )

    best = {
        "obj_cmp": e0["obj"]["cmp"],
        "obj_atom": e0["obj"]["atom"],
        "rel": 0.0,
        "story_cmp": 0.0,
        "state_scorer": {k: v.detach().cpu().clone() for k, v in scorer.state_dict().items()},
    }
    cache_train = {}

    for step in range(1, 20001):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, 1e-3, 200)
        losses = []
        for _ in range(6):
            if not right_p:
                break
            ex = rr.choice(right_p)
            ws = ex["phrase"].split()
            packed = collate_word_id_batch([ex], stoi, model.max_len, model.pad_id, device)
            if packed is None:
                continue
            ids, mask, tgt = packed
            with torch.no_grad():
                logits = model.logits_last_from_batch(ids, mask)[0]
                topk_ids = [int(i) for i in logits.topk(min(K, logits.numel())).indices.tolist()]
            gold = int(tgt[0])
            if gold not in topk_ids:
                continue
            feats = feats_for_topk(
                logits.detach(), topk_ids, words, stoi, ws[: ex["prefix_len"]],
                unigram, triple_cnt, by_lr, mem, stack, cache_train, device,
            )
            scores = scorer(feats)
            target = torch.tensor([topk_ids.index(gold)], device=device)
            losses.append(F.cross_entropy(scores.unsqueeze(0), target))
        # keep scorer calibrated on non-right: prefer atom ranking (imitate z)
        for _ in range(2):
            if not other_p:
                break
            ex = rr.choice(other_p)
            packed = collate_word_id_batch([ex], stoi, model.max_len, model.pad_id, device)
            if packed is None:
                continue
            ids, mask, tgt = packed
            with torch.no_grad():
                logits = model.logits_last_from_batch(ids, mask)[0]
                topk_ids = [int(i) for i in logits.topk(min(K, logits.numel())).indices.tolist()]
            gold = int(tgt[0])
            if gold not in topk_ids:
                continue
            feats = feats_for_topk(
                logits.detach(), topk_ids, words, stoi, ex["phrase"].split()[: ex["prefix_len"]],
                unigram, triple_cnt, by_lr, mem, stack, cache_train, device,
            )
            scores = scorer(feats)
            target = torch.tensor([topk_ids.index(gold)], device=device)
            losses.append(0.3 * F.cross_entropy(scores.unsqueeze(0), target))
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 2000 == 0 or step == 20000:
            scorer.eval()
            s = eval_compare(model, scorer, ev_seen, words, stoi, unigram, triple_cnt, by_lr, mem, stack, device)
            st = eval_compare(model, scorer, ev_story, words, stoi, unigram, triple_cnt, by_lr, mem, stack, device)
            # rel from atom (frozen)
            suite = eval_id_capacity_suite(model, ev_seen, train, words, stoi, device)
            rel = suite["roles"].get("rel", {}).get("hit1", 0.0)
            log(
                f"  [130] step {step}: SEEN obj atom={100*s['obj']['atom']:.1f}% "
                f"cmp={100*s['obj']['cmp']:.1f}% oracle={100*s['obj']['oracle']:.1f}% "
                f"rel={100*rel:.1f}% | STORY cmp={100*st['roles']['ALL']['cmp']:.1f}%"
            )
            if rel >= 0.70 and s["obj"]["cmp"] >= best["obj_cmp"]:
                best.update(
                    obj_cmp=s["obj"]["cmp"], obj_atom=s["obj"]["atom"], rel=rel,
                    story_cmp=st["roles"]["ALL"]["cmp"],
                    state_scorer={k: v.detach().cpu().clone() for k, v in scorer.state_dict().items()},
                    eval_seen=s, eval_story=st,
                )
            scorer.train()

    scorer.load_state_dict(best["state_scorer"])
    lift = best["obj_cmp"] - best["obj_atom"]
    lift_ref = best["obj_cmp"] - REF100["obj"]
    if best["rel"] >= 0.70 and (lift >= 0.03 or lift_ref >= 0.03):
        verdict = "PASS"
    elif best["rel"] >= 0.70 and (lift >= 0.015 or lift_ref >= 0.015):
        verdict = "PARTIAL"
    else:
        verdict = "PARITY"

    ck = CKPT / "stage130_topk_compare.pt"
    torch.save({
        "scorer": best["state_scorer"],
        "surfaces": words,
        "atom": ATOM125.name if ATOM125.exists() else ATOM100.name,
        "K": K,
    }, ck)
    out = {
        "dig": "130_topk_compare",
        "verdict": verdict,
        "seen_obj_atom": best["obj_atom"],
        "seen_obj_cmp": best["obj_cmp"],
        "cmp_minus_atom_pp": lift,
        "obj_lift_vs_ref_pp": lift_ref,
        "rel": best["rel"],
        "story_cmp": best["story_cmp"],
        "ckpt": str(ck),
        "feats": "z_logit,rank,in_left_rel,log_count,mem_sim,in_hop_soft,log_uni,is_func",
    }
    (RES / f"stage130_compare_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[130] {verdict} cmp_obj={100*best['obj_cmp']:.1f}% (atom {100*best['obj_atom']:.1f}% Δ={100*lift:+.1f}pp)")
    return out, model, scorer, stack, cfg, train, hold_seen, hold_story, words, stoi, fps, unigram, triple_cnt, by_lr, mem, device


def dig_131_func_bias(model, phrases, ctx):
    log("\n======== 131 func-bias push ========")
    (out130, model, scorer, stack, cfg, train, hold_seen, hold_story, words, stoi, fps,
     unigram, triple_cnt, by_lr, mem, device) = ctx
    # unfreeze atom lightly; keep scorer
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(list(model.parameters()) + list(scorer.parameters()), lr=3e-4)

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p = lines_to_pairs(fat, stoi)
    story_p = lines_to_pairs(_subsample(story, 50000, 11), stoi)
    rr = random.Random(17)
    ev_seen = _subsample(hold_seen, 600, 1)
    ev_story = _subsample(hold_story, 400, 2)
    cache = {}

    def func_frac_top5(logits):
        top = [words[int(i)] for i in logits.topk(min(5, logits.numel())).indices.tolist()]
        return sum(1 for w in top if w in FUNC) / max(len(top), 1)

    best = {
        "story": -1.0,
        "obj": 0.0,
        "rel": 0.0,
        "func_top5": 1.0,
        "state_m": None,
        "state_s": None,
    }

    for step in range(1, 25001):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, 3e-4, 200)
        losses = []
        # 5x story/other with func penalty if gold is content
        for _ in range(5):
            ex = rr.choice(story_p) if story_p else rr.choice(fat_p)
            gold_w = ex["target_word"]
            packed = collate_word_id_batch([ex], stoi, model.max_len, model.pad_id, device)
            if packed is None:
                continue
            ids, mask, tgt = packed
            logits = model.logits_last_from_batch(ids, mask)
            ce = F.cross_entropy(logits, tgt)
            # if gold content: push down mass on FUNC vocab (soft)
            if gold_w not in FUNC:
                func_ids = [stoi[w] for w in FUNC if w in stoi]
                if func_ids:
                    # penalize mean logit of func words relative to gold
                    fl = logits[0, func_ids].mean()
                    gl = logits[0, int(tgt[0])]
                    ce = ce + 0.15 * F.relu(fl + 0.5 - gl)
            losses.append(ce)
        # 3x fat right — keep SEEN + compare CE when gold in topK
        for _ in range(3):
            ex = rr.choice(fat_p)
            packed = collate_word_id_batch([ex], stoi, model.max_len, model.pad_id, device)
            if packed is None:
                continue
            ids, mask, tgt = packed
            logits = model.logits_last_from_batch(ids, mask)[0]
            losses.append(F.cross_entropy(logits.unsqueeze(0), tgt))
            gold = int(tgt[0])
            topk_ids = [int(i) for i in logits.topk(min(K, logits.numel())).indices.tolist()]
            if gold in topk_ids and _role(ex, ex["phrase"].split()) == "right":
                feats = feats_for_topk(
                    logits.detach(), topk_ids, words, stoi, ex["phrase"].split()[: ex["prefix_len"]],
                    unigram, triple_cnt, by_lr, mem, stack, cache, device,
                )
                scores = scorer(feats)
                losses.append(F.cross_entropy(scores.unsqueeze(0), torch.tensor([topk_ids.index(gold)], device=device)))
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 2000 == 0 or step == 25000:
            model.eval()
            scorer.eval()
            suite = eval_id_capacity_suite(model, ev_seen, train, words, stoi, device)
            st_suite = eval_id_capacity_suite(model, ev_story, train, words, stoi, device)
            cmp_s = eval_compare(model, scorer, ev_seen, words, stoi, unigram, triple_cnt, by_lr, mem, stack, device)
            # mean func in top5 on story sample
            ff = []
            with torch.no_grad():
                for ln in ev_story[:80]:
                    ws = ln["words"]
                    ids = [stoi[w] for w in ws if w in stoi]
                    if len(ids) != len(ws) or len(ws) < 2:
                        continue
                    for t in range(1, min(len(ws), 4)):
                        logits = model.logits_from_prefix(ids[:t][-model.max_len :])
                        ff.append(func_frac_top5(logits))
            fmean = sum(ff) / max(len(ff), 1)
            obj = cmp_s["obj"]["cmp"]
            rel = suite["roles"].get("rel", {}).get("hit1", 0.0)
            st_all = st_suite["roles"].get("ALL", {}).get("hit1", 0.0)
            log(
                f"  [131] step {step}: STORY={100*st_all:.1f}% obj_cmp={100*obj:.1f}% "
                f"rel={100*rel:.1f}% story_top5_func={100*fmean:.1f}%"
            )
            if rel >= 0.70 and st_all >= best["story"] and obj >= REF100["obj"] - 0.05:
                best.update(
                    story=st_all, obj=obj, rel=rel, func_top5=fmean,
                    state_m={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    state_s={k: v.detach().cpu().clone() for k, v in scorer.state_dict().items()},
                )
            model.train()
            scorer.train()

    lift = best["story"] - REF100["story_all"]
    # compare to stage129 ALL sote func ~62%
    func_drop = 0.62 - best["func_top5"]
    if best["rel"] >= 0.70 and lift >= 0.03 and func_drop >= 0.05:
        verdict = "PASS"
    elif best["rel"] >= 0.70 and (lift >= 0.015 or func_drop >= 0.05):
        verdict = "PARTIAL"
    else:
        verdict = "PARITY"

    ck = CKPT / "stage131_func_bias.pt"
    torch.save({"word_tf": best["state_m"], "scorer": best["state_s"], "surfaces": words}, ck)
    out = {
        "dig": "131_func_bias",
        "verdict": verdict,
        "story_all": best["story"],
        "story_lift_pp": lift,
        "seen_obj_cmp": best["obj"],
        "rel": best["rel"],
        "story_top5_func_frac": best["func_top5"],
        "func_drop_vs_129approx_pp": func_drop,
        "ckpt": str(ck),
    }
    (RES / f"stage131_func_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[131] {verdict} STORY={100*best['story']:.1f}% func_top5={100*best['func_top5']:.1f}%")
    return out


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"130/131 start {datetime.now(timezone.utc).isoformat()}")
    log("128 STOPPED by user request — membership hops@K abandoned")
    try:
        # mark 128 aborted
        abort = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage128": {"verdict": "ABORTED", "note": "stopped; membership hopK inert; moved to compare-layer"},
        }
        prev = RES / "stage127_128_decision.json"
        if prev.exists():
            try:
                d = json.loads(prev.read_text(encoding="utf-8"))
                d["stage128"] = abort["stage128"]
                d["aborted"] = True
                prev.write_text(json.dumps(d, indent=2), encoding="utf-8")
            except Exception:
                pass
        else:
            (RES / "stage128_ABORTED.txt").write_text(json.dumps(abort, indent=2), encoding="utf-8")

        phrases = ensure_100k()
        pack = dig_130(phrases)
        r130 = pack[0]
        r131 = dig_131_func_bias(pack[1], phrases, pack)

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage128": "ABORTED",
            "stage130": r130,
            "stage131": r131,
            "next_if_130_helps_seen_but_story_out5": (
                "cross-attn atom↔memory to propose candidates outside atom topK "
                "(addresses neither/bpe_only proposal hole; heavier arch dig)"
            ),
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**128 ABORTED** (membership hopK). "
                f"**130 compare-layer** {r130['verdict']} cmp_obj={100*r130['seen_obj_cmp']:.1f}% "
                f"(Δatom {100*r130['cmp_minus_atom_pp']:+.1f}pp). "
                f"**131 func-bias** {r131['verdict']} STORY={100*r131['story_all']:.1f}%. "
                f"`stage130_131_decision.json`. Cross-attn mem = later if proposal still out5.\n"
            )
            if "128 ABORTED" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")
        log("DONE 130/131")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
