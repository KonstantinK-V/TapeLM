"""
Post-EFG priority digs — sequential, descending chance:

  0) C′  toks/word by freq band (diag; gates G′ density reading)
  1) #5  same-role shortlist CE + pairwise hinge on right (highest chance)
  2) G′  morph-root positions as atom input (stem[+affix]+surface)

Skip (already closed / low chance): F, temp, role-reweight, hybrid conflict,
multi-head-by-role, more phrase-mem.

Gate vs REF100 (obj 42.6% / rel 95.8% / STORY 19.6%):
  #5 PASS if SEEN obj +3pp & rel>=0.70; PARTIAL +1.5pp
  G′ PASS if STORY +3pp & SEEN obj held within -5pp

Run:
  python _post_efg_priority_digs.py
"""
from __future__ import annotations

import json
import random
import sys
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
    _warmup_then_constant,
    build_rights_index,
    build_ts_repeat_mix,
    collate_word_id_batch,
    eval_id_capacity_suite,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
    surface,
    target_freq_bucket,
    AFFIXES,
)
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402
from _stage116_117_hops_morph_follow import crude_stem  # noqa: E402

ATOM100 = CKPT / "stage100_scale_100k.pt"
BPE_TOK = RES / "stage112_bpe_tokenizer.json"
LOG = RES / "_post_efg_priority_log.txt"
DEC = RES / "stage125_127_priority_decision.json"


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def setup_data(phrases):
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
    words = sorted({w for ln in train + hold_seen + hold_rare + hold_story for w in ln["words"]})
    stoi = {s: i for i, s in enumerate(words)}
    fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)
    return cfg, device, stack, train, hold_seen, hold_rare, hold_story, words, stoi, fps


def load_atom100(words, fps, device):
    model = WordIdTransformer(len(words), 256, 4, 2, 16, 0.1).to(device)
    if ATOM100.exists():
        ck = torch.load(ATOM100, map_location="cpu", weights_only=False)
        if list(ck.get("surfaces", [])) == words:
            model.load_state_dict(ck["word_tf"], strict=True)
            log("[init] Stage100 warm")
            return model
    model.init_from_fps(fps)
    log("[init] fp (no Stage100 match)")
    return model


# ---------- 0) C′ ----------
def dig_C_prime(phrases: list[str]) -> dict:
    log("\n======== 0) C′ BPE toks/word by freq band ========")
    if not BPE_TOK.exists():
        out = {"available": False}
        log("[C′] no BPE tokenizer")
        return out
    tok = Tokenizer.from_file(str(BPE_TOK))
    bos, eos = tok.token_to_id("[BOS]"), tok.token_to_id("[EOS]")
    # unigram from phrases
    uni = Counter()
    for p in phrases:
        for w in p.split():
            uni[w] += 1
    bands = defaultdict(lambda: {"n_words": 0, "n_toks": 0, "n_multi": 0})
    rng = random.Random(272)
    sample = phrases if len(phrases) <= 5000 else [phrases[i] for i in rng.sample(range(len(phrases)), 5000)]
    for p in sample:
        ws = p.split()
        if not ws:
            continue
        # whole-phrase encode
        ids = [i for i in tok.encode(p).ids if i not in (bos, eos)]
        # also per-word encode
        for w in ws:
            wids = [i for i in tok.encode(w).ids if i not in (bos, eos)]
            b = target_freq_bucket(uni.get(w, 0))
            bands[b]["n_words"] += 1
            bands[b]["n_toks"] += max(len(wids), 1)
            bands[b]["n_multi"] += int(len(wids) >= 2)
            bands["ALL"]["n_words"] += 1
            bands["ALL"]["n_toks"] += max(len(wids), 1)
            bands["ALL"]["n_multi"] += int(len(wids) >= 2)
        bands["phrase_level"]["n_words"] += len(ws)
        bands["phrase_level"]["n_toks"] += len(ids)
    packed = {}
    for k, v in bands.items():
        nw = max(v["n_words"], 1)
        packed[k] = {
            "n_words": v["n_words"],
            "mean_toks_per_word": v["n_toks"] / nw,
            "frac_multi_piece": v["n_multi"] / nw if k != "phrase_level" else None,
        }
    mean_all = packed.get("ALL", {}).get("mean_toks_per_word", 1.0)
    head = packed.get("head21p", {}).get("mean_toks_per_word", 1.0)
    rare = packed.get("rare1", {}).get("mean_toks_per_word", 1.0)
    # LIVE density if overall >=1.5 OR rare/mid clearly multi
    signal = "LIVE" if mean_all >= 1.5 or rare >= 2.0 else ("WEAK" if mean_all >= 1.15 or rare >= 1.4 else "NULL")
    out = {
        "dig": "C_prime",
        "signal": signal,
        "bands": packed,
        "note": "per-word BPE encode; phrase_level = whole-string ids / n_words",
        "head_tpw": head,
        "rare_tpw": rare,
        "all_tpw": mean_all,
    }
    log(f"[C′] signal={signal} ALL={mean_all:.3f} head={head:.3f} rare={rare:.3f}")
    for k in ("ALL", "head21p", "mid6_20", "low2_5", "rare1", "phrase_level"):
        if k in packed:
            log(f"  {k}: {json.dumps(packed[k])}")
    (RES / "stage125_Cprime_toks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ---------- 1) #5 shortlist + hinge ----------
def dig_5_same_role_rank(phrases: list[str]) -> dict:
    log("\n======== 1) #5 same-role shortlist CE + pairwise hinge ========")
    cfg, device, stack, train, hold_seen, hold_rare, hold_story, words, stoi, fps = setup_data(phrases)
    model = load_atom100(words, fps, device)
    by_rel, all_rights = build_rights_index(train)
    all_right_ids = [stoi[w] for w in sorted(all_rights) if w in stoi]
    by_rel_ids = {
        r: [stoi[w] for w in sorted(s) if w in stoi] for r, s in by_rel.items()
    }

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p = lines_to_pairs(fat, stoi)
    story_p = lines_to_pairs(_subsample(story, 50000, 5), stoi)
    # right-only pools
    def rights_only(pairs, lines):
        out = []
        for ex in pairs:
            ln = lines[ex["line_i"]] if ex["line_i"] < len(lines) else None
            # pairs from fat/story separately — store words on ex via rebuild
            out.append(ex)
        return out

    # annotate pairs with role using embedded words from phrase field
    def role_of(ex):
        ws = ex["phrase"].split()
        return _role(ex, ws)

    right_fat = [ex for ex in fat_p if role_of(ex) == "right"]
    right_story = [ex for ex in story_p if role_of(ex) == "right"]
    other_fat = [ex for ex in fat_p if role_of(ex) != "right"]
    other_story = [ex for ex in story_p if role_of(ex) != "right"]
    log(f"[#5] right_fat={len(right_fat)} right_story={len(right_story)} all_rights={len(all_right_ids)}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rr = random.Random(7)
    hinge_m = 0.5
    ev_seen = _subsample(hold_seen, 600, 1)
    ev_story = _subsample(hold_story, 400, 2)
    best = {"obj": -1.0, "rel": 0.0, "story": 0.0, "hit5_obj": 0.0, "state": None}

    def shortlist_for(ex):
        ws = ex["phrase"].split()
        pl = int(ex["prefix_len"])
        rel = ws[pl - 1] if pl >= 1 else ""
        gold = ex["target_word"]
        cands = list(by_rel_ids.get(rel, [])) or list(all_right_ids)
        if stoi[gold] not in cands:
            cands = cands + [stoi[gold]]
        # cap shortlist size
        if len(cands) > 64:
            # keep gold + sample
            others = [c for c in cands if c != stoi[gold]]
            rr.shuffle(others)
            cands = [stoi[gold]] + others[:63]
        return cands

    @torch.no_grad()
    def eval_obj_hit5():
        # quick obj hit1/hit5 on seen
        n = h1 = h5 = 0
        for ln in ev_seen:
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
                gid = stoi[gold]
                top5 = set(int(i) for i in logits.topk(min(5, logits.numel())).indices.tolist())
                n += 1
                h1 += int(int(logits.argmax()) == gid)
                h5 += int(gid in top5)
        return h1 / max(n, 1), h5 / max(n, 1)

    for step in range(1, 40001):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, 1e-3, 200)
        # batch mix: 4 right shortlist + 2 hinge + 2 full-V other
        losses = []
        # --- shortlist CE on rights ---
        for _ in range(4):
            pool = right_fat if rr.random() < 0.75 else right_story
            if not pool:
                continue
            ex = rr.choice(pool)
            cands = shortlist_for(ex)
            packed = collate_word_id_batch([ex], stoi, model.max_len, model.pad_id, device)
            if packed is None:
                continue
            ids, mask, _tgt = packed
            logits = model.logits_last_from_batch(ids, mask)[0]
            gold_i = stoi[ex["target_word"]]
            # map to shortlist indices
            sl = torch.tensor(cands, device=device, dtype=torch.long)
            sl_logits = logits.index_select(0, sl)
            target = torch.tensor([cands.index(gold_i)], device=device)
            losses.append(F.cross_entropy(sl_logits.unsqueeze(0), target))

        # --- pairwise hinge: gold vs hard neg from same shortlist ---
        for _ in range(2):
            pool = right_fat if rr.random() < 0.75 else right_story
            if not pool:
                continue
            ex = rr.choice(pool)
            cands = shortlist_for(ex)
            gold_i = stoi[ex["target_word"]]
            negs = [c for c in cands if c != gold_i]
            if not negs:
                continue
            packed = collate_word_id_batch([ex], stoi, model.max_len, model.pad_id, device)
            if packed is None:
                continue
            ids, mask, _ = packed
            logits = model.logits_last_from_batch(ids, mask)[0]
            # hardest neg in shortlist
            neg_scores = logits[torch.tensor(negs, device=device)]
            hard = negs[int(neg_scores.argmax())]
            # hinge: logit_gold >= logit_neg + m
            losses.append(F.relu(hinge_m - (logits[gold_i] - logits[hard])))

        # --- standard CE on non-right to hold rel ---
        for _ in range(2):
            pool = other_fat if rr.random() < 0.75 else other_story
            if not pool:
                continue
            ex = rr.choice(pool)
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

        if step % 2000 == 0 or step == 40000:
            model.eval()
            seen = eval_id_capacity_suite(model, ev_seen, train, words, stoi, device)
            story_e = eval_id_capacity_suite(model, ev_story, train, words, stoi, device)
            obj = seen["obj"]["hit1"]
            rel = seen["roles"].get("rel", {}).get("hit1", 0.0)
            st = story_e["roles"].get("ALL", {}).get("hit1", 0.0)
            h1, h5 = eval_obj_hit5()
            log(
                f"  [#5] step {step}: obj={obj*100:.1f}% rel={rel*100:.1f}% "
                f"STORY={st*100:.1f}% | obj hit1={h1*100:.1f}% hit5={h5*100:.1f}% "
                f"(gap { (h5-h1)*100:.1f}pp)"
            )
            if obj > best["obj"] and rel >= 0.70:
                best.update(
                    obj=obj, rel=rel, story=st, hit5_obj=h5,
                    state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                )
            model.train()

    lift = best["obj"] - REF100["obj"]
    if best["rel"] >= 0.70 and lift >= 0.03:
        verdict = "PASS"
    elif best["rel"] >= 0.70 and lift >= 0.015:
        verdict = "PARTIAL"
    else:
        verdict = "PARITY"
    ck = CKPT / "stage125_same_role_rank.pt"
    if best["state"] is not None:
        torch.save({"word_tf": best["state"], "surfaces": words}, ck)
    out = {
        "dig": "5_same_role_rank",
        "verdict": verdict,
        "seen": {"obj": best["obj"], "rel": best["rel"], "obj_hit5": best["hit5_obj"]},
        "story_all": best["story"],
        "obj_lift_pp": lift,
        "near_miss_gap_pp": best["hit5_obj"] - best["obj"] if best["obj"] >= 0 else None,
        "ckpt": str(ck),
    }
    (RES / f"stage125_same_role_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[#5] {verdict} obj={100*best['obj']:.1f}% lift={100*lift:+.1f}pp STORY={100*best['story']:.1f}%")
    return out


# ---------- 2) G′ morph-root positions ----------
def dig_G_prime_morph(phrases: list[str], c_prime: dict) -> dict:
    log("\n======== 2) G′ morph-root positions (stem[+affix]+surface) ========")
    log(f"[G′] C′ signal was {c_prime.get('signal')} — density thesis "
        f"{'SUPPORTED' if c_prime.get('signal')=='LIVE' else 'WEAK/NULL (still try stem-sharing)'}")
    cfg, device, stack, train, hold_seen, hold_rare, hold_story, words, stoi_w, fps_w = setup_data(phrases)

    # piece vocab
    stem_set = set()
    affix_toks = [f"+{a}" for a in AFFIXES]  # +ing +ed +s
    for w in words:
        st = crude_stem(w)
        if st and st != w:
            stem_set.add(f"${st}")  # $stem marker
        else:
            # still add $w as stem identity for sharing across rare morph if any
            stem_set.add(f"${w}")
    stems = sorted(stem_set)[:8000]
    surfaces = words + stems + affix_toks
    stoi = {s: i for i, s in enumerate(surfaces)}

    fps_list = []
    for s in surfaces:
        if s.startswith("$"):
            body = s[1:]
            try:
                fps_list.append(stack.w(body).detach())
            except Exception:
                fps_list.append(torch.zeros(cfg.dim, device=device))
        elif s.startswith("+"):
            # affix: small random-ish from mean of surfaces using that affix
            fps_list.append(torch.randn(cfg.dim, device=device) * 0.01)
        else:
            fps_list.append(stack.w(s).detach())
    fps = F.normalize(torch.stack(fps_list, 0), dim=-1).to(device)

    def expand_prefix(ws):
        ids = []
        for w in ws:
            st = crude_stem(w)
            stem_tok = f"${st}"
            if stem_tok in stoi:
                ids.append(stoi[stem_tok])
            # affix if surface differs
            if st != w:
                for a in AFFIXES:
                    try:
                        if surface(st, a) == w:
                            at = f"+{a}"
                            if at in stoi:
                                ids.append(stoi[at])
                            break
                    except Exception:
                        pass
            if w in stoi:
                ids.append(stoi[w])
        return ids

    model = WordIdTransformer(len(surfaces), 256, 4, 2, max_len=32, dropout=0.1).to(device)
    model.init_from_fps(fps)
    # warm surface rows from stage100 if possible
    if ATOM100.exists():
        ck = torch.load(ATOM100, map_location="cpu", weights_only=False)
        old = list(ck.get("surfaces", []))
        if old:
            with torch.no_grad():
                sd = ck["word_tf"]
                old_emb = sd["tok.weight"]
                for i, w in enumerate(old):
                    if w in stoi and i < old_emb.size(0):
                        model.tok.weight[stoi[w]].copy_(old_emb[i].to(device))
            log("[G′] warmed surface emb from Stage100")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    rr = random.Random(9)

    def sample_ex():
        ln = rr.choice(fat) if rr.random() < 0.75 else rr.choice(story)
        ws = ln["words"]
        if len(ws) < 2:
            return None
        t = rr.randrange(1, len(ws))
        gold = ws[t]
        if gold not in stoi:
            return None
        pref = expand_prefix(ws[:t])[-model.max_len :]
        if not pref:
            return None
        return pref, stoi[gold]

    @torch.no_grad()
    def eval_hold(hold_lines):
        n = h = 0
        obj_n = obj_h = 0
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
                ok = int(pred == gold)
                n += 1
                h += ok
                if ws[t - 1] in RELS:
                    obj_n += 1
                    obj_h += ok
        return h / max(n, 1), obj_h / max(obj_n, 1)

    ev_story = _subsample(hold_story, 400, 2)
    ev_seen = _subsample(hold_seen, 400, 1)
    best = {"story": -1.0, "seen_all": 0.0, "seen_obj": 0.0, "state": None}

    for step in range(1, 40001):
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
        # targets are surface ids only — stem/affix never predicted
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 2000 == 0 or step == 40000:
            model.eval()
            st, _ = eval_hold(ev_story)
            sn, so = eval_hold(ev_seen)
            log(f"  [G′] step {step}: STORY={st*100:.1f}% SEEN_ALL={sn*100:.1f}% SEEN_obj={so*100:.1f}%")
            if st >= best["story"] and so >= REF100["obj"] - 0.08:
                best.update(
                    story=st, seen_all=sn, seen_obj=so,
                    state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                )
            model.train()

    lift = best["story"] - REF100["story_all"]
    obj_ok = best["seen_obj"] >= REF100["obj"] - 0.05
    if lift >= 0.03 and obj_ok:
        verdict = "PASS"
    elif lift >= 0.015 and obj_ok:
        verdict = "PARTIAL"
    else:
        verdict = "PARITY"
    ck = CKPT / "stage126_Gprime_morph_roots.pt"
    if best["state"] is not None:
        torch.save({"word_tf": best["state"], "surfaces": surfaces}, ck)
    out = {
        "dig": "G_prime_morph_roots",
        "verdict": verdict,
        "story_all": best["story"],
        "seen_all": best["seen_all"],
        "seen_obj": best["seen_obj"],
        "story_lift_pp": lift,
        "c_prime_signal": c_prime.get("signal"),
        "ckpt": str(ck),
        "expand": "[$stem, +affix?, surface] per word; predict surface only",
    }
    (RES / f"stage126_Gprime_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[G′] {verdict} STORY={100*best['story']:.1f}% lift={100*lift:+.1f}pp obj={100*best['seen_obj']:.1f}%")
    return out


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"Priority digs start {datetime.now(timezone.utc).isoformat()}")
    log("Order by chance: C′(diag) → #5 same-role rank → G′ morph-roots")
    try:
        phrases = ensure_100k()
        c_prime = dig_C_prime(phrases)
        r5 = dig_5_same_role_rank(phrases)
        rG = dig_G_prime_morph(phrases, c_prime)
        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "order": ["C_prime", "5_same_role_rank", "G_prime_morph_roots"],
            "C_prime": c_prime,
            "dig5": r5,
            "G_prime": rG,
            "next_if_fail": "stop densify/rank soft ideas; revisit data/curriculum not temp/hybrid/reweight",
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")

        plan = RES / "PRIORITY_DIGS_PLAN.txt"
        plan.write_text(
            "\n".join([
                "SOTE priority digs (descending chance)",
                f"timestamp: {out['timestamp']}",
                "",
                f"0 C′  signal={c_prime.get('signal')} ALL_tpw={c_prime.get('all_tpw')} "
                f"head={c_prime.get('head_tpw')} rare={c_prime.get('rare_tpw')}",
                f"1 #5  {r5['verdict']} obj={100*r5['seen']['obj']:.1f}% "
                f"lift={100*r5['obj_lift_pp']:+.1f}pp STORY={100*r5['story_all']:.1f}%",
                f"2 G′  {rG['verdict']} STORY={100*rG['story_all']:.1f}% "
                f"lift={100*rG['story_lift_pp']:+.1f}pp",
                "",
                "Skipped: F(done PARITY/-10pp), temp, role-reweight, hybrid, multi-head, more mem",
                f"Decision JSON: {DEC}",
            ]) + "\n",
            encoding="utf-8",
        )

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Priority digs 125–126:** C′ {c_prime.get('signal')}; "
                f"#5 {r5['verdict']} obj_lift={100*r5['obj_lift_pp']:+.1f}pp; "
                f"G′ {rG['verdict']} STORY_lift={100*rG['story_lift_pp']:+.1f}pp. "
                f"`stage125_127_priority_decision.json`.\n"
            )
            if "Priority digs 125" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")
        log("DONE priority digs")
        log(plan.read_text(encoding="utf-8"))
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
