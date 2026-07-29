"""
Stage 116–117 after Stage115:

 dual-channel digs that use hops as path/object prior (NOT next-word teacher),
 plus morph-tolerant diagnostics beside exact@1 (never as gate).

  116 morph_diag (eval-only on Stage100 atom):
      report exact@1 AND morph_tolerant@1 / stem_hit on SEEN + STORY.
      Gate remains exact; morph is diagnostic for false misses.

  117a hops_rerank: FT atom with recipe98; on role==right only:
      score = logit + α * hop_prior(candidate)
      hop_prior from SoftPhraseMemory over fat/path phrases (right words of retrieved facts).
      Non-right slots: plain CE (hops silent).

  117b hops_loss: FT atom; on role==right:
      loss = CE + λ * (1 - P_mass_on_hop_candidates)
      else plain CE.

Run (waits for stage115_context_decision.json):
  python _stage116_117_hops_morph_follow.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    AFFIXES,
    CKPT,
    FACT_BANK_EXP_F,
    RELS,
    RES,
    Config,
    SoftPhraseMemory,
    WordIdTransformer,
    _role,
    _warmup_then_constant,
    build_ts_repeat_mix,
    collate_word_id_batch,
    eval_id_capacity_suite,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
    surface,
)
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402

DEC_PRIOR = RES / "stage115_context_decision.json"
LOG = RES / "_stage116_117_log.txt"
DEC = RES / "stage116_117_decision.json"
ATOM100 = CKPT / "stage100_scale_100k.pt"


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def wait_prior(timeout_s: int = 16 * 3600, poll_s: int = 45) -> dict:
    log(f"[wait] for {DEC_PRIOR} ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC_PRIOR.exists():
            d = json.loads(DEC_PRIOR.read_text(encoding="utf-8"))
            if d.get("stage115a") is not None:
                log("[wait] 115 done")
                return d
        time.sleep(poll_s)
        if int(time.time() - t0) % 300 < poll_s:
            log(f"[wait] still ... {int(time.time()-t0)}s")
    raise TimeoutError("115 not ready")


# ---------- morph diagnostics ----------
def crude_stem(w: str) -> str:
    w = w.lower()
    for suf in ("ing", "ies", "ied", "ers", "est", "ely", "ed", "es", "ly", "er", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            stem = w[: -len(suf)]
            if suf == "ing" and len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                stem = stem[:-1]  # running -> runn -> run
            if suf in ("ies", "ied") and stem:
                stem = stem + "y"
            return stem
    return w


def morph_family(w: str) -> set[str]:
    """Gold surface + stem + common affix expansions (SOTE AFFIXES)."""
    w = w.lower()
    fam = {w, crude_stem(w)}
    st = crude_stem(w)
    for aff in AFFIXES:
        try:
            fam.add(surface(st, aff))
        except Exception:
            pass
        # also try surface from raw w if different
        try:
            fam.add(surface(w, aff))
        except Exception:
            pass
    return {x for x in fam if x}


def morph_hit(pred: str, gold: str) -> bool:
    if pred == gold:
        return True
    return bool(morph_family(pred) & morph_family(gold))


@torch.no_grad()
def eval_exact_and_morph(model, hold_lines, train_lines, surfaces, stoi, device):
    """Exact@1 (gate) + morph_tolerant@1 + stem_hit (diag only)."""
    from collections import defaultdict

    pairs = lines_to_pairs(hold_lines, stoi)
    buckets = defaultdict(lambda: {"n": 0, "exact": 0, "morph": 0, "stem": 0})
    obj = {"n": 0, "exact": 0, "morph": 0, "stem": 0}
    for ex in pairs:
        if int(ex["prefix_len"]) < 1:
            continue
        line = hold_lines[ex["line_i"]]
        ws = line["words"]
        gold = ex["target_word"]
        if gold not in stoi:
            continue
        role = _role(ex, ws)
        ids = list(ex["prefix_word_ids"][-model.max_len :])
        pred = surfaces[model.pred_id(ids)]
        ex_ok = int(pred == gold)
        mo_ok = int(morph_hit(pred, gold))
        st_ok = int(crude_stem(pred) == crude_stem(gold))
        for name in (role, "ALL"):
            buckets[name]["n"] += 1
            buckets[name]["exact"] += ex_ok
            buckets[name]["morph"] += mo_ok
            buckets[name]["stem"] += st_ok
        if role == "right":
            obj["n"] += 1
            obj["exact"] += ex_ok
            obj["morph"] += mo_ok
            obj["stem"] += st_ok

    def pack(d):
        out = {}
        for k, v in d.items():
            n = max(v["n"], 1)
            out[k] = {
                "n": v["n"],
                "exact": v["exact"] / n,
                "morph": v["morph"] / n,
                "stem": v["stem"] / n,
                "morph_minus_exact_pp": (v["morph"] - v["exact"]) / n,
            }
        return out

    return {
        "roles": pack(buckets),
        "obj": {
            "n": obj["n"],
            "exact": obj["exact"] / max(obj["n"], 1),
            "morph": obj["morph"] / max(obj["n"], 1),
            "stem": obj["stem"] / max(obj["n"], 1),
            "morph_minus_exact_pp": (obj["morph"] - obj["exact"]) / max(obj["n"], 1),
        },
    }


# ---------- hop prior from path memory ----------
def build_path_memory(stack, cfg, train_lines, device, fact_path: Path | None = None):
    """SoftPhraseMemory over path-like phrases + right-index by (left,rel) and by rel."""
    mem = SoftPhraseMemory(cfg.dim, cfg, device)
    by_left_rel = defaultdict(set)
    by_rel = defaultdict(set)
    phrases = []

    def add_triple(left, rel, right, phrase=None):
        if rel not in RELS:
            return
        ph = phrase or f"{left} {rel} {right}"
        phrases.append({"phrase": ph, "left": left, "rel": rel, "right": right, "words": ph.split()})
        by_left_rel[(left, rel)].add(right)
        by_rel[rel].add(right)

    for ln in train_lines:
        ws = ln["words"]
        if ln.get("bucket") in ("fat_train", "fat_hold") and ln.get("triple"):
            t = ln["triple"]
            add_triple(t[0], t[1], t[2], ln.get("phrase"))
            continue
        for i, w in enumerate(ws):
            if w in RELS and i > 0 and i + 1 < len(ws):
                add_triple(ws[i - 1], w, ws[i + 1])

    if fact_path and fact_path.exists():
        for raw in fact_path.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 3 and parts[1] in RELS:
                add_triple(parts[0], parts[1], parts[2], " ".join(parts[:3]))

    for ph in phrases[:5000]:  # cap observe cost
        mem.observe_strict(
            ph["phrase"],
            stack.phrase_fp(ph["words"]),
            fact=ph,
            left_fp=stack.w(ph["left"]),
        )
    log(f"[hopmem] phrases~{len(phrases)} left_rel_keys={len(by_left_rel)} observed={len(mem.slots)}")
    return mem, by_left_rel, by_rel


def hop_candidates_for_prefix(ws_prefix, by_left_rel, by_rel, mem, stack, stoi, topk=8):
    """Return set of candidate right-word ids for object slot; empty if unknown."""
    if not ws_prefix or ws_prefix[-1] not in RELS:
        return set()
    rel = ws_prefix[-1]
    cands = set()
    if len(ws_prefix) >= 2:
        left = ws_prefix[-2]
        cands |= by_left_rel.get((left, rel), set())
    if len(cands) < 2:
        cands |= by_rel.get(rel, set())
    # retrieve phrase memory boost
    try:
        cue = " ".join(ws_prefix[-3:])
        fp = stack.phrase_fp(cue.split())
        for name, _sim in mem.topk(fp, k=topk):
            meta = mem.fact_meta.get(name) or {}
            r = meta.get("right")
            if r:
                cands.add(r)
    except Exception:
        pass
    return {stoi[w] for w in cands if w in stoi}


def annotate_roles(pairs, lines):
    for ex in pairs:
        ex["role"] = _role(ex, lines[ex["line_i"]]["words"])
        ex["words"] = lines[ex["line_i"]]["words"]
    return pairs


def train_hops_prior(
    *,
    stage: int,
    tag: str,
    phrases: list[str],
    mode: str,  # "rerank" | "loss"
    alpha: float = 2.0,
    lam: float = 0.3,
    ft_steps: int = 40000,
    eval_every: int = 1000,
    fat_frac: float = 0.75,
    ref: dict | None = None,
) -> dict:
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

    out_ckpt = CKPT / f"stage{stage}_{tag}.pt"
    out_txt = RES / f"stage{stage}_{tag}_report.txt"
    out_json = RES / f"stage{stage}_{tag}_metrics.json"

    log(f"\n======== Stage {stage} {tag} mode={mode} α={alpha} λ={lam} ========")
    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=272)
    all_lines = train + hold_seen + hold_rare + hold_story
    for ln in all_lines:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")

    all_words = sorted({w for ln in all_lines for w in ln["words"]})
    word_fps = F.normalize(
        torch.stack([stack.w(w).detach() for w in all_words], 0), dim=-1
    ).to(device)
    surf = all_words
    stoi = {s: i for i, s in enumerate(surf)}

    mem, by_lr, by_rel = build_path_memory(stack, cfg, train, device, FACT_BANK_EXP_F)

    fat_lines = [ln for ln in train if ln.get("bucket") == "fat_train"]
    story_lines = [ln for ln in train if ln.get("bucket") != "fat_train"]
    story_src = story_lines if len(story_lines) <= 60000 else _subsample(story_lines, 60000, 12)
    fat_pairs = annotate_roles(lines_to_pairs(fat_lines, stoi), fat_lines)
    story_pairs = annotate_roles(lines_to_pairs(story_src, stoi), story_src)
    if not fat_pairs:
        fat_pairs = story_pairs

    ev_seen = _subsample(hold_seen, 600, 1001)
    ev_story = _subsample(hold_story, 400, 1003)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 1200), 2001)
    fin_story = _subsample(hold_story, min(len(hold_story), 800), 2003)

    model = WordIdTransformer(
        n_vocab=len(surf), d_model=256, n_heads=4, n_layers=2, max_len=16, dropout=0.1
    ).to(device)
    # warm start from Stage100 if vocab compatible-ish: else fp init
    if ATOM100.exists():
        try:
            ck = torch.load(ATOM100, map_location="cpu", weights_only=False)
            if list(ck.get("surfaces", [])) == surf:
                model.load_state_dict(ck["word_tf"], strict=True)
                log(f"[init] loaded {ATOM100.name}")
            else:
                model.init_from_fps(word_fps)
                log("[init] vocab mismatch; fp init")
        except Exception as e:
            model.init_from_fps(word_fps)
            log(f"[init] fp init ({e})")
    else:
        model.init_from_fps(word_fps)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = random.Random(272)
    batch = 8

    def _sample_batch():
        n_fat = max(1, int(round(batch * fat_frac)))
        n_fat = min(n_fat, batch)
        ex = [rng.choice(fat_pairs) for _ in range(n_fat)]
        ex += [rng.choice(story_pairs) for _ in range(batch - n_fat)]
        rng.shuffle(ex)
        return ex

    def forward_logits_with_prior(ids, pad_mask, exs_valid):
        """ids from collate; exs_valid aligned list of examples that survived collate."""
        logits = model.logits_last_from_batch(ids, pad_mask)
        if mode != "rerank":
            return logits
        # add hop bonus on right-role rows
        for i, ex in enumerate(exs_valid):
            if ex.get("role") != "right":
                continue
            pref = ex["words"][: ex["prefix_len"]]
            cands = hop_candidates_for_prefix(pref, by_lr, by_rel, mem, stack, stoi)
            if not cands:
                continue
            for cid in cands:
                logits[i, cid] = logits[i, cid] + alpha
        return logits

    def loss_with_prior(logits, tgt, exs_valid):
        ce = F.cross_entropy(logits, tgt, reduction="none")
        if mode != "loss":
            return ce.mean()
        extra = []
        for i, ex in enumerate(exs_valid):
            if ex.get("role") != "right":
                extra.append(torch.zeros((), device=logits.device))
                continue
            pref = ex["words"][: ex["prefix_len"]]
            cands = hop_candidates_for_prefix(pref, by_lr, by_rel, mem, stack, stoi)
            if len(cands) < 2:
                extra.append(torch.zeros((), device=logits.device))
                continue
            # encourage probability mass on hop set
            logp = F.log_softmax(logits[i], dim=-1)
            idx = torch.tensor(sorted(cands), device=logits.device, dtype=torch.long)
            mass = torch.logsumexp(logp[idx], dim=0)
            extra.append(-mass)  # maximize mass
        extra_t = torch.stack(extra)
        return ce.mean() + lam * extra_t.mean()

    best = {
        "step": 0, "obj": 0.0, "rel": 0.0, "story_all": 0.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []

    def snap(step):
        model.eval()
        # eval with same prior if rerank
        def _pred_id_prior(id_list, words_prefix):
            ids = torch.tensor([id_list[-model.max_len :]], dtype=torch.long, device=device)
            logits = model.forward(ids)[0, -1]
            if mode == "rerank" and words_prefix and words_prefix[-1] in RELS:
                cands = hop_candidates_for_prefix(words_prefix, by_lr, by_rel, mem, stack, stoi)
                for cid in cands:
                    logits[cid] = logits[cid] + alpha
            return int(logits.argmax())

        # use standard suite for roles; obj from morph eval exact
        seen = eval_id_capacity_suite(model, ev_seen, train, surf, stoi, device)
        story = eval_id_capacity_suite(model, ev_story, train, surf, stoi, device)
        # morph diag on seen
        morph_seen = eval_exact_and_morph(model, ev_seen, train, surf, stoi, device)
        obj_s = seen["obj"]["hit1"]
        rel_s = seen["roles"].get("rel", {}).get("hit1", 0.0)
        st_all = story["roles"].get("ALL", {}).get("hit1", 0.0)
        # optional: rerank obj estimate
        if mode == "rerank":
            # recompute object with prior on ev_seen right pairs
            pairs = lines_to_pairs(ev_seen, stoi)
            n = h = 0
            for ex in pairs:
                ws = ev_seen[ex["line_i"]]["words"]
                if _role(ex, ws) != "right":
                    continue
                pref = ws[: ex["prefix_len"]]
                pred = surf[_pred_id_prior(ex["prefix_word_ids"], pref)]
                n += 1
                h += int(pred == ex["target_word"])
            obj_prior = h / max(n, 1)
        else:
            obj_prior = obj_s
        curve.append({
            "step": step, "obj": obj_s, "obj_prior": obj_prior, "rel": rel_s,
            "story_all": st_all, "obj_morph": morph_seen["obj"]["morph"],
        })
        key = (1 if rel_s >= 0.70 else 0, obj_prior, obj_s, st_all)
        bkey = (1 if best["rel"] >= 0.70 else 0, best.get("obj_prior", best["obj"]), best["obj"], best["story_all"])
        best["obj_prior"] = best.get("obj_prior", best["obj"])
        if key >= bkey:
            best.update(
                step=step, obj=obj_s, obj_prior=obj_prior, rel=rel_s, story_all=st_all,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        log(
            f"  step {step:5d}: SEEN obj={obj_s*100:.1f}% obj+prior={obj_prior*100:.1f}% "
            f"rel={rel_s*100:.1f}% | STORY ALL={st_all*100:.1f}% | "
            f"obj_morph={morph_seen['obj']['morph']*100:.1f}%"
        )
        model.train()

    log("=== FT hops-as-prior ===")
    snap(0)
    model.train()
    for step in range(1, ft_steps + 1):
        lr = _warmup_then_constant(step, 1e-3, 200)
        for g in opt.param_groups:
            g["lr"] = lr
        exs = _sample_batch()
        # filter like collate
        exs_valid = []
        for ex in exs:
            if ex["target_word"] not in stoi:
                continue
            if not ex["prefix_word_ids"]:
                continue
            exs_valid.append(ex)
        packed = collate_word_id_batch(exs_valid, stoi, model.max_len, model.pad_id, device)
        if packed is None:
            continue
        ids, pad_mask, tgt = packed
        # realign exs_valid length to tgt (collate may drop)
        if len(exs_valid) != tgt.shape[0]:
            exs_valid = exs_valid[: tgt.shape[0]]
        logits = forward_logits_with_prior(ids, pad_mask, exs_valid)
        loss = loss_with_prior(logits, tgt, exs_valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == ft_steps:
            snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    f_seen = eval_exact_and_morph(model, fin_seen, train, surf, stoi, device)
    f_story = eval_exact_and_morph(model, fin_story, train, surf, stoi, device)
    plain = eval_id_capacity_suite(model, fin_seen, train, surf, stoi, device)
    plain_st = eval_id_capacity_suite(model, fin_story, train, surf, stoi, device)
    obj_s = plain["obj"]["hit1"]
    rel_s = plain["roles"].get("rel", {}).get("hit1", 0.0)
    st_all = plain_st["roles"].get("ALL", {}).get("hit1", 0.0)
    ref = ref or REF100
    obj_lift = obj_s - ref["obj"]

    if rel_s >= 0.70 and obj_lift >= 0.03:
        verdict = "PASS_OBJ"
    elif rel_s >= 0.70 and obj_lift >= 0.015:
        verdict = "PARTIAL"
    elif rel_s < 0.70:
        verdict = "FAIL_REL"
    else:
        verdict = "PARITY"

    report = "\n".join([
        f"SOTE Stage {stage} — {tag} (hops as path/object prior)",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"mode={mode} alpha={alpha} lam={lam}",
        f"SEEN exact obj={obj_s*100:.1f}% rel={rel_s*100:.1f}%",
        f"SEEN morph obj={f_seen['obj']['morph']*100:.1f}% "
        f"(+{f_seen['obj']['morph_minus_exact_pp']*100:.1f}pp vs exact)",
        f"STORY exact ALL={st_all*100:.1f}% morph ALL={f_story['roles'].get('ALL',{}).get('morph',0)*100:.1f}%",
        f"vs100 obj {obj_lift*100:+.1f}pp",
        f"Verdict: {verdict} (GATE=exact only; morph is diagnostic)",
    ]) + "\n"
    log("\n" + report)
    out_txt.write_text(report, encoding="utf-8")
    (RES / f"stage{stage}_{tag}_{verdict}.txt").write_text(report, encoding="utf-8")
    result = {
        "stage": stage, "tag": tag, "verdict": verdict, "mode": mode,
        "alpha": alpha, "lam": lam,
        "seen": {"obj_exact": obj_s, "rel": rel_s, "obj_morph": f_seen["obj"]["morph"],
                 "morph_gap_pp": f_seen["obj"]["morph_minus_exact_pp"]},
        "story": {
            "all_exact": st_all,
            "all_morph": f_story["roles"].get("ALL", {}).get("morph", 0.0),
            "morph_gap_pp": f_story["roles"].get("ALL", {}).get("morph_minus_exact_pp", 0.0),
        },
        "obj_lift_pp": obj_lift, "curve": curve, "ckpt": str(out_ckpt),
        "best_step": best["step"],
    }
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({
        "stage": stage, "tag": tag, "verdict": verdict, "mode": mode,
        "word_tf": best["state"], "surfaces": surf, "cfg": asdict(cfg),
    }, out_ckpt)
    log(f"Saved {out_ckpt}")
    return result


def run_morph_diag(phrases) -> dict:
    """Stage116: eval-only morph diagnostics on Stage100 weights if possible."""
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

    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=272)
    all_lines = train + hold_seen + hold_rare + hold_story
    for ln in all_lines:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")
    all_words = sorted({w for ln in all_lines for w in ln["words"]})
    word_fps = F.normalize(
        torch.stack([stack.w(w).detach() for w in all_words], 0), dim=-1
    ).to(device)
    surf = all_words
    stoi = {s: i for i, s in enumerate(surf)}

    model = WordIdTransformer(
        n_vocab=len(surf), d_model=256, n_heads=4, n_layers=2, max_len=16, dropout=0.1
    ).to(device)
    if ATOM100.exists():
        ck = torch.load(ATOM100, map_location="cpu", weights_only=False)
        if list(ck.get("surfaces", [])) == surf:
            model.load_state_dict(ck["word_tf"], strict=True)
            src = ATOM100.name
        else:
            model.init_from_fps(word_fps)
            # train briefly? no — load state mismatched; just fp and note
            src = "fp_init_vocab_mismatch"
            log("[116] Stage100 vocab mismatch; morph diag on freshly fp-inited (weak). Prefer retrain path.")
    else:
        model.init_from_fps(word_fps)
        src = "fp_init"
    model.eval()

    fin_seen = _subsample(hold_seen, min(len(hold_seen), 1200), 2001)
    fin_story = _subsample(hold_story, min(len(hold_story), 800), 2003)
    seen = eval_exact_and_morph(model, fin_seen, train, surf, stoi, device)
    story = eval_exact_and_morph(model, fin_story, train, surf, stoi, device)

    report = "\n".join([
        "SOTE Stage 116 — morph-tolerant diagnostic (exact remains GATE)",
        f"timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"weights: {src}",
        f"SEEN ALL exact={seen['roles'].get('ALL',{}).get('exact',0)*100:.1f}% "
        f"morph={seen['roles'].get('ALL',{}).get('morph',0)*100:.1f}% "
        f"stem={seen['roles'].get('ALL',{}).get('stem',0)*100:.1f}%",
        f"SEEN obj exact={seen['obj']['exact']*100:.1f}% morph={seen['obj']['morph']*100:.1f}% "
        f"(gap +{seen['obj']['morph_minus_exact_pp']*100:.1f}pp)",
        f"STORY ALL exact={story['roles'].get('ALL',{}).get('exact',0)*100:.1f}% "
        f"morph={story['roles'].get('ALL',{}).get('morph',0)*100:.1f}% "
        f"(gap +{story['roles'].get('ALL',{}).get('morph_minus_exact_pp',0)*100:.1f}pp)",
        "Verdict: DIAG (no gate change)",
    ]) + "\n"
    log("\n" + report)
    (RES / "stage116_morph_diag_report.txt").write_text(report, encoding="utf-8")
    (RES / "stage116_morph_diag_DIAG.txt").write_text(report, encoding="utf-8")
    out = {"stage": 116, "tag": "morph_diag", "verdict": "DIAG", "weights": src,
           "seen": seen, "story": story}
    (RES / "stage116_morph_diag_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"Stage116/117 start {datetime.now(timezone.utc).isoformat()}")
    try:
        prior = wait_prior()
        phrases = ensure_100k()

        r116 = run_morph_diag(phrases)
        r117a = train_hops_prior(
            stage=117, tag="hops_rerank", phrases=phrases,
            mode="rerank", alpha=2.0, ft_steps=40000, ref=REF100,
        )
        r117b = train_hops_prior(
            stage=117, tag="hops_loss", phrases=phrases,
            mode="loss", lam=0.3, ft_steps=40000, ref=REF100,
        )

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prior_115": {
                "115a": prior.get("stage115a", {}).get("verdict") if isinstance(prior.get("stage115a"), dict) else None,
            },
            "stage116_morph_diag": {
                "verdict": r116["verdict"],
                "story_morph_gap_pp": r116["story"]["roles"].get("ALL", {}).get("morph_minus_exact_pp"),
                "seen_obj_morph_gap_pp": r116["seen"]["obj"].get("morph_minus_exact_pp"),
            },
            "stage117a_rerank": r117a,
            "stage117b_loss": r117b,
            "note": (
                "hops used only as path/object prior on right slot; "
                "morph@1 diagnostic only; GATE stays exact@1"
            ),
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stages 116–117:** morph diag STORY gap "
                f"+{100*(r116['story']['roles'].get('ALL',{}).get('morph_minus_exact_pp') or 0):.1f}pp; "
                f"hops_rerank {r117a['verdict']} obj={r117a['seen']['obj_exact']*100:.1f}%; "
                f"hops_loss {r117b['verdict']} obj={r117b['seen']['obj_exact']*100:.1f}%. "
                f"`stage116_117_decision.json`.\n"
            )
            if "Stages 116–117" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")
        log("DONE 116/117")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
