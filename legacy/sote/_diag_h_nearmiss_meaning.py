"""Near-miss meaning dig: when @1 wrong but @5 hit, is pred near gold in F85 fp space?"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    CKPT,
    RES,
    Config,
    WordIdTransformer,
    _role,
    build_ts_repeat_mix,
    line_subkind,
    load_foundation_85,
)
from _stage109_110_slot_baseline import ensure_100k, _subsample  # noqa: E402
from _stage116_117_hops_morph_follow import crude_stem, morph_hit  # noqa: E402


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def analyze(model, fps, words, stoi, hold, tag, n=400):
    hold = _subsample(hold, min(n, len(hold)), 21)
    buckets = defaultdict(
        lambda: {
            "n": 0,
            "h1": 0,
            "h5": 0,
            "miss_in5": 0,
            "miss_out5": 0,
            "cos_hit": [],
            "cos_miss5": [],
            "cos_miss_out": [],
            "cos_rand": [],
            "rank_miss5": [],
            "rank_miss_out": [],
            "same_role": 0,
            "morph_miss": 0,
            "same_stem": 0,
            "margin_miss5": [],  # logit[pred]-logit[gold]
        }
    )
    rng = random.Random(0)
    with torch.no_grad():
        for ln in hold:
            ws = ln["words"]
            ids = [stoi[w] for w in ws if w in stoi]
            if len(ids) != len(ws):
                continue
            for t in range(1, len(ws)):
                gold = ws[t]
                gid = stoi[gold]
                ex = {"prefix_len": t, "target_word": gold}
                role = _role(ex, ws)
                logits = model.logits_from_prefix(ids[:t][-model.max_len :])
                order = torch.argsort(logits, descending=True)
                pred_i = int(order[0])
                rank = int((order == gid).nonzero()[0]) + 1
                top5 = {int(i) for i in order[:5].tolist()}
                pred = words[pred_i]
                h1 = pred_i == gid
                h5 = gid in top5
                for key in (role, "ALL"):
                    B = buckets[key]
                    B["n"] += 1
                    B["h1"] += int(h1)
                    B["h5"] += int(h5)
                    cos_pg = float((fps[pred_i] * fps[gid]).sum())
                    rj = rng.randrange(len(words))
                    B["cos_rand"].append(float((fps[pred_i] * fps[rj]).sum()))
                    if h1:
                        B["cos_hit"].append(1.0)  # identical
                    elif h5:
                        B["miss_in5"] += 1
                        B["cos_miss5"].append(cos_pg)
                        B["rank_miss5"].append(rank)
                        B["margin_miss5"].append(float(logits[pred_i] - logits[gid]))
                        pred_role = _role({"prefix_len": t, "target_word": pred}, ws)
                        B["same_role"] += int(pred_role == role)
                        B["morph_miss"] += int(morph_hit(pred, gold))
                        B["same_stem"] += int(crude_stem(pred) == crude_stem(gold))
                    else:
                        B["miss_out5"] += 1
                        B["cos_miss_out"].append(cos_pg)
                        B["rank_miss_out"].append(rank)

    out = {}
    for k, v in buckets.items():
        n = max(v["n"], 1)
        mi = v["miss_in5"]
        out[k] = {
            "n": v["n"],
            "hit1": v["h1"] / n,
            "hit5": v["h5"] / n,
            "miss_in_top5_frac": v["miss_in5"] / n,
            "miss_out_top5_frac": v["miss_out5"] / n,
            "mean_cos_miss_in5_pred_gold": mean(v["cos_miss5"]),
            "mean_cos_miss_out5_pred_gold": mean(v["cos_miss_out"]),
            "mean_cos_pred_random": mean(v["cos_rand"]),
            "mean_rank_when_miss_in5": mean(v["rank_miss5"]),
            "mean_rank_when_miss_out5": mean(v["rank_miss_out"]),
            "mean_logit_margin_miss_in5": mean(v["margin_miss5"]),
            "among_miss_in5_same_role_frac": (v["same_role"] / mi) if mi else None,
            "among_miss_in5_morph_frac": (v["morph_miss"] / mi) if mi else None,
            "among_miss_in5_same_stem_frac": (v["same_stem"] / mi) if mi else None,
        }
    print(f"=== {tag} ===")
    for k in ("ALL", "rel", "right", "other"):
        if k not in out:
            continue
        o = out[k]
        print(
            f"{k}: n={o['n']} hit1={o['hit1']*100:.1f}% hit5={o['hit5']*100:.1f}% "
            f"miss@5={o['miss_in_top5_frac']*100:.1f}% miss>5={o['miss_out_top5_frac']*100:.1f}%"
        )
        c5 = o["mean_cos_miss_in5_pred_gold"]
        c_out = o["mean_cos_miss_out5_pred_gold"]
        cr = o["mean_cos_pred_random"]
        print(
            f"  cos miss@5 pred-gold={None if c5 is None else f'{c5:.3f}'} "
            f"cos miss>5={None if c_out is None else f'{c_out:.3f}'} "
            f"cos pred-rand={None if cr is None else f'{cr:.3f}'}"
        )
        print(
            f"  rank miss@5={o['mean_rank_when_miss_in5']} "
            f"margin={o['mean_logit_margin_miss_in5']} "
            f"same_role={o['among_miss_in5_same_role_frac']} "
            f"morph={o['among_miss_in5_morph_frac']} stem={o['among_miss_in5_same_stem_frac']}"
        )
    return out


def main():
    device = torch.device("cpu")
    cfg = Config()
    cfg.c87_n_fat, cfg.c87_n_rare = 200, 80
    cfg.c87_fat_copies, cfg.c87_rare_copies = 40, 2
    cfg.c87_seen_hold_frac = 0.2
    cfg.c87_story_keep_frac = 0.5
    cfg.c87_hold_frac = 0.15
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
    fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1)
    model = WordIdTransformer(len(words), 256, 4, 2, 16, 0.1)
    ck = torch.load(CKPT / "stage100_scale_100k.pt", map_location="cpu", weights_only=False)
    if list(ck.get("surfaces", [])) != words:
        raise SystemExit("vocab mismatch with stage100")
    model.load_state_dict(ck["word_tf"], strict=True)
    model.eval()

    seen = analyze(model, fps, words, stoi, hold_seen, "SEEN", 500)
    story = analyze(model, fps, words, stoi, hold_story, "STORY", 500)
    res = {
        "note": (
            "Selection = plain argmax over CE logits (no separate decoder). "
            "Meaning proxy = F85 frozen word_fp cosine(pred, gold). "
            "Critical if miss@5 cos ≈ random (wrong sense); soft if cos >> random."
        ),
        "seen": seen,
        "story": story,
    }
    path = RES / "stage124b_nearmiss_meaning.json"
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("saved", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
