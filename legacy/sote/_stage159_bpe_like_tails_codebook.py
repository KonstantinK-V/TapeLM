"""
Stage 159 — BPE-like tails in one codebook (after 156–158).

Problem with 156: morph inventory collapsed to 3 English AFFIXES (ed/s/ing)
via crude_stem+surface — not a rich shared-tail channel like BPE merges.

Here:
  - mine character suffixes (len 1..4) by *type* support across wordforms
  - keep top-K tails as shared +tail atoms (K≈128)
  - stems = $prefix when word = stem+tail under longest-match
  - codebook = words ∪ $stems ∪ +tails (words kept)
  - expand: [$stem, +tail, surface] when match; else [surface]

Trunk = S+ lock (matched to 156). Gate = word exact@1. Hops OUT.

Waits for stage158_compose_layer_decision.json (or 156-158 pipeline rollup).

Run:
  python _stage159_bpe_like_tails_codebook.py
  # or via queue after 156-158
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
import zlib
from collections import Counter
from datetime import datetime, timezone
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
    _warmup_then_constant,
    build_ts_repeat_mix,
    collate_word_id_batch,
    line_subkind,
    load_foundation_85,
)
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402
from _stage156_157_morph_codebook_pipeline import (  # noqa: E402
    TRUNK,
    FT_STEPS,
    WARMUP,
    EVAL_EVERY,
    SEED_MIX,
    make_opt,
)

DEC158 = RES / "stage158_compose_layer_decision.json"
DEC_PIPE = RES / "stage156_158_morph_compose_pipeline_decision.json"
DEC156 = RES / "stage156_shared_morph_codebook_decision.json"
DEC = RES / "stage159_bpe_like_tails_codebook_decision.json"
LOG = RES / "_stage159_bpe_like_tails_log.txt"
PLAN = RES / "plan_150_plus_clean_compare.md"

TAIL_LENS = (1, 2, 3, 4)
MIN_STEM_LEN = 3
MIN_TAIL_TYPES = 8
MAX_TAILS = 128
MAX_STEMS = 2000
MIN_STEM_SUPPORT = 8


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8")) & 0x7FFFFFFF


def wait_upstream(timeout_s=72 * 3600, poll=60):
    log("[wait] for stage158 (or 156-158 pipeline) ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        for path in (DEC158, DEC_PIPE):
            if path.exists():
                d = json.loads(path.read_text(encoding="utf-8"))
                if d.get("verdict"):
                    log(f"[wait] upstream {path.name} verdict={d.get('verdict')}")
                    return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still 158 ... {int(time.time()-t0)}s")
    raise TimeoutError("158 not ready")


def done(path: Path) -> dict | None:
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("verdict"):
            log(f"[skip] {path.name}")
            return d
    return None


def write_dec(path: Path, out: dict):
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"[write] {path.name}")


def mine_bpe_like_tails(train_lines):
    """
    Character-suffix inventory scored by distinct word *types* (BPE-ish reuse).
    Longest-match at expand time prefers longer tails when both kept.
    """
    type_ctr = Counter()  # suffix -> #distinct word types
    tok_ctr = Counter()
    word_types = set()
    for ln in train_lines:
        for w in ln["words"]:
            word_types.add(w)
            for L in TAIL_LENS:
                if len(w) <= L + MIN_STEM_LEN - 1:
                    continue
                stem, tail = w[:-L], w[-L:]
                if len(stem) < MIN_STEM_LEN or not stem.isalpha() or not tail.isalpha():
                    continue
                tok_ctr[tail] += 1
    for w in word_types:
        seen_for_w = set()
        for L in TAIL_LENS:
            if len(w) <= L + MIN_STEM_LEN - 1:
                continue
            stem, tail = w[:-L], w[-L:]
            if len(stem) < MIN_STEM_LEN or not stem.isalpha() or not tail.isalpha():
                continue
            if tail in seen_for_w:
                continue
            seen_for_w.add(tail)
            type_ctr[tail] += 1

    # Prefer high type-support; break ties by token count then longer len
    cands = [
        (t, type_ctr[t], tok_ctr[t], len(t))
        for t in type_ctr
        if type_ctr[t] >= MIN_TAIL_TYPES
    ]
    cands.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    tails = [t for t, *_ in cands[:MAX_TAILS]]
    tail_set = set(tails)

    stem_tok = Counter()
    n_decomp = 0
    for ln in train_lines:
        for w in ln["words"]:
            stem = match_stem(w, tail_set)
            if stem is None:
                continue
            n_decomp += 1
            stem_tok[stem] += 1
    stems = [s for s, c in stem_tok.most_common(MAX_STEMS) if c >= MIN_STEM_SUPPORT]
    meta = {
        "n_decomp_tokens": n_decomp,
        "n_tail_types_kept": len(tails),
        "n_stem_types_kept": len(stems),
        "min_tail_types": MIN_TAIL_TYPES,
        "min_stem_support": MIN_STEM_SUPPORT,
        "max_tails": MAX_TAILS,
        "tail_lens": list(TAIL_LENS),
        "top_tails": [(t, type_ctr[t], tok_ctr[t]) for t in tails[:24]],
        "n_word_types_scanned": len(word_types),
    }
    return tails, stems, meta


def match_stem(w: str, tail_set: set[str]) -> str | None:
    for L in sorted(TAIL_LENS, reverse=True):
        if len(w) <= L + MIN_STEM_LEN - 1:
            continue
        stem, tail = w[:-L], w[-L:]
        if tail in tail_set and len(stem) >= MIN_STEM_LEN and stem.isalpha():
            return stem
    return None


def match_tail(w: str, tail_set: set[str]) -> tuple[str, str] | None:
    for L in sorted(TAIL_LENS, reverse=True):
        if len(w) <= L + MIN_STEM_LEN - 1:
            continue
        stem, tail = w[:-L], w[-L:]
        if tail in tail_set and len(stem) >= MIN_STEM_LEN and stem.isalpha():
            return stem, tail
    return None


def build_codebook(words, tails, stems):
    word_atoms = list(words)
    tail_toks = [f"+{t}" for t in tails]
    stem_toks = [f"${s}" for s in stems]
    pieces = word_atoms + stem_toks + tail_toks
    seen = set()
    uniq = []
    for p in pieces:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    stoi = {p: i for i, p in enumerate(uniq)}
    return uniq, stoi, len(word_atoms), {
        "n_word_atoms": len(word_atoms),
        "n_stem_atoms": len(stem_toks),
        "n_tail_atoms": len(tail_toks),
        "V": len(uniq),
    }


def expand_word(w: str, stoi: dict, mode: str, tail_set: set[str]) -> list[int]:
    if mode == "word_only":
        return [stoi[w]] if w in stoi else []
    out = []
    mt = match_tail(w, tail_set)
    if mt is not None:
        st, tail = mt
        st_tok, at = f"${st}", f"+{tail}"
        if st_tok in stoi and at in stoi:
            out.extend([stoi[st_tok], stoi[at]])
    if w in stoi:
        out.append(stoi[w])
    return out if out else ([stoi[w]] if w in stoi else [])


def expand_line(ws, stoi, mode, tail_set) -> list[int]:
    ids = []
    for w in ws:
        ids.extend(expand_word(w, stoi, mode, tail_set))
    return ids


def pairs_from_expanded_tails(lines, stoi, mode, max_len, tail_set):
    pairs = []
    for li, ln in enumerate(lines):
        ids = expand_line(ln["words"], stoi, mode, tail_set)
        for t in range(1, len(ids)):
            pairs.append(
                {
                    "line_i": li,
                    "prefix_word_ids": ids[:t][-max_len:],
                    "target_word_id": ids[t],
                    "target_word": None,
                    "prefix_len": t,
                    "split": ln.get("split", "x"),
                    "phrase": ln.get("phrase", ""),
                }
            )
    return pairs


@torch.no_grad()
def eval_next_word_exact_tails(model, hold_lines, pieces, stoi, n_word, mode, device, tail_set):
    from collections import defaultdict
    from train import _role, RELS

    roles = defaultdict(lambda: {"n": 0, "h1": 0})
    obj = {"n": 0, "h1": 0}
    for ln in hold_lines:
        ws = ln["words"]
        for t in range(1, len(ws)):
            gold = ws[t]
            pref = expand_line(ws[:t], stoi, mode, tail_set)
            if not pref:
                continue
            pref = pref[-model.max_len :]
            x = torch.tensor([pref], dtype=torch.long, device=device)
            logits = model(x)[0, -1]
            if gold in stoi and stoi[gold] < n_word:
                pred = pieces[int(logits[:n_word].argmax())]
            else:
                pi = int(logits.argmax())
                pred = pieces[pi]
                if pred.startswith("+") or pred.startswith("$"):
                    pred = ""
            ok = int(pred == gold)
            ex = {"target_word": gold, "prefix_len": t}
            role = _role(ex, ws)
            for name in (role, "ALL"):
                roles[name]["n"] += 1
                roles[name]["h1"] += ok
            if t >= 1 and ws[t - 1] in RELS:
                obj["n"] += 1
                obj["h1"] += ok

    def pack(d):
        return {k: {"n": v["n"], "hit1": v["h1"] / max(v["n"], 1)} for k, v in d.items()}

    return {
        "roles": pack(roles),
        "obj": {"hit1": obj["h1"] / max(obj["n"], 1), "n": obj["n"]},
    }


def train_tails_arm(tag, phrases, device, cfg, stack, seed):
    log(f"\n======== {tag} mode=bpe_like_tails ========")
    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=SEED_MIX)
    for ln in train + hold_seen + hold_rare + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")

    words = sorted({w for ln in train + hold_seen + hold_rare + hold_story for w in ln["words"]})
    tails, stems, mine_meta = mine_bpe_like_tails(train)
    tail_set = set(tails)
    pieces, stoi, n_word, vmeta = build_codebook(words, tails, stems)
    log(
        f"[codebook] V={vmeta['V']} words={vmeta['n_word_atoms']} "
        f"stems={vmeta['n_stem_atoms']} tails={vmeta['n_tail_atoms']}"
    )
    log(f"[mine] { {k: mine_meta[k] for k in mine_meta if k != 'top_tails'} }")
    log(f"[mine] top_tails={mine_meta['top_tails'][:16]}")

    n_tok = n_exp = 0
    for ln in train[:5000]:
        for w in ln["words"]:
            n_tok += 1
            mt = match_tail(w, tail_set)
            if mt and f"${mt[0]}" in stoi and f"+{mt[1]}" in stoi:
                n_exp += 1
    cov = n_exp / max(n_tok, 1)
    log(f"[index] tail_expand_coverage~{100*cov:.1f}% (sample)")

    fps = []
    for p in pieces:
        try:
            if p.startswith("$") or p.startswith("+"):
                fps.append(F.normalize(stack.w(p[1:]).detach(), dim=-1))
            else:
                fps.append(F.normalize(stack.w(p).detach(), dim=-1))
        except Exception:
            fps.append(torch.zeros(256, device=device))
    fps_t = torch.stack(fps, 0).to(device)
    model = WordIdTransformer(
        len(pieces), TRUNK["d"], TRUNK["n_head"], TRUNK["n_layer"], TRUNK["max_len"], 0.1
    ).to(device)
    model.init_from_fps(fps_t)
    if TRUNK["d"] > fps_t.shape[1]:
        with torch.no_grad():
            model.tok.weight[: len(pieces), fps_t.shape[1] :].normal_(std=0.02)
            model.tok.weight[: len(pieces)] = F.normalize(model.tok.weight[: len(pieces)], dim=-1)

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    mode = "bpe_like_tails"
    fat_p = pairs_from_expanded_tails(fat, stoi, mode, TRUNK["max_len"], tail_set)
    story_p = pairs_from_expanded_tails(_subsample(story, 40000, seed + 1), stoi, mode, TRUNK["max_len"], tail_set)
    if not fat_p:
        fat_p = story_p

    ev_story = _subsample(hold_story, 300, seed + 11)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 800), seed + 20)
    fin_story = _subsample(hold_story, min(len(hold_story), 500), seed + 21)

    opt = make_opt(model.parameters())
    rr = random.Random(seed)
    best = {
        "story_all": -1.0,
        "obj": -1.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []
    batch = TRUNK["batch"]
    fat_frac = TRUNK["fat_frac"]

    def snap(step):
        model.eval()
        st = eval_next_word_exact_tails(
            model, ev_story, pieces, stoi, n_word, mode, device, tail_set
        )
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        obj = st["obj"]["hit1"]
        curve.append({"step": step, "story_all": sall, "obj": obj})
        log(f"  [{tag}] step {step}: STORY={100*sall:.1f}% obj~={100*obj:.1f}%")
        if (sall, obj) >= (best["story_all"], best["obj"]):
            best.update(
                story_all=sall,
                obj=obj,
                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        model.train()

    model.train()
    snap(0)
    for step in range(1, FT_STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = _warmup_then_constant(step, TRUNK["lr"], WARMUP)
        n_fat = max(1, int(round(batch * fat_frac)))
        batch_ex = [rr.choice(fat_p) for _ in range(n_fat)] + [
            rr.choice(story_p) for _ in range(batch - n_fat)
        ]
        for ex in batch_ex:
            tid = ex["target_word_id"]
            ex["target_word"] = pieces[tid]
        packed = collate_word_id_batch(batch_ex, stoi, model.max_len, model.pad_id, device)
        if packed is None:
            continue
        ids, mask, tgt = packed
        loss = F.cross_entropy(model.logits_last_from_batch(ids, mask), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0 or step == FT_STEPS:
            snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    fs = eval_next_word_exact_tails(
        model, fin_seen, pieces, stoi, n_word, mode, device, tail_set
    )
    ft = eval_next_word_exact_tails(
        model, fin_story, pieces, stoi, n_word, mode, device, tail_set
    )
    ck = CKPT / f"stage159_{tag}.pt"
    torch.save(
        {
            "word_tf": best["state"],
            "pieces": pieces,
            "n_word": n_word,
            "mode": mode,
            "tails": tails,
            "stems": stems,
        },
        ck,
    )
    return {
        "arm": tag,
        "mode": mode,
        "codebook": vmeta,
        "mine": mine_meta,
        "tail_expand_coverage": cov,
        "seen_obj": fs["obj"]["hit1"],
        "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
        "story_obj": ft["obj"]["hit1"],
        "ckpt": str(ck),
        "curve": curve,
        "trunk": TRUNK,
    }


def main():
    RES.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    if not LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"159 bpe-like tails start {datetime.now(timezone.utc).isoformat()}")
    log(f"plan={PLAN}")
    prev = done(DEC)
    if prev:
        return 0
    try:
        up = wait_upstream()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = Config()
        cfg.c87_n_fat, cfg.c87_n_rare = 200, 80
        cfg.c87_fat_copies, cfg.c87_rare_copies = 40, 2
        cfg.c87_seen_hold_frac, cfg.c87_story_keep_frac, cfg.c87_hold_frac = 0.2, 0.5, 0.15
        parent = CKPT / "stage85_foundation_FROZEN.pt"
        if not parent.exists():
            parent = CKPT / "stage85_foundation.pt"
        stack, _ = load_foundation_85(device, cfg, path=parent)
        for mod in (stack.encoder, stack.composer, stack.morph, stack.phrase, stack.binder):
            for p in mod.parameters():
                p.requires_grad_(False)
            mod.eval()
        phrases = ensure_100k()

        arm = train_tails_arm(
            "159_bpe_like_tails",
            phrases,
            device,
            cfg,
            stack,
            seed=stable_seed(159, "tails"),
        )

        ctrl_s = None
        morph_s = None
        if DEC156.exists():
            d156 = json.loads(DEC156.read_text(encoding="utf-8"))
            ctrl_s = (d156.get("ctrl_word") or {}).get("story_all")
            morph_s = (d156.get("shared_morph") or {}).get("story_all")
        lift_c = (arm["story_all"] - ctrl_s) if ctrl_s is not None else None
        lift_m = (arm["story_all"] - morph_s) if morph_s is not None else None
        # gate vs 156 ctrl (same S+ ritual)
        if lift_c is None:
            verdict = "TABLE"
        elif lift_c >= 0.03:
            verdict = "PASS"
        elif lift_c >= 0.015:
            verdict = "PARTIAL"
        elif lift_c >= -0.02:
            verdict = "PARITY"
        else:
            verdict = "HARM"

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "bpe_like_tails_in_one_codebook",
            "framing": (
                "Rich shared character-suffix inventory (not only AFFIXES ed/s/ing). "
                "Same codebook entity: words kept + $stem + +tail; longest-match expand."
            ),
            "bpe_like_tails": arm,
            "story_lift_vs_156_ctrl": lift_c,
            "story_lift_vs_156_shared_morph": lift_m,
            "verdict": verdict,
            "ref_156_ctrl_story": ctrl_s,
            "ref_156_shared_morph_story": morph_s,
            "upstream": {"158_or_pipe": up.get("verdict")},
            "ref100": REF100,
            "vs_156_note": (
                "156 used crude_stem→3 affixes; 159 mines char tails by type-support. "
                "Coverage should be much higher if BPE-like reuse matters."
            ),
        }
        write_dec(DEC, out)
        lc = f"{100*lift_c:+.1f}pp" if lift_c is not None else "n/a"
        log(f"[159] {verdict} vs_156_ctrl={lc} story={100*arm['story_all']:.1f}%")
        return 0
    except Exception as e:
        log(f"[FAIL] {e}\n{traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
