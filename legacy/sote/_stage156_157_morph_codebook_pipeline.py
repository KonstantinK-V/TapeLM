"""
Stages 156-158 — morph codebook + ComposeLayer (after 150+ hparams).

  156  words ∪ data-driven morphs in one codebook
  157  rare surfaces drop word-id (morph-only path)
  158  ComposeLayer as *separate preprocess*: freeze compose index on
       dataset → LM trains only on composed id sequences; word HOLD at eval

Trunk = S+ lock. Gate = word exact@1. Hops OUT.

Waits for stage150_155_clean_compare_pipeline_decision.json

Run:
  python _stage156_157_morph_codebook_pipeline.py
  # or via queue: python _run_queue_150_158.py
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
    AFFIXES,
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
    surface,
)
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402
from _stage116_117_hops_morph_follow import crude_stem  # noqa: E402

DEC_PIPE = RES / "stage150_155_clean_compare_pipeline_decision.json"
DEC150 = RES / "stage150_dual_trunk_decision.json"
LOG = RES / "_stage156_158_log.txt"
PIPELINE_DEC = RES / "stage156_158_morph_compose_pipeline_decision.json"
PLAN = RES / "plan_150_plus_clean_compare.md"
COMPOSE_INDEX = RES / "stage158_compose_index.json"
COMPOSE_CORPUS = ROOT / "data" / "stage158_composed_sequences.jsonl"

# S+ lock (must match plan_150_plus / stage150 TRUNKS["Splus"])
TRUNK = {
    "batch": 8,
    "lr": 1e-3,
    "opt": "Adam",
    "wd": 0.0,
    "d": 512,
    "n_layer": 4,
    "n_head": 8,
    "fat_frac": 0.75,
    "max_len": 40,  # longer: morph expand needs room
}
FT_STEPS = 40_000
WARMUP = 200
EVAL_EVERY = 4000
SEED_MIX = 272
MORPH_MIN_SUPPORT = 8
MAX_STEM_ATOMS = 2000
RARE_MAX_COUNT = 2  # 157: surfaces with train count <= this lose word-id


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
    """Prefer full 150-155 rollup; do not start morph digs on 150-only when queueing."""
    log("[wait] for stage150_155_clean_compare_pipeline_decision.json ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC_PIPE.exists():
            d = json.loads(DEC_PIPE.read_text(encoding="utf-8"))
            if d.get("verdict"):
                log(f"[wait] pipe150-155 done verdict={d.get('verdict')}")
                return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still 150-155 ... {int(time.time()-t0)}s")
    raise TimeoutError("150-155 rollup not ready")


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


def make_opt(params):
    if TRUNK["opt"] == "AdamW":
        return torch.optim.AdamW(params, lr=TRUNK["lr"], weight_decay=TRUNK["wd"])
    return torch.optim.Adam(params, lr=TRUNK["lr"])


def affix_of(word: str, stem: str):
    if not stem or stem == word:
        return None
    for a in AFFIXES:
        try:
            if surface(stem, a) == word:
                return a
        except Exception:
            continue
    return None


def mine_morphs(train_lines, min_support=MORPH_MIN_SUPPORT, max_stems=MAX_STEM_ATOMS):
    """Data-driven shared morph inventory from corpus (one codebook entity add-on)."""
    aff_ctr = Counter()
    stem_ctr = Counter()
    n_decomp = 0
    for ln in train_lines:
        for w in ln["words"]:
            st = crude_stem(w) or w
            aff = affix_of(w, st)
            if aff is None:
                continue
            n_decomp += 1
            aff_ctr[aff] += 1
            stem_ctr[st] += 1
    affixes = [a for a, c in aff_ctr.most_common() if c >= min_support]
    stems = [s for s, c in stem_ctr.most_common(max_stems) if c >= min_support]
    meta = {
        "n_decomp_tokens": n_decomp,
        "n_affix_types_kept": len(affixes),
        "n_stem_types_kept": len(stems),
        "min_support": min_support,
        "top_affixes": aff_ctr.most_common(12),
    }
    return affixes, stems, meta


def build_codebook(words, affixes, stems, drop_rare_word_ids: set[str] | None = None):
    """
    One codebook entity:
      - word surfaces (optional drop rare for 157)
      - shared +aff
      - shared $stem
    """
    drop = drop_rare_word_ids or set()
    word_atoms = [w for w in words if w not in drop]
    aff_toks = [f"+{a}" for a in affixes]
    stem_toks = [f"${s}" for s in stems]
    pieces = word_atoms + stem_toks + aff_toks
    # stable unique
    seen = set()
    uniq = []
    for p in pieces:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    stoi = {p: i for i, p in enumerate(uniq)}
    n_word = len(word_atoms)
    return uniq, stoi, n_word, {
        "n_word_atoms": n_word,
        "n_stem_atoms": len(stem_toks),
        "n_affix_atoms": len(aff_toks),
        "V": len(uniq),
        "n_dropped_rare_words": len(drop),
    }


def expand_word(w: str, stoi: dict, mode: str) -> list[int]:
    """
    mode:
      word_only — surface if present else []
      shared_morph — if decomposable and morph ids exist: $stem +aff [+ surface if in stoi]
      morph_only_rare — like shared but if surface missing, morph path only
    """
    if mode == "word_only":
        return [stoi[w]] if w in stoi else []

    st = crude_stem(w) or w
    aff = affix_of(w, st)
    out = []
    if aff is not None:
        st_tok, at = f"${st}", f"+{aff}"
        if st_tok in stoi and at in stoi:
            out.extend([stoi[st_tok], stoi[at]])
    if w in stoi:
        out.append(stoi[w])
    elif mode == "morph_only_rare" and out:
        return out
    return out if out else ([stoi[w]] if w in stoi else [])


def expand_line(ws, stoi, mode) -> list[int]:
    ids = []
    for w in ws:
        ids.extend(expand_word(w, stoi, mode))
    return ids


def pairs_from_expanded(lines, stoi, mode, max_len):
    """Teacher pairs on expanded stream; target may be morph or word id."""
    pairs = []
    for li, ln in enumerate(lines):
        ids = expand_line(ln["words"], stoi, mode)
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
def eval_next_word_exact(model, hold_lines, pieces, stoi, n_word, mode, device):
    from collections import defaultdict
    from train import _role, RELS

    roles = defaultdict(lambda: {"n": 0, "h1": 0})
    obj = {"n": 0, "h1": 0}
    for ln in hold_lines:
        ws = ln["words"]
        for t in range(1, len(ws)):
            gold = ws[t]
            pref = expand_line(ws[:t], stoi, mode)
            if not pref:
                continue
            pref = pref[-model.max_len :]
            x = torch.tensor([pref], dtype=torch.long, device=device)
            logits = model(x)[0, -1]
            if gold in stoi and stoi[gold] < n_word:
                # gold is a word-atom — rank among word atoms
                pred = pieces[int(logits[:n_word].argmax())]
            else:
                # 157 rare: no word id — take full argmax; hit only if decoded path
                # equals gold via morph (weak): require pred piece string == gold
                # or stem+aff compose equals gold
                pi = int(logits.argmax())
                pred = pieces[pi]
                if pred.startswith("+") or pred.startswith("$"):
                    # cannot confirm word from single morph piece alone → miss
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


def train_morph_arm(
    tag: str,
    phrases,
    device,
    cfg,
    stack,
    *,
    mode: str,
    drop_rare: bool,
    seed: int,
):
    log(f"\n======== {tag} mode={mode} drop_rare={drop_rare} ========")
    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=SEED_MIX)
    for ln in train + hold_seen + hold_rare + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")

    words = sorted({w for ln in train + hold_seen + hold_rare + hold_story for w in ln["words"]})
    unigram = Counter(w for ln in train for w in ln["words"])
    affixes, stems, mine_meta = mine_morphs(train)
    drop = set()
    if drop_rare:
        drop = {w for w, c in unigram.items() if c <= RARE_MAX_COUNT}
        # never drop if cannot morph-decompose
        drop = {w for w in drop if affix_of(w, crude_stem(w) or w) is not None}

    pieces, stoi, n_word, vmeta = build_codebook(words, affixes, stems, drop_rare_word_ids=drop)
    log(
        f"[codebook] V={vmeta['V']} words={vmeta['n_word_atoms']} "
        f"stems={vmeta['n_stem_atoms']} aff={vmeta['n_affix_atoms']} dropped={vmeta['n_dropped_rare_words']}"
    )
    log(f"[mine] {mine_meta}")

    # coverage: share of train word tokens that expand with shared morph
    n_tok = n_exp = 0
    for ln in train[:5000]:
        for w in ln["words"]:
            n_tok += 1
            st = crude_stem(w) or w
            aff = affix_of(w, st)
            if aff and f"${st}" in stoi and f"+{aff}" in stoi:
                n_exp += 1
    cov = n_exp / max(n_tok, 1)
    log(f"[index] morph_expand_coverage~{100*cov:.1f}% (sample)")

    fps = []
    for p in pieces:
        try:
            if p.startswith("$") or p.startswith("+"):
                fps.append(F.normalize(stack.w(p[1:]).detach(), dim=-1))
            else:
                fps.append(F.normalize(stack.w(p).detach(), dim=-1))
        except Exception:
            fps.append(torch.zeros(256, device=device))
    # stack.w dim = foundation 256
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
    fat_p = pairs_from_expanded(fat, stoi, mode, TRUNK["max_len"])
    story_p = pairs_from_expanded(_subsample(story, 40000, seed + 1), stoi, mode, TRUNK["max_len"])
    if not fat_p:
        fat_p = story_p

    ev_story = _subsample(hold_story, 300, seed + 11)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 800), seed + 20)
    fin_story = _subsample(hold_story, min(len(hold_story), 500), seed + 21)

    opt = make_opt(model.parameters())
    rr = random.Random(seed)
    best = {"story_all": -1.0, "obj": -1.0, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    curve = []
    batch = TRUNK["batch"]
    fat_frac = TRUNK["fat_frac"]

    def snap(step):
        model.eval()
        st = eval_next_word_exact(model, ev_story, pieces, stoi, n_word, mode, device)
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        obj = st["obj"]["hit1"]
        curve.append({"step": step, "story_all": sall, "obj": obj})
        log(f"  [{tag}] step {step}: STORY={100*sall:.1f}% obj~={100*obj:.1f}%")
        if (sall, obj) >= (best["story_all"], best["obj"]):
            best.update(story_all=sall, obj=obj, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
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
        # collate expects target_word in stoi — patch fake
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
    fs = eval_next_word_exact(model, fin_seen, pieces, stoi, n_word, mode, device)
    ft = eval_next_word_exact(model, fin_story, pieces, stoi, n_word, mode, device)
    ck = CKPT / f"stage156_157_{tag}.pt"
    torch.save({"word_tf": best["state"], "pieces": pieces, "n_word": n_word, "mode": mode}, ck)
    return {
        "arm": tag,
        "mode": mode,
        "drop_rare": drop_rare,
        "codebook": vmeta,
        "mine": {k: mine_meta[k] for k in ("n_affix_types_kept", "n_stem_types_kept", "min_support", "n_decomp_tokens")},
        "morph_expand_coverage": cov,
        "seen_obj": fs["obj"]["hit1"],
        "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
        "story_obj": ft["obj"]["hit1"],
        "ckpt": str(ck),
        "curve": curve,
        "trunk": TRUNK,
    }


def run_ctrl_word(phrases, device, cfg, stack, seed):
    """Plain word atoms only under S+ (matched ritual control)."""
    log("\n======== CTRL word_only S+ ========")
    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=SEED_MIX)
    for ln in train + hold_seen + hold_rare + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")
    words = sorted({w for ln in train + hold_seen + hold_rare + hold_story for w in ln["words"]})
    stoi = {s: i for i, s in enumerate(words)}
    fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)
    model = WordIdTransformer(
        len(words), TRUNK["d"], TRUNK["n_head"], TRUNK["n_layer"], 16, 0.1
    ).to(device)
    model.init_from_fps(fps)
    if TRUNK["d"] > fps.shape[1]:
        with torch.no_grad():
            model.tok.weight[: len(words), fps.shape[1] :].normal_(std=0.02)
            model.tok.weight[: len(words)] = F.normalize(model.tok.weight[: len(words)], dim=-1)

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_p = lines_to_pairs(fat, stoi)
    story_p = lines_to_pairs(_subsample(story, 60000, seed + 1), stoi)
    ev_story = _subsample(hold_story, 300, seed + 11)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 800), seed + 20)
    fin_story = _subsample(hold_story, min(len(hold_story), 500), seed + 21)
    opt = make_opt(model.parameters())
    rr = random.Random(seed)
    best = {"story_all": -1.0, "obj": -1.0, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    batch, fat_frac = TRUNK["batch"], TRUNK["fat_frac"]

    def snap(step):
        model.eval()
        st = eval_id_capacity_suite(model, ev_story, train, words, stoi, device)
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        obj = st["obj"]["hit1"]
        log(f"  [ctrl_word] step {step}: STORY={100*sall:.1f}%")
        if (sall, obj) >= (best["story_all"], best["obj"]):
            best.update(story_all=sall, obj=obj, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
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
    fs = eval_id_capacity_suite(model, fin_seen, train, words, stoi, device)
    ft = eval_id_capacity_suite(model, fin_story, train, words, stoi, device)
    ck = CKPT / "stage156_ctrl_word_splus.pt"
    torch.save({"word_tf": best["state"], "surfaces": words}, ck)
    return {
        "arm": "ctrl_word_splus",
        "V": len(words),
        "seen_obj": fs["obj"]["hit1"],
        "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
        "story_obj": ft["obj"]["hit1"],
        "ckpt": str(ck),
        "trunk": TRUNK,
    }


def build_compose_layer_and_corpus(phrases, cfg, seed=SEED_MIX):
    """
    Separate ComposeLayer (offline dataset processing):
      mine morphs → freeze codebook → write composed id sequences for every line.
    LM later reads only those ids (no on-the-fly morph rules in the train loop).
    """
    train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=seed)
    for ln in train + hold_seen + hold_rare + hold_story:
        ln.setdefault("subkind", line_subkind(ln))
        ln["split"] = ln.get("bucket", "x")
    words = sorted({w for ln in train + hold_seen + hold_rare + hold_story for w in ln["words"]})
    unigram = Counter(w for ln in train for w in ln["words"])
    affixes, stems, mine_meta = mine_morphs(train)
    # Frequent wholes stay as word-atoms; rare decomposable → morph-only in compose stream
    drop = {
        w
        for w, c in unigram.items()
        if c <= RARE_MAX_COUNT and affix_of(w, crude_stem(w) or w) is not None
    }
    pieces, stoi, n_word, vmeta = build_codebook(words, affixes, stems, drop_rare_word_ids=drop)
    mode = "morph_only_rare"

    def pack_split(lines, split_name):
        rows = []
        for ln in lines:
            ids = expand_line(ln["words"], stoi, mode)
            rows.append(
                {
                    "split": split_name,
                    "bucket": ln.get("bucket"),
                    "words": ln["words"],
                    "composed_ids": ids,
                }
            )
        return rows

    corpus = (
        pack_split(train, "train")
        + pack_split(hold_seen, "hold_seen")
        + pack_split(hold_rare, "hold_rare")
        + pack_split(hold_story, "hold_story")
    )
    COMPOSE_CORPUS.parent.mkdir(exist_ok=True)
    with COMPOSE_CORPUS.open("w", encoding="utf-8") as f:
        for row in corpus:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    index = {
        "pieces": pieces,
        "n_word": n_word,
        "vmeta": vmeta,
        "mine": mine_meta,
        "drop_rare_max_count": RARE_MAX_COUNT,
        "mode": mode,
        "corpus": str(COMPOSE_CORPUS),
        "n_rows": len(corpus),
    }
    COMPOSE_INDEX.write_text(json.dumps(index, indent=2), encoding="utf-8")
    log(f"[158 compose] wrote {COMPOSE_CORPUS.name} rows={len(corpus)} V={vmeta['V']}")
    return index, train, hold_seen, hold_story, pieces, stoi, n_word, mode


def train_compose_lm(tag, index, stack, device, seed):
    """LM on frozen composed corpus only (ComposeLayer already applied)."""
    log(f"\n======== {tag} ComposeLayer-LM (preprocess frozen) ========")
    pieces = index["pieces"]
    n_word = index["n_word"]
    stoi = {p: i for i, p in enumerate(pieces)}
    mode = index["mode"]

    rows = []
    with COMPOSE_CORPUS.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    train_rows = [r for r in rows if r["split"] == "train"]
    hold_seen = [r for r in rows if r["split"] == "hold_seen"]
    hold_story = [r for r in rows if r["split"] == "hold_story"]

    fat_seqs = [r["composed_ids"] for r in train_rows if r.get("bucket") == "fat_train" and len(r["composed_ids"]) >= 2]
    story_seqs = [r["composed_ids"] for r in train_rows if r.get("bucket") != "fat_train" and len(r["composed_ids"]) >= 2]
    if not fat_seqs:
        fat_seqs = story_seqs
    if not story_seqs:
        story_seqs = fat_seqs

    fps = []
    for p in pieces:
        try:
            raw = p[1:] if p.startswith("$") or p.startswith("+") else p
            fps.append(F.normalize(stack.w(raw).detach(), dim=-1))
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

    # teacher pairs from frozen composed ids
    def seqs_to_pairs(seqs, cap, s0):
        rr = random.Random(s0)
        seqs = seqs if len(seqs) <= cap else [seqs[i] for i in rr.sample(range(len(seqs)), cap)]
        pairs = []
        for ids in seqs:
            for t in range(1, len(ids)):
                pairs.append(
                    {
                        "prefix_word_ids": ids[:t][-TRUNK["max_len"] :],
                        "target_word_id": ids[t],
                        "target_word": pieces[ids[t]],
                    }
                )
        return pairs

    fat_p = seqs_to_pairs(fat_seqs, 20000, seed + 1)
    story_p = seqs_to_pairs(story_seqs, 60000, seed + 2)
    ev_story_lines = [{"words": r["words"]} for r in _subsample(hold_story, 300, seed + 11)]
    fin_seen_lines = [{"words": r["words"]} for r in _subsample(hold_seen, min(len(hold_seen), 800), seed + 20)]
    fin_story_lines = [{"words": r["words"]} for r in _subsample(hold_story, min(len(hold_story), 500), seed + 21)]

    opt = make_opt(model.parameters())
    rr = random.Random(seed)
    best = {
        "story_all": -1.0,
        "obj": -1.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    batch, fat_frac = TRUNK["batch"], TRUNK["fat_frac"]
    curve = []

    def snap(step):
        model.eval()
        st = eval_next_word_exact(model, ev_story_lines, pieces, stoi, n_word, mode, device)
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        obj = st["obj"]["hit1"]
        curve.append({"step": step, "story_all": sall, "obj": obj})
        log(f"  [{tag}] step {step}: STORY={100*sall:.1f}%")
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
    fs = eval_next_word_exact(model, fin_seen_lines, pieces, stoi, n_word, mode, device)
    ft = eval_next_word_exact(model, fin_story_lines, pieces, stoi, n_word, mode, device)
    ck = CKPT / f"stage158_{tag}.pt"
    torch.save({"word_tf": best["state"], "pieces": pieces, "compose_index": str(COMPOSE_INDEX)}, ck)
    return {
        "arm": tag,
        "protocol": "compose_layer_preprocess_then_lm",
        "compose_index": str(COMPOSE_INDEX),
        "compose_corpus": str(COMPOSE_CORPUS),
        "V": len(pieces),
        "n_word": n_word,
        "seen_obj": fs["obj"]["hit1"],
        "story_all": ft["roles"].get("ALL", {}).get("hit1", 0.0),
        "story_obj": ft["obj"]["hit1"],
        "ckpt": str(ck),
        "curve": curve,
        "trunk": TRUNK,
    }


def run_158(phrases, device, cfg, stack, d156):
    dec = RES / "stage158_compose_layer_decision.json"
    prev = done(dec)
    if prev:
        return prev
    log("\n########## STAGE 158 ComposeLayer preprocess + LM ##########")
    index, *_rest = build_compose_layer_and_corpus(phrases, cfg)
    compose = train_compose_lm("158_compose_lm", index, stack, device, seed=stable_seed(158, "lm"))
    ctrl_s = (d156.get("ctrl_word") or {}).get("story_all") or 0.0
    s157 = None
    p157 = RES / "stage157_freq_morph_only_rare_decision.json"
    if p157.exists():
        s157 = (json.loads(p157.read_text(encoding="utf-8")).get("morph_only_rare") or {}).get("story_all")
    lift_c = compose["story_all"] - ctrl_s
    lift_157 = (compose["story_all"] - s157) if s157 is not None else None
    verdict = (
        "PASS" if lift_c >= 0.03 else "PARTIAL" if lift_c >= 0.015 else "PARITY" if lift_c >= -0.02 else "HARM"
    )
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "compose_layer_separate_preprocess",
        "framing": (
            "ComposeLayer freezes composition at dataset indexing; LM sees only composed ids. "
            "Word identity remains for HOLD decode/confirm."
        ),
        "compose_lm": compose,
        "story_lift_vs_ctrl": lift_c,
        "story_lift_vs_157": lift_157,
        "verdict": verdict,
        "ref_ctrl_story": ctrl_s,
        "ref_157_story": s157,
    }
    write_dec(dec, out)
    log(f"[158] {verdict} vs_ctrl={100*lift_c:+.1f}pp")
    return out


def main():
    RES.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    if not LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"156-158 morph+compose start {datetime.now(timezone.utc).isoformat()}")
    log(f"plan={PLAN}")
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

        # --- 156 ---
        dec156 = RES / "stage156_shared_morph_codebook_decision.json"
        d156 = done(dec156)
        if not d156:
            ctrl = run_ctrl_word(phrases, device, cfg, stack, seed=stable_seed(156, "ctrl"))
            shared = train_morph_arm(
                "156_shared_morph",
                phrases,
                device,
                cfg,
                stack,
                mode="shared_morph",
                drop_rare=False,
                seed=stable_seed(156, "shared"),
            )
            lift = shared["story_all"] - ctrl["story_all"]
            verdict = (
                "PASS" if lift >= 0.03 else "PARTIAL" if lift >= 0.015 else "PARITY" if lift >= -0.02 else "HARM"
            )
            d156 = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "protocol": "shared_morph_in_one_codebook_words_kept",
                "framing": (
                    "Codebook=one indexing entity; add data-driven morph atoms so endings "
                    "are shared ids; words remain atoms."
                ),
                "ctrl_word": ctrl,
                "shared_morph": shared,
                "story_lift_vs_ctrl": lift,
                "verdict": verdict,
                "upstream": {"pipe": up.get("verdict")},
                "ref100": REF100,
            }
            write_dec(dec156, d156)
            log(f"[156] {verdict} lift={100*lift:+.1f}pp story={100*shared['story_all']:.1f}%")

        # --- 157 ---
        dec157 = RES / "stage157_freq_morph_only_rare_decision.json"
        d157 = done(dec157)
        if not d157:
            rare = train_morph_arm(
                "157_morph_only_rare",
                phrases,
                device,
                cfg,
                stack,
                mode="morph_only_rare",
                drop_rare=True,
                seed=stable_seed(157, "rare"),
            )
            ctrl_s = (d156.get("ctrl_word") or {}).get("story_all") or 0.0
            shared_s = (d156.get("shared_morph") or {}).get("story_all") or 0.0
            lift_c = rare["story_all"] - ctrl_s
            lift_s = rare["story_all"] - shared_s
            verdict = (
                "PASS" if lift_c >= 0.03 else "PARTIAL" if lift_c >= 0.015 else "PARITY" if lift_c >= -0.02 else "HARM"
            )
            d157 = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "protocol": "freq_gated_rare_without_word_id",
                "framing": (
                    "Stronger BPE-analog: rare decomposable surfaces dropped from word-atoms; "
                    "morph-only indexing path; eval still word HOLD."
                ),
                "morph_only_rare": rare,
                "story_lift_vs_ctrl": lift_c,
                "story_lift_vs_156_shared": lift_s,
                "verdict": verdict,
                "ref_ctrl_story": ctrl_s,
                "ref_156_story": shared_s,
            }
            write_dec(dec157, d157)
            log(f"[157] {verdict} vs_ctrl={100*lift_c:+.1f}pp vs_156={100*lift_s:+.1f}pp")

        # --- 158 ---
        d158 = run_158(phrases, device, cfg, stack, d156)

        pipe = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "morph_codebook_156_158",
            "plan": str(PLAN),
            "stages": {
                "156": d156.get("verdict"),
                "157": d157.get("verdict"),
                "158": d158.get("verdict"),
            },
            "verdict": "PIPELINE_DONE",
        }
        write_dec(PIPELINE_DEC, pipe)
        # compat alias
        (RES / "stage156_157_morph_codebook_pipeline_decision.json").write_text(
            json.dumps(pipe, indent=2), encoding="utf-8"
        )
        log("DONE 156-158")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
