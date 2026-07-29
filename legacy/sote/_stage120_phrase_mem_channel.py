"""
Stage 120 — phrase-memory as second channel (SOTE-native), after 119.

Thesis:
  Don't chase STORY ALL by lengthening the word-TF alone.
  Channel A: WordIdTransformer next-word on current sentence window.
  Channel B: SoftPhraseMemory over *previous sentences in the same story*;
             retrieve by current-prefix phrase_fp; boost logits for words
             that appear in retrieved phrases (narrative prior).

Data: rebuild multi-sentence episodes from TinyStories raw (same a-z+digit filter).
Eval:
  - STORY_LOCAL: current window only (no memory) — comparable to Stage100
  - STORY_MEM: same windows with previous sentences in memory
  Gate: exact@1. PASS if STORY_MEM ALL >= STORY_LOCAL + 3pp with SEEN held.

Run (waits for stage119_decision.json):
  python _stage120_phrase_mem_channel.py
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    CHAR2ID,
    CKPT,
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
)

DEC_PRIOR = RES / "stage119_decision.json"
LOG = RES / "_stage120_log.txt"
DEC = RES / "stage120_decision.json"
RAW = ROOT / "data" / "_tinystories_raw_scale.txt"
RAW_FALLBACK = ROOT / "data" / "_tinystories_raw_100k.txt"
ATOM100 = CKPT / "stage100_scale_100k.pt"
REF100 = {"obj": 0.426, "rel": 0.958, "story_all": 0.196}


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def wait_prior(timeout_s: int = 12 * 3600, poll_s: int = 45) -> dict:
    log(f"[wait] for {DEC_PRIOR} ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC_PRIOR.exists():
            d = json.loads(DEC_PRIOR.read_text(encoding="utf-8"))
            if d.get("stage119a_speed") is not None or d.get("stage119b_freeze_emb") is not None:
                log("[wait] 119 done")
                return d
        time.sleep(poll_s)
        if int(time.time() - t0) % 300 < poll_s:
            log(f"[wait] still ... {int(time.time()-t0)}s")
    raise TimeoutError("119 not ready")


def _clean_words(text: str, max_word_len: int) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return [
        w for w in text.split()
        if w and all(c in CHAR2ID and c != " " for c in w) and len(w) <= max_word_len
    ]


def build_story_episodes(
    raw_path: Path,
    *,
    max_stories: int = 8000,
    min_sents: int = 2,
    max_sents: int = 6,
    max_words_per_sent: int = 8,
    min_words: int = 3,
    max_word_len: int = 24,
    seed: int = 272,
) -> list[dict]:
    """
    Each episode: {story_id, sents: [list[str] words, ...]}
    Sentences from .!? splits; keep short windows like Stage85 filter.
    """
    raw = raw_path.read_text(encoding="utf-8", errors="ignore")
    # stories roughly separated by blank lines in our raw dump
    blocks = re.split(r"\n\s*\n+", raw)
    rng = random.Random(seed)
    rng.shuffle(blocks)
    episodes = []
    for bi, block in enumerate(blocks):
        if len(episodes) >= max_stories:
            break
        chunks = re.split(r"[.!?\n]+", block.lower())
        sents = []
        for ch in chunks:
            ws = _clean_words(ch, max_word_len)
            if len(ws) < min_words:
                continue
            # take first window of sentence (path-ish bias optional)
            win = ws[:max_words_per_sent]
            if len(win) < min_words:
                continue
            sents.append(win)
            if len(sents) >= max_sents:
                break
        if len(sents) < min_sents:
            continue
        episodes.append({"story_id": bi, "sents": sents})
    log(f"[data] episodes={len(episodes)} from {raw_path.name}")
    return episodes


def episodes_to_examples(episodes: list[dict]) -> list[dict]:
    """
    Flatten to training examples:
      words = current sentence
      history = previous sentences (list[list[str]])
      bucket = story_mem
    Plus extract path triples into fat-like copies later via build_ts_repeat_mix on phrases.
    """
    ex = []
    for ep in episodes:
        sents = ep["sents"]
        for i, sent in enumerate(sents):
            hist = sents[:i]
            phrase = " ".join(sent)
            ex.append({
                "phrase": phrase,
                "words": list(sent),
                "history": [list(h) for h in hist],
                "bucket": "story_mem",
                "story_id": ep["story_id"],
                "sent_i": i,
                "subkind": line_subkind({"words": sent}),
                "split": "train",
            })
    return ex


def _phrase_words_fit(stack, ws: list[str]) -> list[str]:
    """PhraseComposer max_len includes END slot; truncate to fit pos emb."""
    max_words = max(1, int(getattr(stack.phrase, "max_len", 5)) - 1)
    if len(ws) <= max_words:
        return ws
    return ws[-max_words:]  # keep recent words (cue/history tail)


def fill_memory(stack, cfg, device, history: list[list[str]]) -> SoftPhraseMemory:
    mem = SoftPhraseMemory(cfg.dim, cfg, device)
    for hi, ws in enumerate(history):
        if len(ws) < 1:
            continue
        ws_fit = _phrase_words_fit(stack, ws)
        key = f"h{hi}:" + " ".join(ws)
        fp = stack.phrase_fp(ws_fit)
        left = stack.w(ws_fit[0]) if ws_fit else None
        mem.observe_strict(key, fp, fact={"phrase": " ".join(ws), "words": ws}, left_fp=left)
    return mem


def memory_word_boost(mem: SoftPhraseMemory, stack, prefix_words: list[str], stoi: dict, topk: int = 3) -> dict[int, float]:
    """Return {word_id: boost} from words in top retrieved history phrases."""
    if not mem.slots or not prefix_words:
        return {}
    cue = prefix_words[-min(4, len(prefix_words)) :]
    cue = _phrase_words_fit(stack, cue)
    fp = stack.phrase_fp(cue)
    hits = mem.topk(fp, k=min(topk, len(mem.slots)))
    boost: dict[int, float] = {}
    for rank, (name, sim) in enumerate(hits):
        meta = mem.fact_meta.get(name) or {}
        ws = meta.get("words") or name.split(":")[-1].split()
        weight = float(sim) * (1.0 / (1 + rank))
        for w in ws:
            if w in stoi:
                wid = stoi[w]
                boost[wid] = max(boost.get(wid, 0.0), weight)
    return boost


def _subsample(lines, n, seed):
    if not lines or len(lines) <= n:
        return list(lines)
    rng = random.Random(seed)
    idx = list(range(len(lines)))
    rng.shuffle(idx)
    return [lines[i] for i in idx[:n]]


@torch.no_grad()
def eval_with_memory(model, stack, cfg, lines, stoi, surf, device, use_mem: bool, alpha: float = 3.0):
    """Exact@1 ALL + obj; optional phrase-memory boost."""
    from collections import defaultdict

    roles = defaultdict(lambda: {"n": 0, "h": 0})
    obj = {"n": 0, "h": 0}
    for ln in lines:
        ws = ln["words"]
        hist = ln.get("history") or []
        mem = fill_memory(stack, cfg, device, hist) if (use_mem and hist) else None
        # pairs for this single line
        ids_full = [stoi[w] for w in ws if w in stoi]
        if len(ids_full) != len(ws):
            continue
        for t in range(1, len(ws)):
            gold = ws[t]
            if gold not in stoi:
                continue
            pref_ids = ids_full[:t][-model.max_len :]
            logits = model.logits_from_prefix(pref_ids)
            if mem is not None:
                boost = memory_word_boost(mem, stack, ws[:t], stoi)
                for wid, b in boost.items():
                    logits[wid] = logits[wid] + alpha * b
            pred = surf[int(logits.argmax())]
            ok = int(pred == gold)
            ex = {"target_word": gold, "prefix_len": t}
            role = _role(ex, ws)
            for name in (role, "ALL"):
                roles[name]["n"] += 1
                roles[name]["h"] += ok
            if role == "right":
                obj["n"] += 1
                obj["h"] += ok
    def pack(d):
        return {k: {"n": v["n"], "hit1": v["h"] / max(v["n"], 1)} for k, v in d.items()}
    return {
        "roles": pack(roles),
        "obj": {"n": obj["n"], "hit1": obj["h"] / max(obj["n"], 1)},
    }


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"Stage120 phrase-mem channel {datetime.now(timezone.utc).isoformat()}")
    try:
        wait_prior()
        cfg = Config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        parent85 = CKPT / "stage85_foundation_FROZEN.pt"
        if not parent85.exists():
            parent85 = CKPT / "stage85_foundation.pt"
        stack, _ = load_foundation_85(device, cfg, path=parent85)

        raw = RAW if RAW.exists() else RAW_FALLBACK
        assert raw.exists(), f"missing {raw}"
        episodes = build_story_episodes(raw, max_word_len=int(cfg.max_word_len))
        examples = episodes_to_examples(episodes)
        # split stories
        story_ids = sorted({e["story_id"] for e in examples})
        rng = random.Random(272)
        rng.shuffle(story_ids)
        n_hold = max(1, int(0.15 * len(story_ids)))
        hold_ids = set(story_ids[:n_hold])
        train_ex = [e for e in examples if e["story_id"] not in hold_ids]
        hold_ex = [e for e in examples if e["story_id"] in hold_ids]
        # prefer hold lines that HAVE history for mem gain measurement
        hold_with_hist = [e for e in hold_ex if e.get("history")]
        hold_eval = hold_with_hist if len(hold_with_hist) >= 200 else hold_ex
        hold_eval = _subsample(hold_eval, min(800, len(hold_eval)), 9)
        log(f"train_ex={len(train_ex)} hold_ex={len(hold_ex)} hold_eval={len(hold_eval)} "
            f"with_hist={sum(1 for e in hold_eval if e.get('history'))}")

        # also keep a classic SEEN fat mix from phrases for path lock
        phrases = [e["phrase"] for e in train_ex]
        cfg.c87_n_fat = 200
        cfg.c87_n_rare = 80
        cfg.c87_fat_copies = 40
        cfg.c87_rare_copies = 2
        cfg.c87_seen_hold_frac = 0.20
        cfg.c87_story_keep_frac = 0.35
        cfg.c87_hold_frac = 0.15
        mix_train, hold_seen, hold_rare, hold_story_classic, meta = build_ts_repeat_mix(phrases, cfg, seed=272)
        for ln in mix_train + hold_seen + hold_rare + hold_story_classic:
            ln.setdefault("subkind", line_subkind(ln))
            ln["split"] = ln.get("bucket", "x")
            ln.setdefault("history", [])

        # vocab
        all_lines = train_ex + hold_ex + mix_train + hold_seen + hold_rare + hold_story_classic
        words = sorted({w for ln in all_lines for w in ln["words"]})
        fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)
        stoi = {s: i for i, s in enumerate(words)}
        surf = words

        # pairs
        fat_lines = [ln for ln in mix_train if ln.get("bucket") == "fat_train"]
        story_mem_lines = train_ex
        fat_pairs = lines_to_pairs(fat_lines, stoi) if fat_lines else []
        # attach history pointer via line_i into story_mem_lines
        story_pairs = lines_to_pairs(story_mem_lines, stoi)
        for ex in story_pairs:
            ex["history"] = story_mem_lines[ex["line_i"]].get("history") or []
            ex["words_full"] = story_mem_lines[ex["line_i"]]["words"]
        for ex in fat_pairs:
            ex["history"] = []
            ex["words_full"] = fat_lines[ex["line_i"]]["words"]
        if not fat_pairs:
            fat_pairs = story_pairs

        model = WordIdTransformer(len(surf), 256, 4, 2, 16, 0.1).to(device)
        # warm start if vocab matches Stage100 — unlikely; fp init
        if ATOM100.exists():
            ck = torch.load(ATOM100, map_location="cpu", weights_only=False)
            if list(ck.get("surfaces", [])) == surf:
                model.load_state_dict(ck["word_tf"], strict=True)
                log("[init] Stage100")
            else:
                model.init_from_fps(fps)
                log("[init] fp")
        else:
            model.init_from_fps(fps)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        alpha = 3.0
        fat_frac = 0.5  # less fat — more room for story_mem channel
        batch = 8
        ft_steps = 40000
        eval_every = 2000
        rr = random.Random(272)

        def sample_batch():
            n_fat = max(1, int(round(batch * fat_frac)))
            n_fat = min(n_fat, batch)
            ex = [rr.choice(fat_pairs) for _ in range(n_fat)]
            ex += [rr.choice(story_pairs) for _ in range(batch - n_fat)]
            rr.shuffle(ex)
            return ex

        def loss_batch(exs):
            # collate without memory first
            packed = collate_word_id_batch(exs, stoi, model.max_len, model.pad_id, device)
            if packed is None:
                return None
            ids, pad_mask, tgt = packed
            logits = model.logits_last_from_batch(ids, pad_mask)
            # align exs that survived — rebuild valid list
            valid = []
            for ex in exs:
                if ex["target_word"] not in stoi or not ex["prefix_word_ids"]:
                    continue
                valid.append(ex)
            valid = valid[: tgt.shape[0]]
            # add memory boost on rows with history
            for i, ex in enumerate(valid):
                hist = ex.get("history") or []
                if not hist:
                    continue
                mem = fill_memory(stack, cfg, device, hist)
                pref = (ex.get("words_full") or [])[: ex["prefix_len"]]
                boost = memory_word_boost(mem, stack, pref, stoi)
                for wid, b in boost.items():
                    logits[i, wid] = logits[i, wid] + alpha * b
            return F.cross_entropy(logits, tgt)

        best = {
            "step": 0, "mem_all": 0.0, "local_all": 0.0, "obj": 0.0, "rel": 0.0,
            "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        }
        curve = []

        def snap(step):
            model.eval()
            local = eval_with_memory(model, stack, cfg, hold_eval, stoi, surf, device, use_mem=False, alpha=alpha)
            memed = eval_with_memory(model, stack, cfg, hold_eval, stoi, surf, device, use_mem=True, alpha=alpha)
            # classic SEEN for lock
            seen = eval_id_capacity_suite(model, _subsample(hold_seen, 600, 1), mix_train, surf, stoi, device)
            loc_all = local["roles"].get("ALL", {}).get("hit1", 0.0)
            mem_all = memed["roles"].get("ALL", {}).get("hit1", 0.0)
            obj = seen["obj"]["hit1"]
            rel = seen["roles"].get("rel", {}).get("hit1", 0.0)
            gain = mem_all - loc_all
            curve.append({
                "step": step, "local_all": loc_all, "mem_all": mem_all, "gain_pp": gain,
                "obj": obj, "rel": rel,
            })
            key = (1 if rel >= 0.50 else 0, mem_all, gain, obj)
            bkey = (1 if best["rel"] >= 0.50 else 0, best["mem_all"], best["mem_all"] - best["local_all"], best["obj"])
            if key >= bkey:
                best.update(
                    step=step, mem_all=mem_all, local_all=loc_all, obj=obj, rel=rel,
                    state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                )
            log(
                f"  step {step:5d}: STORY_LOCAL={loc_all*100:.1f}% STORY_MEM={mem_all*100:.1f}% "
                f"(gain {gain*100:+.1f}pp) | SEEN obj={obj*100:.1f}% rel={rel*100:.1f}%"
            )
            model.train()

        log(f"=== FT dual-channel alpha={alpha} fat_frac={fat_frac} steps={ft_steps} ===")
        snap(0)
        model.train()
        for step in range(1, ft_steps + 1):
            for g in opt.param_groups:
                g["lr"] = _warmup_then_constant(step, 1e-3, 200)
            loss = loss_batch(sample_batch())
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % eval_every == 0 or step == ft_steps:
                snap(step)

        model.load_state_dict(best["state"])
        model.eval()
        local = eval_with_memory(model, stack, cfg, hold_eval, stoi, surf, device, False, alpha)
        memed = eval_with_memory(model, stack, cfg, hold_eval, stoi, surf, device, True, alpha)
        seen = eval_id_capacity_suite(model, _subsample(hold_seen, 1200, 2), mix_train, surf, stoi, device)
        loc_all = local["roles"].get("ALL", {}).get("hit1", 0.0)
        mem_all = memed["roles"].get("ALL", {}).get("hit1", 0.0)
        gain = mem_all - loc_all
        obj, rel = seen["obj"]["hit1"], seen["roles"].get("rel", {}).get("hit1", 0.0)

        if rel >= 0.50 and gain >= 0.03:
            verdict = "PASS_MEM"
        elif rel >= 0.50 and gain >= 0.015:
            verdict = "PARTIAL_MEM"
        elif rel >= 0.50 and abs(gain) < 0.01:
            verdict = "NULL_MEM"  # memory channel didn't help
        elif rel < 0.50:
            verdict = "FAIL_REL"
        else:
            verdict = "PARITY"

        report = "\n".join([
            "SOTE Stage 120 — phrase-memory second channel",
            f"timestamp: {datetime.now(timezone.utc).isoformat()}",
            f"episodes train/hold, alpha={alpha}, fat_frac={fat_frac}",
            f"STORY_LOCAL ALL={loc_all*100:.1f}%",
            f"STORY_MEM   ALL={mem_all*100:.1f}% (gain {gain*100:+.1f}pp)",
            f"SEEN obj={obj*100:.1f}% rel={rel*100:.1f}%",
            f"vs Stage100 STORY {REF100['story_all']*100:.1f}%",
            f"Verdict: {verdict}",
            "Channel B = SoftPhraseMemory over previous sentences; boost words from retrieve.",
        ]) + "\n"
        log("\n" + report)
        (RES / "stage120_phrase_mem_report.txt").write_text(report, encoding="utf-8")
        (RES / f"stage120_phrase_mem_{verdict}.txt").write_text(report, encoding="utf-8")
        ckpt = CKPT / "stage120_phrase_mem.pt"
        torch.save({
            "stage": 120, "verdict": verdict, "word_tf": best["state"],
            "surfaces": surf, "alpha": alpha, "cfg": asdict(cfg),
        }, ckpt)
        out = {
            "stage": 120, "verdict": verdict, "alpha": alpha,
            "story_local_all": loc_all, "story_mem_all": mem_all, "gain_pp": gain,
            "seen": {"obj": obj, "rel": rel}, "curve": curve, "ckpt": str(ckpt),
            "best_step": best["step"], "n_episodes": len(episodes),
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        (RES / "stage120_phrase_mem_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stage 120 (phrase-mem channel):** STORY_LOCAL "
                f"{loc_all*100:.1f}% → STORY_MEM {mem_all*100:.1f}% "
                f"(gain {gain*100:+.1f}pp); SEEN obj {obj*100:.1f}%. "
                f"{verdict}. `stage120_decision.json`.\n"
            )
            if "Stage 120 (phrase-mem" not in txt:
                txt = txt.replace(
                    "**F85 dual-channel FREEZE:**",
                    block + "\n**F85 dual-channel FREEZE:**",
                )
                replay.write_text(txt, encoding="utf-8")
        log(f"Saved {ckpt}")
        log("DONE 120")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
