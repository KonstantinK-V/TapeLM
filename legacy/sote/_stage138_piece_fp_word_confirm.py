"""
Stage 138 — train on SOTE letter-fp PIECES; confirm with WORDS (exact@1).

After 137. Distinct from G' (126): there CE targets were surface-only while
pieces were input. Here PRIMARY CE = next-piece on the full piece stream;
word exact@1 is the confirmation / gate metric (argmax restricted to word rows).

Piece expand (per word):
  [$stem, +affix?, surface]  — stem/affix from crude_stem + AFFIXES;
  piece emb init from F85 letter→fp (stem body / affix chars / surface).

Arms (TinyStories 100k, hops OUT, d=256, 2L, 40k steps):
  A word_ce     — plain next-word CE (control)
  B piece_ce    — next-piece CE; word confirm @1 on hold

Gate vs A on STORY word-ALL:
  PASS    +3pp with SEEN word-obj within -2pp and rel>=0.70
  PARTIAL +1.5pp
  else PARITY / HARM

Run (waits for 137):
  python _stage138_piece_fp_word_confirm.py
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
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    AFFIXES,
    CKPT,
    RES,
    Config,
    _role,
    _warmup_then_constant,
    build_ts_repeat_mix,
    line_subkind,
    load_foundation_85,
    surface,
)
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402
from _stage116_117_hops_morph_follow import crude_stem  # noqa: E402
from _stage137_dynamic_bigram_input import WordOutTransformer  # noqa: E402

DEC137 = RES / "stage137_dynamic_bigram_input_decision.json"
LOG = RES / "_stage138_log.txt"
DEC = RES / "stage138_piece_fp_word_confirm_decision.json"

FT_STEPS = 40_000
EVAL_EVERY = 2000
BATCH = 8
FAT_FRAC = 0.75
MAX_LEN = 32


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def wait_137(timeout_s=16 * 3600, poll=60):
    log(f"[wait] for {DEC137.name} ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC137.exists():
            d = json.loads(DEC137.read_text(encoding="utf-8"))
            if d.get("verdict"):
                log(f"[wait] 137 done verdict={d.get('verdict')}")
                return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still ... {int(time.time()-t0)}s")
    raise TimeoutError("137 not ready")


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


def build_piece_vocab(words: list[str]):
    """surfaces first (0..n_word-1), then $stems, then +affixes."""
    stem_toks = set()
    for w in words:
        st = crude_stem(w) or w
        stem_toks.add(f"${st}")
    stems = sorted(stem_toks)[:8000]
    affix_toks = [f"+{a}" for a in AFFIXES]
    pieces = list(words) + stems + affix_toks
    stoi = {p: i for i, p in enumerate(pieces)}
    return pieces, stoi, len(words)


def expand_words(ws, stoi):
    ids = []
    for w in ws:
        st = crude_stem(w) or w
        st_tok = f"${st}"
        if st_tok in stoi:
            ids.append(stoi[st_tok])
        aff = affix_of(w, st)
        if aff is not None:
            at = f"+{aff}"
            if at in stoi:
                ids.append(stoi[at])
        if w in stoi:
            ids.append(stoi[w])
    return ids


def piece_fps(stack, pieces, n_word, dim, device):
    rows = []
    for i, p in enumerate(pieces):
        try:
            if p.startswith("$"):
                rows.append(stack.w(p[1:]).detach())
            elif p.startswith("+"):
                # affix characters as fp if possible, else small vec
                body = p[1:]
                rows.append(stack.w(body).detach() if body else torch.zeros(dim))
            else:
                rows.append(stack.w(p).detach())
        except Exception:
            rows.append(torch.zeros(dim, device=device))
    return F.normalize(torch.stack(rows, 0), dim=-1).to(device)


def collate(prefs, targets, pad_id, device, max_len):
    if not prefs:
        return None
    prefs = [p[-max_len:] for p in prefs]
    tmax = max(len(p) for p in prefs)
    bsz = len(prefs)
    ids = torch.full((bsz, tmax), pad_id, dtype=torch.long, device=device)
    mask = torch.ones((bsz, tmax), dtype=torch.bool, device=device)
    for i, p in enumerate(prefs):
        ids[i, -len(p) :] = torch.tensor(p, dtype=torch.long, device=device)
        mask[i, -len(p) :] = False
    tgt = torch.tensor(targets, dtype=torch.long, device=device)
    return ids, mask, tgt


@torch.no_grad()
def eval_word_confirm(model, hold, stoi, n_word, words, mode: str):
    """
    mode=word: prefix = word ids; full softmax over words.
    mode=piece: prefix = expand(words); argmax restricted to word rows (confirm).
    """
    roles = defaultdict(lambda: {"n": 0, "h": 0})
    obj = {"n": 0, "h": 0}
    for ln in hold:
        ws = ln["words"]
        for t in range(1, len(ws)):
            gold = ws[t]
            if gold not in stoi:
                continue
            if mode == "word":
                ids = [stoi[w] for w in ws[:t] if w in stoi][-model.max_len :]
                if not ids:
                    continue
                logits = model.logits_from_prefix(ids)
                pred = int(logits[:n_word].argmax())
            else:
                ids = expand_words(ws[:t], stoi)[-model.max_len :]
                if not ids:
                    continue
                logits = model.logits_from_prefix(ids)
                pred = int(logits[:n_word].argmax())  # word confirm only
            ok = int(pred == stoi[gold])
            fake = {"target_word": gold, "prefix_len": t}
            role = _role(fake, ws)
            for name in (role, "ALL"):
                roles[name]["n"] += 1
                roles[name]["h"] += ok
            if role == "right":
                obj["n"] += 1
                obj["h"] += ok
    pack = {k: {"n": v["n"], "hit1": v["h"] / max(v["n"], 1)} for k, v in roles.items()}
    return {"roles": pack, "obj": {"n": obj["n"], "hit1": obj["h"] / max(obj["n"], 1)}}


@torch.no_grad()
def eval_piece_hit(model, hold, stoi, mode_piece: bool):
    """Diag: teacher-forced next-piece exact (full piece vocab)."""
    if not mode_piece:
        return None
    n = h = 0
    for ln in hold:
        ids = expand_words(ln["words"], stoi)
        for i in range(1, len(ids)):
            pref = ids[:i][-model.max_len :]
            gold = ids[i]
            pred = model.pred_id(pref)
            n += 1
            h += int(pred == gold)
    return {"n": n, "hit1": h / max(n, 1)}


def train_arm(
    *,
    tag: str,
    mode: str,  # word | piece
    pieces,
    stoi,
    n_word,
    fps,
    train,
    hold_seen,
    hold_story,
    device,
):
    n_emb = len(pieces) if mode == "piece" else n_word
    # word arm uses only first n_word rows / vocab
    model = WordOutTransformer(
        n_word=n_word if mode == "word" else n_word,
        n_emb=n_emb if mode == "piece" else n_word,
        d_model=256,
        n_heads=4,
        n_layers=2,
        max_len=MAX_LEN if mode == "piece" else 16,
        dropout=0.1,
    ).to(device)
    if mode == "word":
        model.init_word_rows(fps[:n_word])
    else:
        # all piece rows from fps; word confirm uses first n_word
        with torch.no_grad():
            d = min(model.d_model, fps.size(1))
            model.tok.weight[:n_emb, :d] = fps[:n_emb, :d]
            model.tok.weight[:n_emb] = F.normalize(model.tok.weight[:n_emb], dim=-1)
        # WordOut forward uses weight[:n_word] for logits — WRONG for piece CE.
        # Need logits over full piece vocab for training.
        # Patch: store n_piece and override forward for piece arm.

    if mode == "piece":
        # replace output projection size by wrapping
        class PieceLM(nn.Module):
            def __init__(self, base, n_piece):
                super().__init__()
                self.base = base
                self.n_piece = n_piece
                self.n_word = base.n_word
                self.pad_id = base.pad_id
                self.max_len = base.max_len
                self.tok = base.tok

            def forward(self, ids, key_padding_mask=None):
                bsz, tmax = ids.shape
                device = ids.device
                pos = torch.arange(tmax, device=device).unsqueeze(0).expand(bsz, -1)
                x = self.base.tok(ids) + self.base.pos(pos)
                mask = torch.triu(torch.ones(tmax, tmax, device=device, dtype=torch.bool), diagonal=1)
                h = self.base.tr(x, mask=mask, src_key_padding_mask=key_padding_mask)
                return F.linear(h, self.base.tok.weight[: self.n_piece])

            def logits_last_from_batch(self, ids, key_padding_mask=None):
                return self.forward(ids, key_padding_mask)[:, -1, :]

            def logits_from_prefix(self, id_list):
                ids = torch.tensor(
                    [id_list[-self.max_len :]], dtype=torch.long, device=self.tok.weight.device
                )
                return self.forward(ids)[0, -1]

            def pred_id(self, id_list):
                return int(self.logits_from_prefix(id_list).argmax())

            def word_logits_from_prefix(self, id_list):
                return self.logits_from_prefix(id_list)[: self.n_word]

        model = PieceLM(model, n_emb)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"\n======== 138 {tag} mode={mode} n_emb={n_emb} n_word={n_word} params={n_params/1e6:.2f}M ========")

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    story = _subsample(story, 50000, 18) if len(story) > 50000 else story

    # build example pools
    def build_pool(lines):
        ex = []
        for ln in lines:
            ws = ln["words"]
            if mode == "word":
                ids = [stoi[w] for w in ws if w in stoi]
                for i in range(1, len(ids)):
                    ex.append((ids[:i], ids[i]))
            else:
                ids = expand_words(ws, stoi)
                for i in range(1, len(ids)):
                    ex.append((ids[:i], ids[i]))
        return ex

    fat_ex = build_pool(fat)
    story_ex = build_pool(story)
    if not fat_ex:
        fat_ex = story_ex
    log(f"  pool fat_ex={len(fat_ex)} story_ex={len(story_ex)}")

    ev_seen = _subsample(hold_seen, 500, 1381)
    ev_story = _subsample(hold_story, 350, 1383)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 1000), 2381)
    fin_story = _subsample(hold_story, min(len(hold_story), 700), 2383)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rr = random.Random(138)
    best = {
        "story_all": -1.0,
        "obj": -1.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []

    def word_eval(hold):
        if mode == "word":
            return eval_word_confirm(model, hold, stoi, n_word, pieces[:n_word], "word")
        # piece model: use word_logits
        roles = defaultdict(lambda: {"n": 0, "h": 0})
        obj = {"n": 0, "h": 0}
        for ln in hold:
            ws = ln["words"]
            for t in range(1, len(ws)):
                gold = ws[t]
                if gold not in stoi:
                    continue
                ids = expand_words(ws[:t], stoi)[-model.max_len :]
                if not ids:
                    continue
                pred = int(model.word_logits_from_prefix(ids).argmax())
                ok = int(pred == stoi[gold])
                fake = {"target_word": gold, "prefix_len": t}
                role = _role(fake, ws)
                for name in (role, "ALL"):
                    roles[name]["n"] += 1
                    roles[name]["h"] += ok
                if role == "right":
                    obj["n"] += 1
                    obj["h"] += ok
        pack = {k: {"n": v["n"], "hit1": v["h"] / max(v["n"], 1)} for k, v in roles.items()}
        return {"roles": pack, "obj": {"n": obj["n"], "hit1": obj["h"] / max(obj["n"], 1)}}

    def snap(step):
        model.eval()
        s = word_eval(ev_seen)
        st = word_eval(ev_story)
        obj = s["obj"]["hit1"]
        rel = s["roles"].get("rel", {}).get("hit1", 0.0)
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        ph = eval_piece_hit(model, ev_story[:80], stoi, mode == "piece")
        curve.append({"step": step, "obj": obj, "rel": rel, "story_all": sall, "piece_hit": ph})
        extra = f" piece@1={100*ph['hit1']:.1f}%" if ph else ""
        log(
            f"  [{tag}] step {step}: SEEN obj={100*obj:.1f}% rel={100*rel:.1f}% | "
            f"STORY wordALL={100*sall:.1f}%{extra}"
        )
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
            g["lr"] = _warmup_then_constant(step, 1e-3, 200)
        n_fat = max(1, int(round(BATCH * FAT_FRAC)))
        batch = [rr.choice(fat_ex) for _ in range(n_fat)]
        batch += [rr.choice(story_ex) for _ in range(BATCH - n_fat)]
        prefs = [b[0] for b in batch]
        tgts = [b[1] for b in batch]
        packed = collate(prefs, tgts, model.pad_id, device, model.max_len)
        if packed is None:
            continue
        ids, mask, tgt = packed
        logits = model.logits_last_from_batch(ids, mask)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0 or step == FT_STEPS:
            snap(step)

    model.load_state_dict(best["state"])
    model.eval()
    f_seen = word_eval(fin_seen)
    f_story = word_eval(fin_story)
    piece_diag = eval_piece_hit(model, fin_story[:120], stoi, mode == "piece")
    ck = CKPT / f"stage138_{tag}.pt"
    torch.save({"model": best["state"], "mode": mode, "pieces": pieces, "n_word": n_word}, ck)
    return {
        "tag": tag,
        "mode": mode,
        "params": n_params,
        "n_emb": n_emb,
        "seen_obj": f_seen["obj"]["hit1"],
        "seen_rel": f_seen["roles"].get("rel", {}).get("hit1", 0.0),
        "seen_all": f_seen["roles"].get("ALL", {}).get("hit1", 0.0),
        "story_all": f_story["roles"].get("ALL", {}).get("hit1", 0.0),
        "story_obj": f_story["obj"]["hit1"],
        "piece_hit1_diag": piece_diag,
        "curve": curve,
        "ckpt": str(ck),
    }


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"138 piece-fp CE + word confirm start {datetime.now(timezone.utc).isoformat()}")
    log("Train=next-piece (letter fps). Confirm/gate=word exact@1. Hops OUT.")
    try:
        upstream = wait_137()
        phrases = ensure_100k()
        cfg = Config()
        cfg.c87_n_fat, cfg.c87_n_rare = 200, 80
        cfg.c87_fat_copies, cfg.c87_rare_copies = 40, 2
        cfg.c87_seen_hold_frac, cfg.c87_story_keep_frac, cfg.c87_hold_frac = 0.2, 0.5, 0.15
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        parent = CKPT / "stage85_foundation_FROZEN.pt"
        if not parent.exists():
            parent = CKPT / "stage85_foundation.pt"
        stack, _ = load_foundation_85(device, cfg, path=parent)
        for mod in (stack.encoder, stack.composer, stack.morph, stack.phrase, stack.binder):
            for p in mod.parameters():
                p.requires_grad_(False)
            mod.eval()

        train, hold_seen, hold_rare, hold_story, meta = build_ts_repeat_mix(phrases, cfg, seed=272)
        for ln in train + hold_seen + hold_rare + hold_story:
            ln.setdefault("subkind", line_subkind(ln))
            ln["split"] = ln.get("bucket", "x")
        words = sorted({w for ln in train + hold_seen + hold_rare + hold_story for w in ln["words"]})
        pieces, stoi, n_word = build_piece_vocab(words)
        fps = piece_fps(stack, pieces, n_word, cfg.dim, device)
        log(f"[init] n_word={n_word} n_piece={len(pieces)} (stems+affix+surf)")

        word = train_arm(
            tag="word_ce",
            mode="word",
            pieces=pieces,
            stoi=stoi,
            n_word=n_word,
            fps=fps,
            train=train,
            hold_seen=hold_seen,
            hold_story=hold_story,
            device=device,
        )
        piece = train_arm(
            tag="piece_ce_word_confirm",
            mode="piece",
            pieces=pieces,
            stoi=stoi,
            n_word=n_word,
            fps=fps,
            train=train,
            hold_seen=hold_seen,
            hold_story=hold_story,
            device=device,
        )

        lift = piece["story_all"] - word["story_all"]
        obj_d = piece["seen_obj"] - word["seen_obj"]
        rel_ok = piece["seen_rel"] >= 0.70 and word["seen_rel"] >= 0.70
        if obj_d < -0.03:
            verdict = "HARM"
        elif rel_ok and lift >= 0.03 and obj_d >= -0.02:
            verdict = "PASS"
        elif rel_ok and lift >= 0.015:
            verdict = "PARTIAL"
        else:
            verdict = "PARITY"

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "piece_fp_ce_word_confirm",
            "upstream_137": upstream.get("verdict"),
            "expand": "[$stem, +affix?, surface]; CE=next-piece; gate=word argmax",
            "n_word": n_word,
            "n_piece": len(pieces),
            "word": word,
            "piece": piece,
            "story_lift_pp": lift,
            "seen_obj_delta_pp": obj_d,
            "ref100": REF100,
            "verdict": verdict,
            "note": (
                "Primary train signal = piece stream with SOTE letter fps. "
                "Word exact@1 is confirmation metric (logits masked to word rows)."
            ),
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        (RES / f"stage138_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
        log(
            f"[138] {verdict} word_STORY={100*word['story_all']:.1f}% "
            f"piece_wordSTORY={100*piece['story_all']:.1f}% lift={100*lift:+.1f}pp"
        )

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stage 138 piece-fp CE / word confirm:** {verdict} "
                f"word={100*word['story_all']:.1f}% piece={100*piece['story_all']:.1f}% "
                f"lift={100*lift:+.1f}pp. `stage138_piece_fp_word_confirm_decision.json`.\n"
            )
            if "Stage 138 piece-fp" not in txt:
                if "**F85 dual-channel FREEZE:**" in txt:
                    txt = txt.replace(
                        "**F85 dual-channel FREEZE:**",
                        block + "\n**F85 dual-channel FREEZE:**",
                    )
                else:
                    txt = txt.rstrip() + "\n" + block
                replay.write_text(txt, encoding="utf-8")
        log("DONE 138")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
