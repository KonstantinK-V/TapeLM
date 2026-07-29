"""
Stage 139 — BPE-like density ONLY on tails; keep whole words otherwise.

Idea (user): use BPE *principle* (extra tokens where language splits), but
keep the word as the primary atom; only cut morphological *tails*
(+ing/+ed/+s) when surface != stem. No full stem+surface expand on every word
(that was 138 / G').

Expand rule per word:
  if affixable tail (surface(stem,a)==word):  [$stem, +aff, surface]
  else:                                       [surface]   # whole word only

Train: next-piece CE on this mixed stream (letter-fp init).
Confirm/gate: word exact@1 (argmax over word rows only).

Arms (TS 100k, hops OUT, d=256 2L, 40k):
  A word_ce     — plain next-word (control; may warm-share setup with 138)
  B tail_cut    — expand only where tail cuts

Vs 138: 138 always emits $stem(+aff)+surface; 139 mostly single surface.

Waits for Stage138.

Run:
  python _stage139_bpe_tail_cut_words.py
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

DEC138 = RES / "stage138_piece_fp_word_confirm_decision.json"
LOG = RES / "_stage139_log.txt"
DEC = RES / "stage139_bpe_tail_cut_words_decision.json"

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


def wait_138(timeout_s=18 * 3600, poll=60):
    log(f"[wait] for {DEC138.name} ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC138.exists():
            d = json.loads(DEC138.read_text(encoding="utf-8"))
            if d.get("verdict"):
                log(f"[wait] 138 done verdict={d.get('verdict')}")
                return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still ... {int(time.time()-t0)}s")
    raise TimeoutError("138 not ready")


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
    stem_toks = set()
    n_cut = 0
    for w in words:
        st = crude_stem(w) or w
        if affix_of(w, st) is not None:
            stem_toks.add(f"${st}")
            n_cut += 1
    stems = sorted(stem_toks)[:8000]
    affix_toks = [f"+{a}" for a in AFFIXES]
    pieces = list(words) + stems + affix_toks
    stoi = {p: i for i, p in enumerate(pieces)}
    return pieces, stoi, len(words), n_cut


def expand_tail_cut(ws, stoi):
    """Whole word by default; [$stem,+aff,surface] only when tail cuts."""
    ids = []
    for w in ws:
        if w not in stoi:
            continue
        st = crude_stem(w) or w
        aff = affix_of(w, st)
        if aff is not None:
            st_tok = f"${st}"
            if st_tok in stoi:
                ids.append(stoi[st_tok])
            at = f"+{aff}"
            if at in stoi:
                ids.append(stoi[at])
        ids.append(stoi[w])
    return ids


def expand_always_morph(ws, stoi):
    """138-style (diag only / optional): always $stem(+aff)+surface."""
    ids = []
    for w in ws:
        if w not in stoi:
            continue
        st = crude_stem(w) or w
        st_tok = f"${st}"
        if st_tok in stoi:
            ids.append(stoi[st_tok])
        aff = affix_of(w, st)
        if aff is not None:
            at = f"+{aff}"
            if at in stoi:
                ids.append(stoi[at])
        ids.append(stoi[w])
    return ids


def piece_fps(stack, pieces, dim, device):
    rows = []
    for p in pieces:
        try:
            if p.startswith("$"):
                rows.append(stack.w(p[1:]).detach())
            elif p.startswith("+"):
                rows.append(stack.w(p[1:]).detach())
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


class PieceLM(nn.Module):
    """Full-piece logits; word confirm via [:n_word]."""

    def __init__(self, base: WordOutTransformer, n_piece: int):
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
        ids = torch.tensor([id_list[-self.max_len :]], dtype=torch.long, device=self.tok.weight.device)
        return self.forward(ids)[0, -1]

    def word_logits_from_prefix(self, id_list):
        return self.logits_from_prefix(id_list)[: self.n_word]

    def pred_id(self, id_list):
        return int(self.logits_from_prefix(id_list).argmax())


def density_report(train_lines, stoi, n_sample=2000):
    """Mean pieces/word under tail-cut vs always-morph vs plain."""
    rr = random.Random(139)
    lines = train_lines if len(train_lines) <= n_sample else [train_lines[i] for i in rr.sample(range(len(train_lines)), n_sample)]
    nw = n_plain = n_tail = n_full = 0
    for ln in lines:
        ws = ln["words"]
        nw += len(ws)
        n_plain += len(ws)
        n_tail += len(expand_tail_cut(ws, stoi))
        # always-morph needs stem toks in stoi — may miss; count structurally
        n_full += sum(3 if affix_of(w, crude_stem(w) or w) else 2 for w in ws)
    return {
        "mean_plain": n_plain / max(nw, 1),
        "mean_tail_cut": n_tail / max(nw, 1),
        "mean_always_morph_est": n_full / max(nw, 1),
        "n_words": nw,
    }


def train_arm(
    *,
    tag: str,
    mode: str,  # word | tail_cut
    pieces,
    stoi,
    n_word,
    fps,
    train,
    hold_seen,
    hold_story,
    device,
):
    if mode == "word":
        model = WordOutTransformer(n_word, n_word, 256, 4, 2, 16, 0.1).to(device)
        model.init_word_rows(fps[:n_word])
        expand = None
        n_emb = n_word
    else:
        n_emb = len(pieces)
        base = WordOutTransformer(n_word, n_emb, 256, 4, 2, MAX_LEN, 0.1).to(device)
        with torch.no_grad():
            d = min(base.d_model, fps.size(1))
            base.tok.weight[:n_emb, :d] = fps[:n_emb, :d]
            base.tok.weight[:n_emb] = F.normalize(base.tok.weight[:n_emb], dim=-1)
        model = PieceLM(base, n_emb)
        expand = expand_tail_cut

    n_params = sum(p.numel() for p in model.parameters())
    log(f"\n======== 139 {tag} mode={mode} n_emb={n_emb} params={n_params/1e6:.2f}M ========")

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    story = _subsample(story, 50000, 19) if len(story) > 50000 else story

    def build_pool(lines):
        ex = []
        for ln in lines:
            ws = ln["words"]
            if mode == "word":
                ids = [stoi[w] for w in ws if w in stoi]
            else:
                ids = expand(ws, stoi)
            for i in range(1, len(ids)):
                ex.append((ids[:i], ids[i]))
        return ex

    fat_ex = build_pool(fat)
    story_ex = build_pool(story)
    if not fat_ex:
        fat_ex = story_ex
    log(f"  pool fat={len(fat_ex)} story={len(story_ex)}")

    ev_seen = _subsample(hold_seen, 500, 1391)
    ev_story = _subsample(hold_story, 350, 1393)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 1000), 2391)
    fin_story = _subsample(hold_story, min(len(hold_story), 700), 2393)

    def word_eval(hold):
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
                    pred = int(model.logits_from_prefix(ids)[:n_word].argmax())
                else:
                    ids = expand_tail_cut(ws[:t], stoi)[-model.max_len :]
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

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rr = random.Random(139)
    best = {
        "story_all": -1.0,
        "obj": -1.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []

    def snap(step):
        model.eval()
        s, st = word_eval(ev_seen), word_eval(ev_story)
        obj, rel = s["obj"]["hit1"], s["roles"].get("rel", {}).get("hit1", 0.0)
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        curve.append({"step": step, "obj": obj, "rel": rel, "story_all": sall})
        log(
            f"  [{tag}] step {step}: SEEN obj={100*obj:.1f}% rel={100*rel:.1f}% | "
            f"STORY wordALL={100*sall:.1f}%"
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
        packed = collate([b[0] for b in batch], [b[1] for b in batch], model.pad_id, device, model.max_len)
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
    f_seen, f_story = word_eval(fin_seen), word_eval(fin_story)
    ck = CKPT / f"stage139_{tag}.pt"
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
        "curve": curve,
        "ckpt": str(ck),
    }


def main():
    RES.mkdir(exist_ok=True)
    if LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log(f"139 BPE-tail-cut words start {datetime.now(timezone.utc).isoformat()}")
    log("Keep whole words; cut only morph tails (+ing/+ed/+s). Word confirm gate.")
    try:
        upstream = wait_138()
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

        train, hold_seen, hold_rare, hold_story, _ = build_ts_repeat_mix(phrases, cfg, seed=272)
        for ln in train + hold_seen + hold_rare + hold_story:
            ln.setdefault("subkind", line_subkind(ln))
            ln["split"] = ln.get("bucket", "x")
        words = sorted({w for ln in train + hold_seen + hold_rare + hold_story for w in ln["words"]})
        pieces, stoi, n_word, n_cut_types = build_piece_vocab(words)
        fps = piece_fps(stack, pieces, cfg.dim, device)
        dens = density_report(train, stoi)
        log(
            f"[init] n_word={n_word} n_piece={len(pieces)} cuttable_types~{n_cut_types} | "
            f"density plain={dens['mean_plain']:.3f} tail={dens['mean_tail_cut']:.3f} "
            f"always~{dens['mean_always_morph_est']:.3f}"
        )

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
        tail = train_arm(
            tag="tail_cut",
            mode="tail_cut",
            pieces=pieces,
            stoi=stoi,
            n_word=n_word,
            fps=fps,
            train=train,
            hold_seen=hold_seen,
            hold_story=hold_story,
            device=device,
        )

        lift = tail["story_all"] - word["story_all"]
        obj_d = tail["seen_obj"] - word["seen_obj"]
        rel_ok = tail["seen_rel"] >= 0.70 and word["seen_rel"] >= 0.70
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
            "protocol": "bpe_principle_tail_cut_keep_words",
            "upstream_138": upstream.get("verdict"),
            "expand": "surface only; OR [$stem,+aff,surface] iff morph tail",
            "density": dens,
            "n_word": n_word,
            "n_piece": len(pieces),
            "word": word,
            "tail_cut": tail,
            "story_lift_pp": lift,
            "seen_obj_delta_pp": obj_d,
            "ref100": REF100,
            "verdict": verdict,
            "note": (
                "BPE-like extra tokens only on morph tails; head/simple words stay one atom. "
                "Aligns with C' (head~1.0 tpw). Distinct from 138 always-morph piece stream."
            ),
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        (RES / f"stage139_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
        log(
            f"[139] {verdict} word_STORY={100*word['story_all']:.1f}% "
            f"tail_STORY={100*tail['story_all']:.1f}% lift={100*lift:+.1f}pp"
        )

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stage 139 BPE-tail-cut (keep words):** {verdict} "
                f"word={100*word['story_all']:.1f}% tail={100*tail['story_all']:.1f}% "
                f"lift={100*lift:+.1f}pp. `stage139_bpe_tail_cut_words_decision.json`.\n"
            )
            if "Stage 139 BPE-tail-cut" not in txt:
                if "**F85 dual-channel FREEZE:**" in txt:
                    txt = txt.replace(
                        "**F85 dual-channel FREEZE:**",
                        block + "\n**F85 dual-channel FREEZE:**",
                    )
                else:
                    txt = txt.rstrip() + "\n" + block
                replay.write_text(txt, encoding="utf-8")
        log("DONE 139")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
