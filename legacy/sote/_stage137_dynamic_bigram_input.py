"""
Stage 137 — dynamic bigram input tokens; target = next WORD (exact@1).

Hypothesis (post-136 discussion):
  One word = one fp/vector, but not one sense. Maybe sliding bigrams as
  *input* atoms give local context units while the gate stays next-word.

Protocol (hops OUT):
  Expand prefix w0..w_{t-1} to interleaved:
      [w0, bg(w0,w1), w1, bg(w1,w2), ..., w_{t-1}]
  where bg is a discrete bigram id (freq bank + UNK_BG).
  CE / argmax only over WORD vocab (not bigram rows).
  Bigram emb init: normalize(fp(a)+fp(b)) from frozen F85 encode.

Arms (same data / steps / d=256 / 2L):
  A word   — plain word-id prefix (control, recipe98-ish)
  B bigram — expanded input as above

Waits for Stage136 decision (pipeline). Corpus: TinyStories 100k (atom line).

Gate vs A on STORY ALL:
  PASS    +3pp with SEEN obj held (rel>=0.70, obj within -2pp of A)
  PARTIAL +1.5pp
  else PARITY / HARM if obj drops >3pp

Run:
  python _stage137_dynamic_bigram_input.py
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    CKPT,
    RES,
    Config,
    WordIdTransformer,
    _role,
    _warmup_then_constant,
    build_ts_repeat_mix,
    eval_id_capacity_suite,
    line_subkind,
    lines_to_pairs,
    load_foundation_85,
)
from _stage109_110_slot_baseline import REF100, ensure_100k, _subsample  # noqa: E402

DEC136 = RES / "stage136_wikitext_word_vs_bpe_decision.json"
LOG = RES / "_stage137_log.txt"
DEC = RES / "stage137_dynamic_bigram_input_decision.json"

FT_STEPS = 40_000
EVAL_EVERY = 2000
BATCH = 8
FAT_FRAC = 0.75
MAX_BIGRAMS = 25_000
MIN_BG_COUNT = 2
MAX_LEN_WORD = 16
MAX_LEN_BG = 32  # interleaved ~2x


def log(msg: str):
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        print(line, end="", flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), end="", flush=True)
    RES.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def wait_136(timeout_s=14 * 3600, poll=60):
    log(f"[wait] for {DEC136.name} ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if DEC136.exists():
            d = json.loads(DEC136.read_text(encoding="utf-8"))
            if d.get("verdict"):
                log(f"[wait] 136 done verdict={d.get('verdict')}")
                return d
        time.sleep(poll)
        if int(time.time() - t0) % 300 < poll:
            log(f"[wait] still ... {int(time.time()-t0)}s")
    raise TimeoutError("136 not ready")


class WordOutTransformer(nn.Module):
    """Like WordIdTransformer but emb may be larger than word vocab; logits = word rows only."""

    def __init__(
        self,
        n_word: int,
        n_emb: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert n_emb >= n_word
        self.n_word = n_word
        self.n_emb = n_emb
        self.pad_id = n_emb
        self.d_model = d_model
        self.max_len = max_len
        self.tok = nn.Embedding(n_emb + 1, d_model, padding_idx=self.pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)

    @torch.no_grad()
    def init_word_rows(self, word_fps: torch.Tensor):
        V, d_fp = word_fps.shape
        assert V == self.n_word
        d = min(self.d_model, d_fp)
        self.tok.weight[:V, :d] = word_fps[:, :d].to(dtype=self.tok.weight.dtype)
        self.tok.weight[:V] = F.normalize(self.tok.weight[:V], dim=-1)

    @torch.no_grad()
    def init_bigram_row(self, bg_id: int, fp: torch.Tensor):
        d = min(self.d_model, fp.numel())
        self.tok.weight[bg_id, :d] = fp[:d].to(dtype=self.tok.weight.dtype)
        self.tok.weight[bg_id] = F.normalize(self.tok.weight[bg_id], dim=-1)

    def forward(self, ids: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        bsz, tmax = ids.shape
        device = ids.device
        pos = torch.arange(tmax, device=device).unsqueeze(0).expand(bsz, -1)
        x = self.tok(ids) + self.pos(pos)
        mask = torch.triu(torch.ones(tmax, tmax, device=device, dtype=torch.bool), diagonal=1)
        h = self.tr(x, mask=mask, src_key_padding_mask=key_padding_mask)
        return F.linear(h, self.tok.weight[: self.n_word])

    def logits_last_from_batch(self, ids, key_padding_mask=None):
        return self.forward(ids, key_padding_mask=key_padding_mask)[:, -1, :]

    def logits_from_prefix(self, id_list):
        ids = torch.tensor([id_list[-self.max_len :]], dtype=torch.long, device=self.tok.weight.device)
        return self.forward(ids)[0, -1]

    def pred_id(self, id_list) -> int:
        return int(self.logits_from_prefix(id_list).argmax())


def build_bigram_bank(train_lines, max_bigrams=MAX_BIGRAMS, min_count=MIN_BG_COUNT):
    cnt = Counter()
    for ln in train_lines:
        ws = ln["words"]
        for i in range(len(ws) - 1):
            cnt[(ws[i], ws[i + 1])] += 1
    items = [(bg, c) for bg, c in cnt.items() if c >= min_count]
    items.sort(key=lambda x: (-x[1], x[0][0], x[0][1]))
    items = items[:max_bigrams]
    # ids: after word vocab; assigned later
    return items


def expand_prefix_ids(ws_prefix, stoi, bg_stoi, unk_bg_id, max_len):
    """Interleave word + bigram ids; truncate from the left to max_len."""
    if not ws_prefix:
        return []
    out = []
    for i, w in enumerate(ws_prefix):
        if w not in stoi:
            return []
        out.append(stoi[w])
        if i + 1 < len(ws_prefix):
            a, b = ws_prefix[i], ws_prefix[i + 1]
            out.append(bg_stoi.get((a, b), unk_bg_id))
    return out[-max_len:]


def make_examples(lines, stoi, bg_stoi, unk_bg_id, mode: str):
    """mode: word | bigram. Each ex: prefix_ids, target_word, role helpers."""
    exs = []
    for li, ln in enumerate(lines):
        ws = ln["words"]
        for t in range(1, len(ws)):
            gold = ws[t]
            if gold not in stoi:
                continue
            pref = ws[:t]
            if mode == "word":
                ids = [stoi[w] for w in pref if w in stoi][-MAX_LEN_WORD:]
            else:
                ids = expand_prefix_ids(pref, stoi, bg_stoi, unk_bg_id, MAX_LEN_BG)
            if not ids:
                continue
            fake = {"target_word": gold, "prefix_len": t}
            exs.append(
                {
                    "prefix_ids": ids,
                    "target_word": gold,
                    "target_id": stoi[gold],
                    "role": _role(fake, ws),
                    "line_i": li,
                }
            )
    return exs


def collate_prefix(examples, pad_id, device, max_len):
    prefs, targets = [], []
    for ex in examples:
        ids = list(ex["prefix_ids"][-max_len:])
        if not ids:
            continue
        prefs.append(ids)
        targets.append(int(ex["target_id"]))
    if not prefs:
        return None
    tmax = max(len(p) for p in prefs)
    bsz = len(prefs)
    ids = torch.full((bsz, tmax), pad_id, dtype=torch.long, device=device)
    pad_mask = torch.ones((bsz, tmax), dtype=torch.bool, device=device)
    for i, p in enumerate(prefs):
        ids[i, -len(p) :] = torch.tensor(p, dtype=torch.long, device=device)
        pad_mask[i, -len(p) :] = False
    tgt = torch.tensor(targets, dtype=torch.long, device=device)
    return ids, pad_mask, tgt


@torch.no_grad()
def eval_arm(model, hold_lines, stoi, bg_stoi, unk_bg_id, mode, words, train_lines, device):
    """Reuse suite for word mode; custom for bigram (pred via expanded prefix)."""
    if mode == "word":
        # monkey: WordOut with n_emb==n_word acts like WordId; use suite via wrapper
        class _Wrap:
            max_len = model.max_len
            def pred_id(self, id_list):
                return model.pred_id(id_list)
        # eval_id_capacity_suite expects WordIdTransformer API + surfaces list
        return eval_id_capacity_suite(model, hold_lines, train_lines, words, stoi, device)

    from collections import defaultdict

    roles = defaultdict(lambda: {"n": 0, "h1": 0})
    obj = {"n": 0, "h1": 0}
    for ln in hold_lines:
        ws = ln["words"]
        for t in range(1, len(ws)):
            gold = ws[t]
            if gold not in stoi:
                continue
            ids = expand_prefix_ids(ws[:t], stoi, bg_stoi, unk_bg_id, model.max_len)
            if not ids:
                continue
            pred = words[model.pred_id(ids)]
            ok = int(pred == gold)
            fake = {"target_word": gold, "prefix_len": t}
            role = _role(fake, ws)
            for name in (role, "ALL"):
                roles[name]["n"] += 1
                roles[name]["h1"] += ok
            if role == "right":
                obj["n"] += 1
                obj["h1"] += ok
    pack = {k: {"n": v["n"], "hit1": v["h1"] / max(v["n"], 1)} for k, v in roles.items()}
    return {
        "roles": pack,
        "obj": {"n": obj["n"], "hit1": obj["h1"] / max(obj["n"], 1)},
    }


def train_arm(
    *,
    tag: str,
    mode: str,
    words,
    stoi,
    word_fps,
    bg_items,
    stack,
    train,
    hold_seen,
    hold_story,
    device,
):
    n_word = len(words)
    if mode == "word":
        n_emb = n_word
        bg_stoi = {}
        unk_bg_id = None
        max_len = MAX_LEN_WORD
        model = WordOutTransformer(n_word, n_emb, 256, 4, 2, max_len, 0.1).to(device)
        model.init_word_rows(word_fps)
    else:
        # bigram ids start at n_word; last is UNK_BG
        bg_stoi = {bg: n_word + i for i, (bg, _c) in enumerate(bg_items)}
        unk_bg_id = n_word + len(bg_items)
        n_emb = unk_bg_id + 1  # words + bigrams + unk (pad = n_emb)
        max_len = MAX_LEN_BG
        model = WordOutTransformer(n_word, n_emb, 256, 4, 2, max_len, 0.1).to(device)
        model.init_word_rows(word_fps)
        # init bigram rows from fp(a)+fp(b)
        with torch.no_grad():
            for (a, b), bg_id in bg_stoi.items():
                fa = stack.w(a).detach()
                fb = stack.w(b).detach()
                model.init_bigram_row(bg_id, F.normalize(fa + fb, dim=-1))
            # unk = zeros then small noise
            model.tok.weight[unk_bg_id].normal_(std=0.02)
            model.tok.weight[unk_bg_id] = F.normalize(model.tok.weight[unk_bg_id], dim=-1)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"\n======== 137 {tag} mode={mode} V_word={n_word} n_emb={model.n_emb} params={n_params/1e6:.2f}M ========")

    fat = [ln for ln in train if ln.get("bucket") == "fat_train"] or train
    story = [ln for ln in train if ln.get("bucket") != "fat_train"] or train
    fat_ex = make_examples(fat, stoi, bg_stoi, unk_bg_id, mode)
    story_ex = make_examples(_subsample(story, 50000, 17), stoi, bg_stoi, unk_bg_id, mode)
    if not fat_ex:
        fat_ex = story_ex
    log(f"  train ex fat={len(fat_ex)} story_pool={len(story_ex)} bg_bank={len(bg_stoi)}")

    ev_seen = _subsample(hold_seen, 500, 1371)
    ev_story = _subsample(hold_story, 350, 1373)
    fin_seen = _subsample(hold_seen, min(len(hold_seen), 1000), 2371)
    fin_story = _subsample(hold_story, min(len(hold_story), 700), 2373)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rr = random.Random(137)
    best = {
        "story_all": -1.0,
        "obj": -1.0,
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    curve = []

    def snap(step):
        model.eval()
        s = eval_arm(model, ev_seen, stoi, bg_stoi, unk_bg_id, mode, words, train, device)
        st = eval_arm(model, ev_story, stoi, bg_stoi, unk_bg_id, mode, words, train, device)
        obj = s["obj"]["hit1"]
        rel = s["roles"].get("rel", {}).get("hit1", 0.0)
        sall = st["roles"].get("ALL", {}).get("hit1", 0.0)
        curve.append({"step": step, "obj": obj, "rel": rel, "story_all": sall})
        log(f"  [{tag}] step {step}: SEEN obj={100*obj:.1f}% rel={100*rel:.1f}% | STORY ALL={100*sall:.1f}%")
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
        batch_ex = [rr.choice(fat_ex) for _ in range(n_fat)]
        batch_ex += [rr.choice(story_ex) for _ in range(BATCH - n_fat)]
        packed = collate_prefix(batch_ex, model.pad_id, device, model.max_len)
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
    f_seen = eval_arm(model, fin_seen, stoi, bg_stoi, unk_bg_id, mode, words, train, device)
    f_story = eval_arm(model, fin_story, stoi, bg_stoi, unk_bg_id, mode, words, train, device)
    ck = CKPT / f"stage137_{tag}.pt"
    torch.save(
        {
            "model": best["state"],
            "mode": mode,
            "surfaces": words,
            "bg_list": [list(bg) for bg, _ in bg_items] if mode == "bigram" else [],
        },
        ck,
    )
    return {
        "tag": tag,
        "mode": mode,
        "params": n_params,
        "n_emb": model.n_emb,
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
    log(f"137 dynamic bigram input start {datetime.now(timezone.utc).isoformat()}")
    log("Input: interleaved word+bigram; target: next WORD. Hops OUT.")
    try:
        upstream = wait_136()
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
        stoi = {s: i for i, s in enumerate(words)}
        word_fps = F.normalize(torch.stack([stack.w(w).detach() for w in words], 0), dim=-1).to(device)
        bg_items = build_bigram_bank(train)
        log(f"[init] V_word={len(words)} bigram_bank={len(bg_items)} mix_keys={list(meta)[:6]}")

        word = train_arm(
            tag="word_ctrl",
            mode="word",
            words=words,
            stoi=stoi,
            word_fps=word_fps,
            bg_items=bg_items,
            stack=stack,
            train=train,
            hold_seen=hold_seen,
            hold_story=hold_story,
            device=device,
        )
        bigram = train_arm(
            tag="bigram_expand",
            mode="bigram",
            words=words,
            stoi=stoi,
            word_fps=word_fps,
            bg_items=bg_items,
            stack=stack,
            train=train,
            hold_seen=hold_seen,
            hold_story=hold_story,
            device=device,
        )

        lift = bigram["story_all"] - word["story_all"]
        obj_delta = bigram["seen_obj"] - word["seen_obj"]
        rel_ok = bigram["seen_rel"] >= 0.70 and word["seen_rel"] >= 0.70
        if obj_delta < -0.03:
            verdict = "HARM"
        elif rel_ok and lift >= 0.03 and obj_delta >= -0.02:
            verdict = "PASS"
        elif rel_ok and lift >= 0.015:
            verdict = "PARTIAL"
        else:
            verdict = "PARITY"

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol": "dynamic_bigram_input_next_word",
            "upstream_136": upstream.get("verdict"),
            "expand": "[w0, bg(w0,w1), w1, ..., w_last]; CE over words only",
            "bg_bank_size": len(bg_items),
            "word": word,
            "bigram": bigram,
            "story_lift_pp": lift,
            "seen_obj_delta_pp": obj_delta,
            "ref100": REF100,
            "verdict": verdict,
            "note": (
                "Tests local bigram atoms as INPUT only. Does not claim BPE polysemy fix; "
                "BPE segmentations are mostly surface-constant across senses."
            ),
        }
        DEC.write_text(json.dumps(out, indent=2), encoding="utf-8")
        (RES / f"stage137_{verdict}.txt").write_text(json.dumps(out, indent=2), encoding="utf-8")
        log(
            f"[137] {verdict} word_STORY={100*word['story_all']:.1f}% "
            f"bigram_STORY={100*bigram['story_all']:.1f}% lift={100*lift:+.1f}pp "
            f"obj_d={100*obj_delta:+.1f}pp"
        )

        replay = RES / "sote_v2_path_replay.md"
        if replay.exists():
            txt = replay.read_text(encoding="utf-8")
            block = (
                f"\n**Stage 137 dynamic bigram input:** {verdict} "
                f"word={100*word['story_all']:.1f}% bg={100*bigram['story_all']:.1f}% "
                f"lift={100*lift:+.1f}pp. `stage137_dynamic_bigram_input_decision.json`.\n"
            )
            if "Stage 137 dynamic" not in txt:
                if "**F85 dual-channel FREEZE:**" in txt:
                    txt = txt.replace(
                        "**F85 dual-channel FREEZE:**",
                        block + "\n**F85 dual-channel FREEZE:**",
                    )
                else:
                    txt = txt.rstrip() + "\n" + block
                replay.write_text(txt, encoding="utf-8")
        log("DONE 137")
        return 0
    except Exception:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
